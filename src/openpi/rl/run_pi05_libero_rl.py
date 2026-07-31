from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from openpi.rl.libero_adapter import Pi05LiberoRLAdapter
from openpi.rl.pi05_action_head import Pi05GaussianActionHead
from openpi.rl.pi05_nested_mdp import Pi05NestedMDP, Pi05NestedMDPConfig
from openpi.rl.pi05_trainer import Pi05PPOTrainConfig, Pi05PPOTrainer
from openpi.rl.rollout_collector import Pi05RolloutCollector, Pi05RolloutCollectorConfig
from openpi.rl.training_loop import Pi05RLLoopConfig, Pi05RLTrainingLoop


def configure_trainable_parameters(model: torch.nn.Module, full_model_training: bool) -> None:
    """Default to RLinf-style expert/action training; optionally unfreeze everything."""

    if full_model_training:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names = (
        "paligemma_with_expert.gemma_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "rl_action_head",
        "value_head",
    )
    for name, parameter in model.named_parameters():
        if any(module_name in name for module_name in trainable_names):
            parameter.requires_grad_(True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0.5 Libero RL training")
    parser.add_argument("--checkpoint-path", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors")
    parser.add_argument("--reference-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT")
    parser.add_argument("--assets-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/assets")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-iterations", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--sample-method", choices=["flow_ode", "flow_sde", "flow_cps", "flow_noise"], default="flow_noise")
    parser.add_argument("--full-model-training", action="store_true", help="Train the entire VLA instead of the RLinf-style action modules")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--fm-anchor-coef", type=float, default=0.01)
    parser.add_argument("--reference-kl-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--path-logprob-reduce", choices=["mean", "sum"], default="sum")
    parser.add_argument("--checkpoint-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/checkpoints/pi05_libero")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=args.checkpoint_path,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        device=args.device,
        default_prompt=args.prompt,
    )

    env = adapter.make_libero_env(task_suite_name=args.task_suite_name, task_id=args.task_id, seed=args.seed)
    model = adapter.load_pi05_model_from_checkpoint()
    model.rl_action_head = Pi05GaussianActionHead(
        input_dim=model.action_out_proj.in_features,
        action_dim=model.config.action_dim,
    ).to(device=next(model.parameters()).device, dtype=torch.float32)
    configure_trainable_parameters(model, args.full_model_training)
    nested_mdp = Pi05NestedMDP(
        model,
        Pi05NestedMDPConfig(
            num_denoise_steps=args.num_denoise_steps,
            sample_method=args.sample_method,
        ),
    )

    reference_model = adapter.load_pi05_model_from_checkpoint()
    reference_model.rl_action_head = Pi05GaussianActionHead(
        input_dim=reference_model.action_out_proj.in_features,
        action_dim=reference_model.config.action_dim,
    ).to(device=next(reference_model.parameters()).device, dtype=torch.float32)
    reference_model.rl_action_head.load_state_dict(model.rl_action_head.state_dict())
    reference_nested_mdp = Pi05NestedMDP(
        reference_model,
        Pi05NestedMDPConfig(num_denoise_steps=args.num_denoise_steps, sample_method=args.sample_method),
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Pi05PPOTrainer(
        nested_mdp=nested_mdp,
        optimizer=optimizer,
        config=Pi05PPOTrainConfig(
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            ppo_loss_mode="path",
            path_logprob_reduce=args.path_logprob_reduce,
            fm_anchor_coef=args.fm_anchor_coef,
            reference_kl_coef=args.reference_kl_coef,
            target_kl=args.target_kl,
        ),
        reference_mdp=reference_nested_mdp,
    )

    collector = Pi05RolloutCollector(
        env=env,
        nested_mdp=nested_mdp,
        env_obs_to_model_obs=adapter.env_obs_to_model_obs,
        action_to_env_action=adapter.action_to_env_action,
        config=Pi05RolloutCollectorConfig(
            rollout_steps=args.rollout_steps,
            mode="train",
            return_values=True,
            execute_action_chunk=True,
            stop_chunk_on_done=True,
        ),
    )

    loop = Pi05RLTrainingLoop(
        collector=collector,
        trainer=trainer,
        collate_observations=adapter.collate_observations,
        config=Pi05RLLoopConfig(
            total_iterations=args.total_iterations,
            rollout_steps=args.rollout_steps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        ),
    )

    if args.dry_run:
        print("Adapter, env, model, and trainer initialized successfully.")
        return

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    loop.run()


if __name__ == "__main__":
    main()
