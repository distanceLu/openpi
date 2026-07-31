from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.rl.pi05_action_head import Pi05GaussianActionHead
from openpi.rl.pi05_denoising import make_denoise_timesteps, normal_logprob, pi05_flow_step_mean_std
from openpi.rl.pi05_types import Pi05LogProbOutput, Pi05RolloutOutput
from openpi.rl.value_head import attach_pi05_value_head


@dataclass
class Pi05NestedMDPConfig:
    """Config for treating pi0.5 denoising as the inner MDP."""

    num_denoise_steps: int = 10
    sample_method: str = "flow_noise"
    noise_level: float = 0.5
    action_chunk: int | None = None
    action_dim: int | None = None
    deterministic_eval: bool = True
    learned_action_head: bool = True
    require_trainable_heads: bool = True
    require_value_head: bool = True
    value_head_input_dim: int | None = None
    value_head_hidden_dims: tuple[int, ...] = (1024, 512, 256)
    value_pool_mode: str = "masked_mean"


class Pi05NestedMDP:
    """Small wrapper that exposes pi0.5 as a two-level MDP.

    Outer MDP: environment transition o_t -> a_t -> r_t -> o_{t+1}.
    Inner MDP: denoising transition x_k -> x_{k+1} conditioned on o_t.

    The wrapped model is expected to be openpi.models_pytorch.pi0_pytorch.PI0Pytorch
    or a compatible subclass.  This class intentionally keeps tensors explicit so
    saved rollout data is easy to inspect while debugging.
    """

    def __init__(self, model: torch.nn.Module, config: Pi05NestedMDPConfig | None = None):
        self.model = model
        self.config = config or Pi05NestedMDPConfig()
        if not getattr(model, "pi05", False):
            raise ValueError("Pi05NestedMDP requires a pi0.5 model with model.pi05=True")
        if self.config.sample_method not in ("flow_ode", "flow_sde", "flow_cps", "flow_noise"):
            raise ValueError(f"Unknown pi0.5 sample_method: {self.config.sample_method}")
        if self.config.sample_method == "flow_noise":
            if not self.config.learned_action_head:
                raise ValueError("flow_noise requires learned_action_head=True")
            action_head = getattr(self.model, "rl_action_head", None)
            if not isinstance(action_head, Pi05GaussianActionHead):
                raise ValueError(
                    "model.rl_action_head must be defined as Pi05GaussianActionHead "
                    "before constructing Pi05NestedMDP with flow_noise"
                )
            action_head_parameters = list(action_head.parameters())
            if not action_head_parameters:
                raise ValueError("model.rl_action_head must have parameters")
            if self.config.require_trainable_heads and not all(parameter.requires_grad for parameter in action_head_parameters):
                raise ValueError("Every model.rl_action_head parameter must be trainable so transition std can learn")
        velocity_head_parameters = list(self.model.action_out_proj.parameters())
        if not velocity_head_parameters:
            raise ValueError("model.action_out_proj must have parameters")
        if self.config.require_trainable_heads and not all(parameter.requires_grad for parameter in velocity_head_parameters):
            raise ValueError("Every model.action_out_proj parameter must be trainable so transition mean can learn")
        if self.config.require_value_head:
            attach_pi05_value_head(
                self.model,
                input_dim=self.config.value_head_input_dim,
                hidden_dims=self.config.value_head_hidden_dims,
                pool_mode=self.config.value_pool_mode,
            )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample_inner_mdp(
        self,
        observation: Any,
        noise: torch.Tensor | None = None,
        mode: str = "train",
        return_values: bool = False,
    ) -> Pi05RolloutOutput:
        """Run the full inner denoising MDP and save every transition."""

        cache = self._build_prefix_cache(observation, train=False)
        state = cache["state"]
        bsz = state.shape[0]
        num_steps = self.config.num_denoise_steps

        if noise is None:
            noise = self.model.sample_noise(
                (bsz, self.model.config.action_horizon, self.model.config.action_dim),
                state.device,
            )
        x_t = noise

        timesteps = make_denoise_timesteps(num_steps, state.device)
        chains = [x_t]
        logprobs = []
        means = []
        stds = []
        velocities = []
        step_indices = []
        used_timesteps = []
        outer_value = self.compute_value_from_cache(cache) if return_values else None

        for step in range(num_steps):
            step_index = torch.full((bsz,), step, device=state.device, dtype=torch.long)
            timestep = timesteps[step].expand(bsz)
            delta = (timesteps[step] - timesteps[step + 1]).expand(bsz)

            velocity, suffix_out = self._velocity_from_cache(
                state=state,
                prefix_pad_masks=cache["prefix_pad_masks"],
                past_key_values=cache["past_key_values"],
                x_t=x_t,
                timestep=timestep,
            )
            mean, std = self._transition_distribution(
                x_t=x_t,
                velocity=velocity,
                suffix_out=suffix_out,
                timestep=timestep,
                delta=delta,
                sample_method=self._sample_method_for_mode(mode),
                timesteps=timesteps,
                step_index=step_index,
            )
            x_next = mean if self._deterministic(mode) else mean + self.model.sample_noise(mean.shape, mean.device) * std
            logprob = normal_logprob(x_next, mean, std)

            chains.append(x_next)
            logprobs.append(logprob)
            means.append(mean)
            stds.append(std)
            velocities.append(velocity)
            step_indices.append(step_index)
            used_timesteps.append(timestep)
            x_t = x_next

        return Pi05RolloutOutput(
            actions=chains[-1],
            chains=torch.stack(chains, dim=1),
            denoise_logprobs=torch.stack(logprobs, dim=1),
            denoise_means=torch.stack(means, dim=1),
            denoise_stds=torch.stack(stds, dim=1),
            denoise_timesteps=torch.stack(used_timesteps, dim=1),
            denoise_indices=torch.stack(step_indices, dim=1),
            velocities=torch.stack(velocities, dim=1),
            values=outer_value,
        )

    def recompute_logprobs(
        self,
        observation: Any,
        chains: torch.Tensor,
        denoise_indices: torch.Tensor | None = None,
        denoise_timesteps: torch.Tensor | None = None,
        return_values: bool = False,
    ) -> Pi05LogProbOutput:
        """Recompute current-policy logprobs for saved inner-MDP transitions."""

        if chains.dim() != 4:
            raise ValueError(f"chains must be [B, K+1, H, D], got {tuple(chains.shape)}")
        cache = self._build_prefix_cache(observation, train=False)
        state = cache["state"]
        bsz, chain_steps = chains.shape[:2]
        num_steps = chain_steps - 1
        timesteps = make_denoise_timesteps(num_steps, chains.device)

        logprobs = []
        means = []
        stds = []
        velocities = []
        values = self.compute_value_from_cache(cache) if return_values else None

        for inner_pos in range(num_steps):
            if denoise_indices is None:
                step_index = torch.full((bsz,), inner_pos, device=chains.device, dtype=torch.long)
            else:
                step_index = denoise_indices[:, inner_pos].to(device=chains.device, dtype=torch.long)
            x_cur = chains[:, inner_pos]
            x_next = chains[:, inner_pos + 1]
            if denoise_timesteps is None:
                timestep = timesteps[step_index]
            else:
                timestep = denoise_timesteps[:, inner_pos].to(device=chains.device, dtype=chains.dtype)
            delta = timesteps[step_index] - timesteps[step_index + 1]

            velocity, suffix_out = self._velocity_from_cache(
                state=state,
                prefix_pad_masks=cache["prefix_pad_masks"],
                past_key_values=cache["past_key_values"],
                x_t=x_cur,
                timestep=timestep,
            )
            mean, std = self._transition_distribution(
                x_t=x_cur,
                velocity=velocity,
                suffix_out=suffix_out,
                timestep=timestep,
                delta=delta,
                sample_method=self.config.sample_method,
                timesteps=timesteps,
                step_index=step_index,
            )
            logprobs.append(normal_logprob(x_next, mean, std))
            means.append(mean)
            stds.append(std)
            velocities.append(velocity)

        return Pi05LogProbOutput(
            logprobs=torch.stack(logprobs, dim=1),
            means=torch.stack(means, dim=1),
            stds=torch.stack(stds, dim=1),
            velocities=torch.stack(velocities, dim=1),
            values=values,
        )

    def _build_prefix_cache(self, observation: Any, train: bool) -> dict[str, Any]:
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(observation, train=train)
        images = [img.to(self.device) for img in images]
        img_masks = [mask.to(self.device) for mask in img_masks]
        lang_tokens = lang_tokens.to(self.device)
        lang_masks = lang_masks.to(self.device)
        state = state.to(self.device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        prefix_outputs, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_hidden_states = prefix_outputs[0]
        return {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_hidden_states": prefix_hidden_states,
            "past_key_values": past_key_values,
        }

    def _velocity_from_cache(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)

        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.model.config.action_horizon :]
        output_dtype = self.model.action_out_proj.weight.dtype
        suffix_out = suffix_out.to(dtype=output_dtype)
        velocity = self.model.action_out_proj(suffix_out)
        return velocity, suffix_out

    def _transition_distribution(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        suffix_out: torch.Tensor,
        timestep: torch.Tensor,
        delta: torch.Tensor,
        sample_method: str,
        timesteps: torch.Tensor,
        step_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match RLinf: flow dynamics define mean; flow_noise learns std."""

        if sample_method == "flow_noise":
            learned_std = self.model.rl_action_head(suffix_out)
        else:
            learned_std = None
        return pi05_flow_step_mean_std(
            x_t=x_t,
            velocity=velocity,
            timestep=timestep,
            delta=delta,
            method=sample_method,
            noise_level=self.config.noise_level,
            learned_std=learned_std,
            timesteps=timesteps,
            step_index=step_index,
        )

    def compute_value(self, observation: Any) -> torch.Tensor:
        """Compute outer-MDP critic V(s) from image-language prefix hidden states."""

        cache = self._build_prefix_cache(observation, train=False)
        return self.compute_value_from_cache(cache)

    def compute_value_from_cache(self, cache: dict[str, Any]) -> torch.Tensor:
        if not hasattr(self.model, "value_head"):
            raise ValueError("PPO/GAE requires model.value_head. Set require_value_head=True or attach one explicitly.")
        return self.model.value_head(cache["prefix_hidden_states"], cache["prefix_pad_masks"])

    def _sample_method_for_mode(self, mode: str) -> str:
        if mode == "eval" and self.config.deterministic_eval:
            return "flow_ode"
        return self.config.sample_method

    def _deterministic(self, mode: str) -> bool:
        return mode == "eval" and self.config.deterministic_eval

