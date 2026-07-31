# pi0.5 LIBERO RL：研发与运行日志

## [2026-07-25] 官方 pi05_libero JAX→PyTorch 基线与双环境胜率测试

### Action
- 从 `gs://openpi-assets/checkpoints/pi05_libero` 下载官方 JAX checkpoint，并使用仓库的 `examples/convert_jax_model_to_pytorch.py --config-name pi05_libero --precision bfloat16` 转成 PyTorch safetensors。
- 当前官方转换结果：
  ```text
  /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/model.safetensors
  /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/config.json
  /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/assets/physical-intelligence/libero/norm_stats.json
  ```
  `model.safetensors` 约 7.23GB；已验证 checkpoint 与 norm-stats 路径可解析。
- 重新审查 `run_pi05_openpi_server.py`、`run_pi05_libero_rollout.py` 和 `Pi05LiberoRLAdapter`。官方 `pi05_libero` 配置为 `extra_delta_transform=False`，评估客户端必须传 `--no-checkpoint-uses-extra-delta-transform`。
- 明确采用双环境：OpenPI 模型服务端使用项目 `uv` 环境和 GPU；LIBERO 客户端使用 `/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python` 和 CPU。二者通过 HTTP 通信，禁止用 OpenPI `uv` 环境直接启动 LIBERO（会缺 `robosuite`）。

### Reason
- `norm_stats.json` 只是框架无关的数据归一化统计，不是 JAX 权重；JAX→PyTorch 转换后的模型必须与其随附的官方 LIBERO stats 配套测试。
- 之前一次新基线命令在创建环境时因 `ModuleNotFoundError: robosuite` 失败，因此该次没有产生官方转换 checkpoint 的有效 0% 结果；日志中此前的 0% 属于更早的 rollout。
- 单环境胜率测试无需四卡模型并行。物理 GPU 4 用于 policy server，LIBERO 仿真保持 CPU，GPU 5–7 留空；`CUDA_VISIBLE_DEVICES=4` 后服务端内部的 `cuda`/`cuda:0` 对应物理 GPU 4。

### Impact / 正确基线配置
```text
model: asset_pi05/pytorch/model.safetensors
norm stats: asset_pi05/pytorch/assets/physical-intelligence/libero/norm_stats.json
config: pi05_libero
extra delta transform: False
LoRA: none
sampling: eval deterministic flow_ode
num denoise steps: 10
execute horizon: 5
max episode steps: 240
```

### 终端 1：OpenPI uv 服务端（物理 GPU 4）

启动前停止旧的 8000 端口 server，防止客户端连接到旧权重。

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

CUDA_VISIBLE_DEVICES=4 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
UV_CACHE_DIR=/tmp/lcx1-uv-cache \
uv run python \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_openpi_server.py \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/assets \
  --device cuda \
  --num-denoise-steps 10 \
  --sample-method flow_ode \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint-interval 0 \
  --tensorboard-log-dir ""
```

健康检查：

```bash
curl --fail http://127.0.0.1:8000/health
```

### 终端 2：LIBERO 客户端（CPU，rlinf_env）

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --mode eval \
  --episodes 1 \
  --steps 100 \
  --execute-horizon 5 \
  --max-episode-steps 240 \
  --seed 0 \
  --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05/pytorch/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO \
  --no-checkpoint-uses-extra-delta-transform \
  --video-path /mnt/data/lcx1/yiqinworkspace/openpi/videos/pi05_official_baseline
```

### 验证与注意事项
- 已在 `rlinf_env` 验证：`robosuite 1.4.1`、`gym 0.26.2`、`libero`、`httpx`、`requests` 可导入；OpenPI rollout 模块也可通过 `PYTHONPATH=.../openpi/src` 导入。
- 客户端 `--assets-dir` 会命中 `assets/physical-intelligence/libero/norm_stats.json`；实际读取到官方 action `q01` 前 7 维为 `[-0.747375, -0.796125, -0.9375, -0.115803, -0.16942972, -0.1945022, -1.0]`。
- 客户端参数中的 checkpoint 路径用于 adapter 配置/校验；真正加载模型的是服务端，所以服务端的三条路径是决定性配置。两端仍传同一路径以防回退到旧 RLinf assets。
- LIBERO 仿真主要运行在 CPU。`CUDA_VISIBLE_DEVICES=""` 防止客户端的 Torch/JAX/TensorFlow 初始化 GPU；EGL 仅用于离屏图形渲染，不等于 CUDA 模型计算。
- 第一次只跑 1 episode 并保存视频，检查画面方向、机械臂运动、gripper、动作范围和 `rollout_diagnostics`。输出 `policy_action_shape` 应为 `(10, 32)`，动作应无 NaN/Inf。
- 本次评估不传 `--lora-adapter-path`，测试的是纯官方少量 LIBERO 微调后的转换基线。

### Refs
- `asset_pi05/pytorch/`
- `examples/convert_jax_model_to_pytorch.py`
- `src/openpi/rl/run_pi05_openpi_server.py`
- `src/openpi/rl/run_pi05_libero_rollout.py`
- `src/openpi/rl/libero_adapter.py`
- `src/openpi/training/config.py` (`pi05_libero`, `extra_delta_transform=False`)

## [2026-07-23] PI0.5 LoRA SFT：HDF5 适配、八卡 DDP、修改记录与验证/正式指令

### 目标与数据流
- **目标**: 读取本地 LIBERO HDF5（image + language + state + action），转换成 OpenPI PI0.5 dataloader 支持的样本，在 `RLinf-Pi05-LIBERO-SFT` 上做 **LoRA SFT**（条件流匹配目标），为后续 RL 提供适配后的策略初始化。
- **数据流**:
  ```text
  LIBERO .hdf5/.h5
    → LiberoHDF5Dataset 输出 image / wrist_image / state / actions / prompt
    → LeRobotLiberoDataConfig.repack（键名对齐）
    → LiberoInputs（三相机 image/mask、state、actions、prompt）
    → Normalize（physical-intelligence/libero/norm_stats.json，quantile）
    → ResizeImages(224) + TokenizePrompt + PadStatesAndActions
    → Observation + actions
    → PI0Pytorch.forward（flow matching MSE）
    → 仅更新 LoRA 参数
  ```
- **有效 batch size（DDP）**:
  \[
  B_{\mathrm{global}} = B_{\mathrm{perGPU}} \times N_{\mathrm{GPU}} \times N_{\mathrm{accumulation}}
  \]
  例如 8 卡、`batch-size=1`、`gradient-accumulation-steps=2` → 全局有效 batch = 16。

### 新增/修改文件
| 路径 | 作用 |
|---|---|
| `src/openpi/sft/libero_hdf5_dataset.py` | 扫描 `**/*.{hdf5,h5}`；episode 帧索引；读 base/wrist 图、8D state、action horizon=10 chunk、language；返回 repack 期望的扁平键 |
| `src/openpi/sft/train_pi05_lora.py` | LoRA 注入、八卡 `torchrun`/NCCL DDP、HDF5 loader、flow-matching 训练循环、checkpoint、TensorBoard、断点续训 |
| `RLinf-Pi05-LIBERO-SFT/model.safetensors` | 初始权重（约 7.47GB，非空） |
| `RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json` | LIBERO state/actions 归一化统计 |

### 修改记录（按排查时间顺序）
1. **数据适配**: 原先误用 LeRobot `meta/info.json` 路径；改为 `LiberoHDF5Dataset` + OpenPI transforms。样本键必须是 `image` / `wrist_image` / `state` / `actions` / `prompt`（不要写成 `observation/image` 等，否则 `RepackTransform` 报 `KeyError: 'image'`）。
2. **norm_stats 路径**: `AssetsConfig(assets_dir=<ckpt>/physical-intelligence, asset_id=libero)`，实际文件为 `<ckpt>/physical-intelligence/libero/norm_stats.json`。
3. **`--tensorboard-dir` argparse**: 默认值为 `None` 时 `type(None)` 导致 `invalid NoneType value`；改为可选参数显式 `type=Path`（`default_prompt` 为 `str`）。
4. **LoRA 后 device 混用**: `inject_lora` 新建参数默认在 CPU，DDP 报 `{'cuda','cpu'}`；注入后再 `model.to(device)`。
5. **Transformers 访问 `.weight`**: SigLIP/PaliGemma 读 `q_proj.weight`；`LoRALinear` 增加 `weight`/`bias`/`in_features`/`out_features` 代理到 `base`。
6. **bf16 vs float LoRA**: 基础权重 bf16、LoRA 默认为 float32 时 linear 报 dtype 不一致；`lora_a/b` 用 `base.weight.new_empty/new_zeros` 继承 device/dtype。
7. **NCCL / barrier**: `init_process_group(..., device_id=device)`；去掉初始化阶段多余 barrier；环境变量改用 `TORCH_NCCL_ASYNC_ERROR_HANDLING`（弃用 `NCCL_ASYNC_ERROR_HANDLING`）。
8. **阶段日志**: 所有 rank 打 `Building HDF5` / `Creating model` / `Loading checkpoint` / `injecting LoRA` / `Wrapping DDP` / `first batch`，避免“无输出像卡住”只能 Ctrl+C。
9. **梯度累积 + DDP**: 非最后一次 micro-batch 使用 `model.no_sync()`，避免每步都 all-reduce。
10. **TensorBoard 隔离**: 日志目录为 `<tensorboard-dir>/<tensorboard-run-name>/`；`smoke` 与 `formal` 分开。说明：旧目录下若已有 event 文件**不改模型权重**，但可能混曲线；正式训练用新 run 名或新父目录。
11. **Checkpoint 内容**: `adapter_model.safetensors`（仅 LoRA）、`trainer_state.pt`（step、optimizer、batching 元数据）、`adapter_config.json`；默认 `save_interval=1000`；支持 `--resume`；中途改 batch size 会 warning 并继续，权重文件结构不变但数值不同。
12. **RL 衔接（说明）**: SFT 产出 adapter，不是完整 `model.safetensors`；RL 需 `base + adapter` 或先 merge。不必加载所有 step，通常选验证最好或最后一步。

### 已知“像卡住”的正常原因
- 8 进程各自读同一 ~7.47GB `model.safetensors`（总读量约 60GB 量级），共享盘上可能 **数分钟** 无 step 日志。
- 只有出现 `step=1` 与 `VERIFY/.../step-00000001/` 才算验证通过；日志末尾 `SignalException ... signal: 2` 是 **Ctrl+C**，不是训练算法错误。

### 静态检查状态
- `python -m py_compile`：`libero_hdf5_dataset.py`、`train_pi05_lora.py` 通过。
- `PYTHONPATH=src .venv/bin/python -m openpi.sft.train_pi05_lora --help` 可加载。
- IDE lint：通过。
- **端到端八卡一步训练**：以用户终端最新结果为准；通过前不要开 30000 步正式长跑。

---

### A. 单步验证指令（先跑这个）

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

# 若有残留 torchrun/train 进程，先 Ctrl+C 停干净
unset NCCL_ASYNC_ERROR_HANDLING

export UV_CACHE_DIR=/mnt/data/lcx1/yiqinworkspace/.uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  src/openpi/sft/train_pi05_lora.py \
  --initial-checkpoint /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --dataset-dir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/dataset/libero_spatial \
  --output-dir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO-VERIFY \
  --steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-6 \
  --warmup-steps 0 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --save-interval 1 \
  --log-interval 1 \
  --gradient-checkpointing \
  --no-tensorboard
```

**验证成功标准**:
- 日志出现：`Training loop ready; requesting first HDF5 batch`
- 日志出现：`step=1 loss=... lr=... grad_norm=...`
- 目录存在：
  ```text
  src/openpi/sft/SFT-PI05-LIBERO-VERIFY/
  ├── latest
  └── step-00000001/
      ├── adapter_model.safetensors
      ├── adapter_config.json
      └── trainer_state.pt
  ```

**验证 checkpoint 检查**:

```bash
python - <<'PY'
from pathlib import Path
root = Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO-VERIFY")
required = [
    root / "latest",
    root / "step-00000001" / "adapter_model.safetensors",
    root / "step-00000001" / "adapter_config.json",
    root / "step-00000001" / "trainer_state.pt",
]
for path in required:
    print(f"{path}: exists={path.exists()}, size={path.stat().st_size if path.exists() else 0}")
if not all(path.exists() and path.stat().st_size > 0 for path in required):
    raise SystemExit("Verification checkpoint is incomplete")
print("SFT one-step verification passed")
PY
```

---

### B. 正式八卡 SFT 指令（仅在 A 验证通过后使用）

全局有效 batch：`1 × 8 × 2 = 16`。输出与 TensorBoard 使用独立 `formal` run，避免与旧 event 混写。

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

unset NCCL_ASYNC_ERROR_HANDLING

export UV_CACHE_DIR=/mnt/data/lcx1/yiqinworkspace/.uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  src/openpi/sft/train_pi05_lora.py \
  --initial-checkpoint /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --dataset-dir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/dataset/libero_spatial \
  --output-dir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO \
  --tensorboard-dir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO/tensorboard \
  --tensorboard-run-name formal \
  --steps 30000 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-6 \
  --warmup-steps 500 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --save-interval 1000 \
  --log-interval 20 \
  --gradient-checkpointing \
  --tensorboard
```

**正式输出路径**:
```text
src/openpi/sft/SFT-PI05-LIBERO/
├── latest
├── step-00001000/ ...
├── step-00002000/ ...
└── tensorboard/
    └── formal/          # TensorBoard event 写入此处
```

**跑稳后可选加速**（仍保持全局 batch=16）:
- `--num-workers 2`（HDFERO 句柄更复杂，有问题时改回 0）
- 或 `--batch-size 2 --gradient-accumulation-steps 1`（每卡 2，需每卡剩余显存足够）

**断点续训**（同一 `output-dir`，且存在 `latest`）: 在正式命令末尾加 `--resume`。

**OOM 时**: 保持 `batch-size 1`；可减少参与 GPU 并提高 `gradient-accumulation-steps` 以维持全局 batch。

---

### C. TensorBoard 查看指令（正式训练开始后）

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi
export UV_CACHE_DIR=/mnt/data/lcx1/yiqinworkspace/.uv-cache

# 只看正式 formal run（推荐）
uv run tensorboard \
  --logdir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO/tensorboard/formal \
  --host 0.0.0.0 \
  --port 6006 \
  --load_fast=false

# 或查看整个 tensorboard 父目录（可对比多个 run）
# uv run tensorboard \
#   --logdir /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft/SFT-PI05-LIBERO/tensorboard \
#   --host 0.0.0.0 --port 6006 --load_fast=false
```

浏览器：`http://服务器IP:6006`；本地隧道：`ssh -L 6006:127.0.0.1:6006 用户@服务器` 后访问 `http://127.0.0.1:6006`。

关注曲线：`train/loss`、`train/learning_rate`、`train/gradient_norm`、`train/effective_batch_size`、`performance/steps_per_second`、`system/gpu_memory_*`。

**关于已有 `.../SFT-PI05-LIBERO/tensorboard` 下旧文件**: 不影响模型训练与 checkpoint 正确性；可能干扰曲线。正式训练请用 `--tensorboard-run-name formal` 写到子目录 `tensorboard/formal`，或换新 `--tensorboard-dir`。验证命令使用 `--no-tensorboard` 与 `SFT-PI05-LIBERO-VERIFY`，不写正式 TensorBoard。

### Refs
- `src/openpi/sft/train_pi05_lora.py`
- `src/openpi/sft/libero_hdf5_dataset.py`
- `src/openpi/training/config.py`（`LeRobotLiberoDataConfig` repack）
- `src/openpi/policies/libero_policy.py`（`LiberoInputs`）
- `RLinf-Pi05-LIBERO-SFT/`
- terminals：多卡启动与 dtype/device/repack 排查日志

---

## [2026-07-22] 增加 rollout 总成功率、逐环境步视频及按日期顺序归档
- **Action**: `run_pi05_libero_rollout.py` 新增跨 rollout 的 `RolloutSummary`，在正常结束、达到训练 iterations 或 `KeyboardInterrupt` 时统一输出 `rollout_summary`，包含 chunk 数、实际环境步数、reward 总和、完整 episode 数、成功 episode 数及成功率；未完成 episode 不计入成功率分母。新增可选 `--video-path`/`--video-fps`，从 reset 首帧开始记录每个真实 `env.step()` 后的 agent-view 帧，以 H.264 MP4 保存并在关闭 writer 后输出最终路径。视频路径统一解析为 `ROOT/YYYY-MM-DD/NNN_name.mp4`：日期目录按运行当天自动创建，每天序号从 `001` 独立递增；传入目录时自动使用 `libero_<suite>_task<id>_seed<seed>` 名称，传入 `.mp4` 时保留其 stem 但仍强制进入日期目录和添加序号。检查 `/mnt/data/lcx1/yiqinworkspace/openpi/videos` 时根目录无现有 MP4，因此本次没有文件需要迁移。
- **Reason**: LIBERO 的标准成功率应以已结束 episode 为单位，而不是按 action chunk 或环境步统计；流程中断时若把半个 episode 当失败会低估成功率。逐环境步视频可直接判断机器人不动、控制饱和、运动方向、gripper 和视觉方向问题。按日期与序号归档避免长时间评估/训练覆盖同名视频，也便于按实验先后回溯。
- **Impact**: 例如传入 `--video-path /mnt/data/lcx1/yiqinworkspace/openpi/videos`，当天首次运行会保存到 `videos/2026-07-22/001_libero_spatial_task0_seed0.mp4`；再次运行自动使用 `002_...mp4`，次日自动创建新日期目录并从 `001` 开始。也可传入 `--video-path .../videos/custom.mp4`，输出为当天目录下的 `NNN_custom.mp4`。长时间 `--iterations 0` 训练会持续写同一视频，正式训练建议关闭视频，仅在短 smoke test 或独立 eval 时启用。实现已通过 `py_compile`、CLI 加载、日期/递增序号行为测试和 linter。
- **Refs**: `src/openpi/rl/run_pi05_libero_rollout.py`; `/mnt/data/lcx1/yiqinworkspace/openpi/videos`; `imageio`; `imageio-ffmpeg`.

## [2026-07-22] 修复 LIBERO SFT 评估的任务条件、初始状态、状态编码、动作变换与成功判定
- **Action**: 系统排查原始 SFT 在 `libero_spatial/task 0` 确定性评估连续 300–600 步仍为零 reward/success 的输入输出链路。`Pi05LiberoRLAdapter.make_libero_env()` 现在读取 benchmark `task.language` 并注入每次 reset 的 `task_description`，按 seed 加载 `task_suite.get_task_init_states()` 的官方初始状态并执行 5 个零动作稳定物理系统；兼容 PyTorch 2.6 将 `torch.load` 默认改为 `weights_only=True` 后旧 LIBERO NumPy initial-state 文件无法加载的问题，仅在识别到该错误时用可信本地文件和 `weights_only=False` 回退。状态输入改为标准 8 维 `eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)`，避免原先直接拼 quaternion 后裁掉一个 gripper 维度。根据 checkpoint 自带 `norm_stats.json` 中 action/state 的互补绝对姿态统计，为 `RLinf-Pi05-LIBERO-SFT` 恢复前 6 维 `DeltaActions`/`AbsoluteActions` 变换；移除未经证实且会与 robosuite `IMAGE_CONVENTION` 重复的额外图像垂直翻转。rollout 每步直接调用 `env.check_success()`，成功时统一写入 `info["success"]`、稀疏 reward 和 termination；首步新增 prompt、observation keys、policy/env action 范围、gripper 和 info 诊断，并拒绝 NaN/Inf 动作。
- **Reason**: 原环境 observation 不携带语言字段，adapter 的空 `default_prompt` 导致 VLA 不知道任务；普通 `env.reset()` 未复现 LIBERO 官方 benchmark initial state；旧 state 拼接产生 `position(3)+quaternion(4)+单 gripper(1)`，与 SFT 的 8 维 proprio 分布不符；checkpoint normalization 中 `state[3]≈+2.97`、`action[3]≈-2.97` 表明该权重使用了额外 delta/absolute action 约定，而当前 `pi05_libero` 默认 `extra_delta_transform=False` 会把绝对姿态语义输出直接作为环境 delta action。与此同时，`ignore_done=True` 下仅查看 `info.get("success")` 不能可靠代表 LIBERO task predicate。value head 和 Gaussian RL action head 均不在 SFT safetensors 中，因此未训练 critic 的负 value 不是基础 SFT 失败根因。
- **Impact**: 后续基础评估必须重启 server/client 以避免新旧 transform 混用，先以 `--mode eval --execute-horizon 1`、不带 `--submit` 跑 10 步检查 `libero_env_initialized` 和 `rollout_diagnostics`，再跑 300 步。现在成功会明确表现为 `reward=1.0, success=1, terminated=True`；value 仍可为负。若输入、动作范围和官方 initial state 均正确但 300 步仍零成功，下一优先级是用同一 observation 对比官方 pi0.5 `sample_actions()` 与自定义 `Pi05NestedMDP` flow-ODE action，检查自定义去噪方程是否严格复现 SFT 推理。修改已通过 `py_compile` 和相关文件 linter。
- **Refs**: `src/openpi/rl/libero_adapter.py`; `src/openpi/rl/run_pi05_libero_rollout.py`; `src/openpi/policies/libero_policy.py`; `src/openpi/training/config.py`; `RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json`; `AcceRL/LIBERO/libero/libero/benchmark/__init__.py`; `AcceRL/LIBERO/libero/lifelong/evaluate.py`; terminals `1.txt`, `4.txt`.

## [2026-07-22] 四卡 DeepSpeed PPO 端到端跑通并增加 SFT 稀疏奖励排查流程
- **Action**: 完成 DeepSpeed ZeRO-2 四卡端到端 smoke test：修复入口 `state` 初始化顺序、value/Gaussian heads 与 pi0.5 `action_in_proj`/`action_out_proj` 的 BF16 输入对齐、分布式完整 buffer 写入错误，以及 `get_global_grad_norm()` 在 `engine.step()` 前读取导致首轮返回 `None` 的时序错误。梯度范数改为在 `engine.step()` 后读取，不再用伪造的 `0.0` 掩盖异常。四条 LIBERO chunk 已成功完成 `/sample`、多 rank backward/step、`/update` 和指标返回，首轮记录 `update_count=1`、真实全局 `grad_norm=440.2863`。明确 LIBERO `reward=0` 来自环境稀疏成功奖励，负 value 是未充分训练 critic 的无约束预测；若长期零奖励，应停止 PPO，重载原始 SFT 后用确定性逐步重规划和多个 seed 排查基础成功率。
- **Reason**: DeepSpeed BF16 会将主模型和附加 heads 权重转为 BFloat16，而原采样噪声、timestep 和 hidden features 部分仍为 Float32；ZeRO-2 的 `get_global_grad_norm()` 仅返回 `_global_grad_norm` 缓存，该缓存由 optimizer step 发布，因此 backward 后、step 前读取首轮必为 `None`。流程跑通后，连续零 reward 的主要风险转为 checkpoint/suite、语言、图像、动作反归一化或稀疏探索问题，而非 HTTP/DeepSpeed 基础设施。
- **Impact**: 当前四卡最小训练链路已验证，日志尾部 EGL context 析构错误发生在 rollout 完成后，不影响已成功的 PPO update。长期训练使用 `--steps 32 --iterations 0` 循环“采样→更新→重新采样”，TensorBoard 由 rank 0 写入；正式长跑前先验证原始 SFT 在 `libero_spatial/task 0`、多个 seed 下至少偶尔成功。`steps` 应不小于并最好整除 world size；当前 `ppo-minibatch-size=1` 为保守显存配置。
- **运行指令**:

```bash
# 终端 1：重载原始 SFT 并启动四卡 DeepSpeed server
cd /mnt/data/lcx1/yiqinworkspace/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_LINK_MODE=copy

uv run deepspeed --num_nodes 1 --num_gpus 4 \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_deepspeed_trainer.py \
  --deepspeed-config /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/deepspeed_zero2.json \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --checkpoint-dir /mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl \
  --monitor-log-dir /mnt/data/lcx1/yiqinworkspace/openpi/logs \
  --tensorboard-log-dir /mnt/data/lcx1/yiqinworkspace/openpi/runs/pi05_libero_rl \
  --ppo-minibatch-size 1 --device cuda --gamma 0.99 \
  --num-denoise-steps 10 --sample-method flow_noise --lr 1e-5 \
  --target-kl 0.03 --checkpoint-interval 10 --host 127.0.0.1 --port 8000

# 终端 2：先检查服务健康
curl --fail http://127.0.0.1:8000/health

# 终端 2：原始 SFT 确定性排查；不要添加 --submit
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 --task-suite-name libero_spatial --task-id 0 \
  --steps 600 --mode eval --execute-horizon 1 --gamma 0.99 --seed 0 --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO

# 将 --seed 依次改为 1、2、3、4，确认基础 SFT 是否至少偶尔 success=1。
# 只有基础评估有效后，才执行长期 PPO：将上方 rollout 改为以下参数
# --steps 32 --iterations 0 --mode train --gamma 0.99 --seed 0 --submit

# 终端 3：TensorBoard
cd /mnt/data/lcx1/yiqinworkspace/openpi
uv run tensorboard \
  --logdir /mnt/data/lcx1/yiqinworkspace/openpi/runs/pi05_libero_rl \
  --host 0.0.0.0 --port 6006
```
- **Refs**: `src/openpi/models_pytorch/pi0_pytorch.py`; `src/openpi/rl/value_head.py`; `src/openpi/rl/pi05_action_head.py`; `src/openpi/rl/pi05_nested_mdp.py`; `src/openpi/rl/pi05_trainer.py`; `src/openpi/rl/distributed_trainer.py`; `src/openpi/rl/run_pi05_deepspeed_trainer.py`; `src/openpi/rl/run_pi05_libero_rollout.py`; `src/openpi/rl/deepspeed_zero2.json`; terminals `1.txt`, `4.txt`。

## [2026-07-22] 修复 DeepSpeed BF16 推理与分布式 buffer 构建
- **Action**: 修复 DeepSpeed BF16 模式下 critic 和 Gaussian action head 输入仍为 Float32 导致的 Linear dtype 冲突，两个 head 现在将输入对齐到自身参数的 device/dtype；修复 `distributed_trainer.update_from_payload()` 构建完整 rollout 时误写未定义局部 `buffer`、应写入 `full_buffer` 的错误；保留此前入口脚本 `state` 初始化顺序修复。
- **Reason**: 最新 `/sample` HTTP 500 的服务端根因是 `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`，不是 LIBERO、EGL 或 HTTP 错误。修复 sample 后，首次 `/update` 还会命中完整 buffer 写入变量错误，因此一并修复以继续推进端到端流程。
- **Impact**: `/sample` 的 value/action heads 可适配 DeepSpeed BF16；分布式 update 可正确先构建完整 SMDP buffer，再按 rank 切分。已通过 RL 目录 `compileall` 和相关文件 linter；GPU 端到端 smoke test 仍需重启四卡 server 后运行 `--steps 4 --iterations 1`，以运行时日志确认一次 PPO backward/step。
- **Refs**: `src/openpi/rl/value_head.py`; `src/openpi/rl/pi05_action_head.py`; `src/openpi/rl/distributed_trainer.py`; `src/openpi/rl/run_pi05_deepspeed_trainer.py`; terminals `1.txt` and `4.txt`。

## [2026-07-22] 修复 DeepSpeed launcher 的 `--local_rank` 参数兼容
- **Action**: 在 `run_pi05_deepspeed_trainer.py` 的 argparse 中增加 DeepSpeed launcher 自动注入的 `--local_rank` 参数；设备绑定优先使用命令行 `--local_rank`，否则回退到环境变量 `LOCAL_RANK`，再由 `torch.cuda.set_device(local_rank)` 绑定对应 GPU。修复了四卡启动时四个 rank 都在 argparse 阶段退出的错误，并使用 Python `compileall` 完成语法检查。
- **Reason**: DeepSpeed launcher 实际启动命令会自动追加 `--local_rank=0..3`。原入口只读取环境变量但没有在 argparse 中声明该选项，导致 `unrecognized arguments: --local_rank=N`，四个子进程均返回码 2，尚未进入 NCCL 或模型初始化。
- **Impact**: 使用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 时，`--local_rank=0..3` 分别映射到物理 GPU 0–3；rank 0 启动 HTTP coordinator，rank 1–3 进入 worker loop。四卡 smoke test 至少使用 `--steps 4`，确保每个 rank 至少有一个 transition；先验证 `--steps 4 --iterations 1`，再扩大 rollout。启动命令如下：

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_LINK_MODE=copy

uv run deepspeed --num_nodes 1 --num_gpus 4 \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_deepspeed_trainer.py \
  --deepspeed-config /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/deepspeed_zero2.json \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --checkpoint-dir /mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl \
  --monitor-log-dir /mnt/data/lcx1/yiqinworkspace/openpi/logs \
  --tensorboard-log-dir /mnt/data/lcx1/yiqinworkspace/openpi/runs/pi05_libero_rl \
  --ppo-minibatch-size 1 \
  --device cuda --gamma 0.99 --num-denoise-steps 10 \
  --sample-method flow_noise --lr 1e-5 --target-kl 0.03 \
  --checkpoint-interval 10 --host 127.0.0.1 --port 8000
```

LIBERO smoke test：

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python \
  /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 --task-suite-name libero_spatial --task-id 0 \
  --steps 4 --iterations 1 --mode train --gamma 0.99 --seed 0 --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO --submit
```
- **Refs**: `src/openpi/rl/run_pi05_deepspeed_trainer.py`; `src/openpi/rl/distributed_trainer.py`; terminal `1.txt:833-999`; `src/openpi/rl/deepspeed_zero2.json`。


## [2026-07-22] 排查首次 PPO update 的 CUDA OOM
- **Action**: 根据终端日志定位到 LIBERO rollout 已正常完成，`/sample` 全部返回 200，但首次 `/update` 在 `Pi05NestedMDP.recompute_logprobs()` 的 Gemma attention 申请额外 490 MiB 时 OOM；原因是单卡进程已占用约 47.37 GiB PyTorch 显存，且同一 GPU 上还有两个约 45.79 GiB 的残留进程。将 server 的 PPO minibatch 默认值收紧为 1，避免原先将 128 条 transition 一次性放入反向图；同时确认这不能替代清理 GPU 残留进程。
- **Reason**: 这次 500 不是 LIBERO 环境、HTTP 协议或 rollout 数据错误，而是 server 训练阶段显存不足。日志中的 GPU 0 对应 `CUDA_VISIBLE_DEVICES` 映射后的第一张卡；`nvidia-smi` 当前又无法正常通信，因此不能由本会话替用户终止残留作业。
- **Impact**: 重启 server 前必须确认物理 GPU 4 上没有旧 OpenPI/DeepSpeed 进程；验证时使用 `--ppo-minibatch-size 1 --steps 2`，正式训练再逐步提高 minibatch。已用系统 Python 对 `src/openpi/rl` 完成 `compileall`；`uv run` 编译检查因 `/home/lcx1/.cache/uv` 所在文件系统只读而无法执行。
- **Refs**: `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/run_pi05_openpi_server.py`; `src/openpi/rl/pi05_nested_mdp.py`; terminals `1.txt` and `4.txt`。


## [2026-07-22] 接入 DeepSpeed ZeRO-2 多卡训练基础设施
- **Action**: 增加 `deepspeed` 项目依赖和 `src/openpi/rl/deepspeed_zero2.json`，为 PPO trainer 增加可选 DeepSpeed engine，使用 `engine.backward()`/`engine.step()`；server 增加 `--deepspeed-config` 参数，并保留 rank-0 HTTP 服务约束。
- **Reason**: 原 OpenPI RL server 是单进程单卡普通 PyTorch optimizer，设置 `CUDA_VISIBLE_DEVICES=4,5,6,7` 并不会形成多卡训练。DeepSpeed ZeRO-2 需要所有训练 rank 共享 batch 并参与梯度同步，不能让多个 rank 同时监听同一个 HTTP 端口。
- **Impact**: DeepSpeed engine 接入训练更新路径，但当前 HTTP server 仍是单 rank 服务入口；正式多卡启动还需要使用 rank-0 trainer/service wrapper，将收到的 rollout batch 广播给其他 rank 后再执行 PPO update。直接用 `deepspeed --num_gpus=4 run_pi05_openpi_server.py` 不可用，因为非零 rank 会被明确拒绝，避免端口冲突和 NCCL 死锁。
- **Refs**: `src/openpi/rl/pi05_trainer.py`; `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/run_pi05_openpi_server.py`; `src/openpi/rl/deepspeed_zero2.json`; `pyproject.toml`。


## [2026-07-21] 修复 reference NestedMDP 的只读参数检查，并更新双终端启动指令
- **Action**: 将 `Pi05NestedMDP` 的“必须可训练”检查拆分为 `require_trainable_heads` 开关；训练侧仍要求 `rl_action_head` 与 `action_out_proj` 可训练，但 reference model 构建时关闭该检查，避免 `reference_model.parameters()` 全冻结后在 `flow_noise` 路径下直接抛出 `ValueError`。同时把运行指令维护为“OpenPI 服务端 + LIBERO 客户端”双终端模式，OpenPI 使用项目 `uv` 环境，LIBERO 使用 `/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env`。
- **Reason**: 现有报错来自 `openpi_policy_server.py` 在构造 reference nested MDP 时冻结了所有参数，却仍复用了“要求头部参数可训练”的校验；这会阻断服务端启动，和训练本身无关。另一方面，LIBERO 与 OpenPI 依赖冲突，必须拆成两个终端、两个进程分别运行，避免环境污染。
- **Impact**: 训练启动顺序固定为：先在终端 1 启动 OpenPI policy server，再在终端 2 启动 LIBERO rollout client；如果只想先做 smoke test，可将 `--iterations 2 --steps 2` 跑通后再切换到正式训练。OpenPI 服务端建议继续绑定 `127.0.0.1:8000`，LIBERO 通过 HTTP 与其交互；GPU 4-7 可通过 `CUDA_VISIBLE_DEVICES=4,5,6,7` 暴露给服务端，但当前代码仍按单进程单 device 执行，`--device cuda:0` 会对应到可见设备里的第一张卡。
- **可执行指令**:

```bash
# 终端 1：OpenPI 服务端（项目 uv 环境）
cd /mnt/data/lcx1/yiqinworkspace/openpi
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_openpi_server.py \
  --device cuda:0 \
  --gamma 0.99 \
  --num-denoise-steps 10 \
  --sample-method flow_noise \
  --lr 1e-5 \
  --target-kl 0.03 \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --checkpoint-dir /mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl \
  --checkpoint-interval 10 \
  --host 127.0.0.1 \
  --port 8000

# 终端 2：LIBERO 采样与回传（rlinf_env 环境）
cd /mnt/data/lcx1/yiqinworkspace/openpi
export PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --steps 128 \
  --iterations 0 \
  --mode train \
  --gamma 0.99 \
  --seed 0 \
  --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO \
  --submit
```

- **Refs**: `src/openpi/rl/pi05_nested_mdp.py`; `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/run_pi05_openpi_server.py`; `src/openpi/rl/run_pi05_libero_rollout.py`; `cursor.md`。


## [2026-07-18] 对齐 RLinf 的 flow-noise Gaussian transition
- **Action**: 将生产训练路径对齐为 RLinf 风格的多步 flow Gaussian policy：pi0.5 action expert 与 `action_out_proj` 预测 velocity，flow dynamics 根据当前状态、timestep 与 velocity 计算下一步 transition mean；`Pi05GaussianActionHead` 从 `suffix_out` 预测 state-dependent `log_std`，经 clamp 和 `exp` 得到正标准差；每一步按 \(x_{k+1}=\mu_{\theta,k}+\sigma_{\theta,k}\epsilon_k\) 采样并计算条件 Gaussian log-prob，重复去噪后以最终 \(x_0\) 作为 action chunk。训练默认 `flow_noise`，确定性评估切换为 `flow_ode`；rollout 与 PPO recompute 共用同一 transition distribution。未修改草稿 `src/openpi/rl/continuous_action_head.py`。
- **Reason**: RLinf 的主 pi0.5 PPO 并非用独立 `mean_layer` 直接替换 flow mean，而是通过可学习 velocity 参数化 transition mean，并在 `flow_noise` 模式由神经网络学习 std。数学 flow 公式不会阻断梯度：PPO log-prob 可通过 mean 回传至 `action_out_proj`、action expert 和 VLA，也可通过 std 回传至 noise head。该设计同时保留 pretrained flow dynamics、条件 Gaussian 密度和 PPO old/new log-prob ratio。
- **Impact**: 当前生产策略为 \(\mu_\theta=f_{\mathrm{flow}}(x_t,v_\theta,t,\Delta t)\)、\(\sigma_\theta=\exp(\operatorname{clip}(\operatorname{LogStdHead}(h_t),-5,2))\)，而不是纯 MLP Gaussian actor。`Pi05NestedMDP` 要求 `sample_method="flow_noise"`、所有 `rl_action_head` 参数可训练，并要求所有 `action_out_proj` 参数可训练，否则立即报错；noise head 默认 `init_std=0.1`，避免初始 \(\sigma=1\) 破坏 flow trajectory。reference policy 使用同构 head，训练模型继续全量优化。`flow_sde`/`flow_cps` 的公式 std 和 `flow_ode` 的零 std 仍保留在通用 denoising helper 中，但 learned-head 训练路径固定使用 `flow_noise`。
- **Refs**: `src/openpi/rl/pi05_action_head.py`; `src/openpi/rl/pi05_nested_mdp.py`; `src/openpi/rl/pi05_denoising.py`; `src/openpi/rl/pi05_trainer.py`; `src/openpi/rl/pi05_losses.py`; `src/openpi/rl/run_pi05_libero_rl.py`; RLinf reference `rlinf/models/embodiment/openpi/openpi_action_model.py::sample_mean_var_val`; validation: production RL modules passed linter and `py_compile`.

## [2026-07-15 15:11] GPU 真实训练前置审查与资源检查
- **Action**: 对当前节点执行 `nvitop --once` 与 `nvidia-smi`；两者均报告 NVIDIA driver/NVML 不可用。完成从 CPU 协议验证切换到多 GPU 真实训练前的代码与实验缺口审查。
- **Reason**: 当前 server 是单进程单卡式 PPO 实现，尚不能将“8 张 GPU”自动转化为有效训练吞吐；真实训练前必须先确定调度节点、显存容量、进程拓扑、checkpoint 恢复和可靠评估机制。
- **Impact**: 当前节点不能启动 CUDA 训练或判断其他用户的 GPU 占用。获得 GPU 节点后先执行资源检查与单卡 smoke run，再实现/验证多 GPU 数据并行。真实实验前仍需补齐 vectorized LIBERO rollout、训练/评估隔离、可恢复 checkpoint、实验记录和长期稳定性测试；不要直接用 8 卡启动现有 server。
- **Refs**: `nvitop --once` -> `NVML ERROR: Driver Not Loaded`; `nvidia-smi` -> driver communication failure; `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/run_pi05_openpi_server.py`; `src/openpi/rl/run_pi05_libero_rollout.py`。


## [2026-07-15 15:06] 恢复可追溯的 cursor.md 维护方式
- **Action**: 将本文档从覆盖式“当前说明”恢复为持续维护的研发日志；保留当前架构、SMDP 语义、环境路径和双终端命令，并在项目规则中固化原子追加、逆时间序、四字段日志格式。
- **Reason**: 覆盖式精简会丢失关键决策、历史运行结论和环境排障上下文，无法支持 Pi0.5/LIBERO 实验复现、回归定位和后续方案比较。
- **Impact**: 后续每次核心算法、配置、bug、性能、架构或实验结论变更都会立即在本文档顶部追加记录；已有信息不再被批量删除或以摘要替代。
- **Refs**: `.cursor/rules/robot-vla-openpi.mdc`; `cursor.md`。

## [2026-07-15 15:06] 修正 SMDP rollout、PPO 输入与 HTTP 更新协议
- **Action**: 将 outer transition 固定为一个 action-chunk SMDP transition，保存 discounted chunk reward、`terminated`、`truncated`、`duration`、`value`、`next_value`；更新 GAE、buffer、server `/update` schema 和 LIBERO HTTP client。训练默认执行完整 chunk；评估支持 `--execute-horizon=1/2` 的滚动重规划。
- **Reason**: 原实现对连续执行的 action chunk 仍按单个环境 step 用 \(\gamma\) 做 GAE，且混淆 `done`、真实终止和时间截断；这会造成 value bootstrap、credit assignment 与 PPO 时间尺度不一致。单步评估 action 还需要保持 chunk transform 的时间轴。
- **Impact**: PPO target 改为 \(R_t^{(m)}+\gamma^m(1-\mathrm{terminated})V(o_{t+m})\)，truncation 可 bootstrap 但 trace 不跨 reset；HTTP client/server 使用同一 `--gamma`。历史 checkpoint 不含 rollout buffer，仍可加载模型/optimizer；旧 client 的 `dones` update 请求不再兼容，必须同时升级 server 与 client。
- **Refs**: `src/openpi/rl/returns.py`; `src/openpi/rl/rollout_buffer.py`; `src/openpi/rl/rollout_collector.py`; `src/openpi/rl/run_pi05_libero_rollout.py`; `src/openpi/rl/http_protocol.py`; `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/run_pi05_openpi_server.py`; SMDP GAE formula above.

## [2026-07-15 15:06] 增加 rollout 边界与数值安全校验
- **Action**: 修复 `Iterable` 类型导入；对 SMDP scalar、正整数 duration、二值终止 mask、完整 `old_velocities`、HTTP 字段长度、非空 observation tensor、`gamma`/`gae_lambda` 范围增加校验；统一 policy value 为单环境标量；禁止 `--mode eval --submit`；对 PPO KL 使用与 ratio 一致的 log-ratio clamp。
- **Reason**: 缺失导入会阻断 adapter 加载；shape 不受控时 buffer 会把单环境 value 错展平，FM anchor 可能错位；评估数据写入训练会污染 PPO；未一致 clamp 的 KL 指标会被极端 logprob 数值破坏。
- **Impact**: 非法 rollout 和配置将尽早失败并给出明确异常，避免静默训练错误。已通过 `compileall`、变 duration/truncation GAE、buffer scalar、HTTP SMDP payload 序列化测试；真实双终端新协议仍需按本文档命令在同一网络命名空间执行最小两轮验证。
- **Refs**: `src/openpi/rl/libero_adapter.py`; `src/openpi/rl/returns.py`; `src/openpi/rl/rollout_buffer.py`; `src/openpi/rl/run_pi05_libero_rollout.py`; `src/openpi/rl/openpi_policy_server.py`; `src/openpi/rl/pi05_losses.py`。


## 架构

`openpi/src/openpi/rl/` 实现 Pi0.5 的双层 PPO：

- **inner MDP**：pi0.5 flow/diffusion 去噪链，保存 `chains`、`old_logprobs`、`denoise_indices`、`denoise_timesteps`、`velocities`。
- **outer SMDP**：一条 transition 对应一个模型预测 action chunk。模型动作是 `[B, action_horizon, action_dim]`，LIBERO 实际动作是 `[executed_steps, env_action_dim]`；默认 `action_horizon=10`、`action_dim=32`、`env_action_dim=7`。
- critic 位于图像+语言 prefix hidden states 上，PPO 使用 path-level + element-level loss，可选 FM anchor 和 frozen-reference KL。
- OpenPI server 负责 `/sample`、`/update`、PPO backward/step/checkpoint；LIBERO client 独立环境通过 HTTP 采样和提交 rollout。

## SMDP 语义（已修复）

每个 chunk transition 保存：

```text
reward       = sum_{j=0}^{m-1} gamma^j r_j
terminated   = 真实成功/失败等终止；不 bootstrap
truncated    = 时间上限/外部截断；可 bootstrap，但不跨 reset 传递 GAE
duration=m   = 实际执行的 LIBERO action 数，1 <= m <= action_horizon
value        = V(o_t)
next_value   = V(o_{t+m})；仅 terminated 时为 0
```

GAE 使用：

\[
\delta_t=R_t^{(m)}+\gamma^m(1-\mathrm{terminated}_t)V(o_{t+m})-V(o_t)
\]

\[
A_t=\delta_t+(\gamma\lambda)^m(1-\mathrm{episode\_end}_t)A_{t+1}
\]

这避免了“执行完整 chunk 但仍按单步 \(\gamma\) 计算”的时间尺度错误。所有 SMDP scalar、duration、HTTP 字段长度和 `old_velocities` 完整性均在 buffer/server 校验。

## 执行策略

- **训练（推荐）**：`--mode train --submit` 且不传 `--execute-horizon`，执行完整 chunk；这是与完整 diffusion action likelihood 一致的 SMDP PPO 设置。
- **评估**：`--mode eval --execute-horizon 1` 或 `2`，仅执行 chunk 前缀，然后基于最新视觉观测重采样。未执行 tail actions 会丢弃，这是 receding-horizon 控制的预期行为。
- 评估禁止 `--submit`，避免将评估轨迹用于 PPO 更新。

## 环境与路径

```text
OpenPI server env: /mnt/data/lcx1/yiqinworkspace/openpi/venv-openpi-libero
LIBERO client env: /mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env
LIBERO repo: /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO
checkpoint: /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors
assets: /mnt/data/lcx1/yiqinworkspace/openpi/assets
checkpoint output: /mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl
```

当前节点无可用 CUDA 时两个终端都传 `--device cpu`；GPU 节点仅 OpenPI server 使用 `CUDA_VISIBLE_DEVICES=0 --device cuda`，LIBERO client 保持 CPU。

## 运行

### 终端 1：启动 OpenPI server

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
/mnt/data/lcx1/yiqinworkspace/openpi/venv-openpi-libero/bin/python \
/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_openpi_server.py \
  --device cpu \
  --gamma 0.99 \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --checkpoint-dir /mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl \
  --checkpoint-interval 10 \
  --host 127.0.0.1 \
  --port 8000
```

等待 `Uvicorn running on http://127.0.0.1:8000`。

### 终端 2：两轮 SMDP PPO 验证

`--gamma` 必须与 server 一致。未设置 `--execute-horizon` 即执行完整预测 chunk。

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python \
/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --steps 2 \
  --iterations 2 \
  --mode train \
  --gamma 0.99 \
  --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO \
  --submit
```

预期 server 显示 `/sample`、`/update` 均为 HTTP 200，并打印 `pi05_rl/policy_loss`、`value_loss`、`total_loss`、`update_count`。

### 终端 2：滚动重规划评估

```bash
cd /mnt/data/lcx1/yiqinworkspace/openpi

PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python \
/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl/run_pi05_libero_rollout.py \
  --server-url http://127.0.0.1:8000 \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --steps 32 \
  --mode eval \
  --execute-horizon 1 \
  --gamma 0.99 \
  --device cpu \
  --checkpoint-path /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors \
  --reference-dir /mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT \
  --assets-dir /mnt/data/lcx1/yiqinworkspace/openpi/assets \
  --libero-repo-dir /mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO
```

## 已验证

```bash
PYTHONPATH=/mnt/data/lcx1/yiqinworkspace/openpi/src \
/mnt/data/lcx1/yiqinworkspace/openpi/venv-openpi-libero/bin/python \
  -m compileall -q /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/rl
```

已通过 SMDP GAE（变 duration、truncation bootstrap、跨 reset trace 截断）、buffer scalar 校验和 HTTP SMDP schema 测试。双终端 HTTP 流程曾在旧协议下完成 rollout/PPO update；本次协议升级后需使用上方两条命令在同一节点重新跑最小两轮验证。
