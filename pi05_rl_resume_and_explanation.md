# Pi0.5 VLA 强化学习项目：简历、实现原理与面试说明

> 本文档是该项目的唯一简历与面试说明文档。内容按“简历表述—问题背景—算法设计—代码实现—设计取舍—面试回答”组织，避免在多个章节重复同一结论。
>
> 真实性边界：当前代码支持双层 MDP/SMDP、Flow-Gaussian policy、去噪轨迹重算、PPO、critic、参考策略约束、LIBERO 适配和 HTTP 训练链路；在没有完整多随机种子实验前，不写“显著提升成功率”“稳定收敛”“超过基线”或“分布式强化学习”。

---

## 一、推荐简历版本

### Pi0.5 VLA 模型的双层 MDP 强化学习微调

**项目背景：** 面向 LIBERO 机器人操作任务，为预训练 Pi0.5 Vision-Language-Action 模型搭建 on-policy PPO 强化学习微调框架。针对 Flow Matching 多步动作生成无法像普通 Gaussian Actor 一样直接计算最终动作概率，以及 action chunk 连续执行导致策略决策与环境反馈时间尺度不一致的问题，将去噪生成过程建模为内层 MDP，将 action chunk 的环境执行建模为外层变时长 SMDP，利用任务奖励优化连续动作生成策略。

**技术栈：** Python / PyTorch / Transformer / VLA / Flow Matching / Gaussian Policy / PPO / GAE / SMDP / LIBERO / OpenPI / FastAPI / HTTP

- **Flow Matching 策略概率建模：** 将 Pi0.5 多步 Flow Matching 去噪过程展开为内层 MDP，保留预训练模型预测的 velocity field，并根据当前带噪动作、velocity、时间步和去噪步长构造下一状态的转移均值；在 action expert 的 suffix hidden states 上增加状态相关的对角 Gaussian 标准差预测头，使每一步去噪转移都具备可采样、可计算 log-prob 的条件策略分布。

- **历史去噪轨迹保存与当前策略概率重算：** rollout 时保存初始噪声、完整 denoising chain、各步旧策略 log-prob、mean/std、去噪 timestep/index 及 velocity；PPO 更新时固定 behavior policy 采样得到的历史 chain，复用视觉—语言 prefix KV cache，通过当前模型重新计算各步转移分布，并评估历史下一去噪状态的 current log-prob，保证新旧策略概率对应同一采样路径。

- **Action-chunk SMDP 与变时长 GAE：** 将一次 action chunk 的实际执行视为一条外层 SMDP transition，按照实际执行步数 \(m\) 聚合 chunk 内折扣奖励，并使用 \(\gamma^m\) 计算 value bootstrap、使用 \((\gamma\lambda)^m\) 递推 GAE；区分任务真实终止和时间限制截断，使 truncation 可以进行价值 bootstrap，但 advantage trace 不跨环境 reset 传播。

- **多粒度 PPO 目标与训练约束：** 实现去噪元素/步骤级 PPO 与完整去噪路径级 PPO，并支持单独使用或加权组合；将外层 action-chunk advantage 广播至生成该动作序列的内层去噪步骤，同时加入 critic value loss、Flow velocity MSE anchor、冻结参考策略 KL、log-ratio 数值裁剪、gradient norm clipping 和 target-KL early stopping，并支持可配置 entropy regularization。

- **多模态 Critic 与 LIBERO 环境适配：** 基于 Pi0.5 图像—语言 prefix hidden states 进行 masked mean pooling，并通过 MLP value head 输出外层 observation value；完成第三人称图像、腕部图像、语言指令、机器人状态、预训练 normalization statistics、action chunk 反归一化及 LIBERO 7 维连续控制动作转换。

- **端到端训练闭环与模型—环境解耦：** 串联 action-chunk rollout、CPU on-policy buffer、SMDP GAE、PPO minibatch update、训练指标及 checkpoint；通过 HTTP 将 OpenPI 模型服务和 LIBERO 仿真环境拆分至独立进程与 Python 环境，支持动作采样、完整去噪轨迹回传、rollout 提交、服务端 PPO 更新，以及执行 action chunk 前缀后基于最新视觉观测重新采样的滚动重规划评估。

---

## 二、项目整体架构

### 2.1 为什么需要双层建模

普通连续控制 PPO 通常直接输出单步 Gaussian action：

\[
a_t\sim\mathcal N(\mu_\theta(s_t),\sigma_\theta(s_t)^2),
\]

所以可以直接计算 \(\log\pi_\theta(a_t\mid s_t)\)。Pi0.5 则从初始噪声开始，通过多步 Flow Matching 去噪生成整个 action chunk。最终动作的精确边缘概率需要积分掉所有中间路径，难以直接计算；同时，一个 action chunk 会在 LIBERO 中连续执行多个环境动作，模型决策频率低于环境 step 频率。

项目因此拆成两层：

1. **内层去噪 MDP**：把每一步去噪转移定义为条件 Gaussian policy，优化完整动作生成路径；
2. **外层环境 SMDP**：把一次 action-chunk 执行定义为变时长 transition，按实际持续时间计算 reward、bootstrap 和 GAE。

### 2.2 完整数据流

```text
LIBERO 第三人称图像、腕部图像、语言指令和机器人状态
→ OpenPI transform 与 normalization
→ Pi0.5 编码视觉—语言 prefix
→ 从初始噪声开始执行多步 Flow-Gaussian 去噪
→ 保存完整 chain、old log-prob、timestep、velocity 和 value
→ 输出并反归一化 action chunk
→ 在 LIBERO 中执行完整 chunk 或其前缀
→ 保存 discounted chunk reward、duration、terminated/truncated 和 next value
→ 按外层 SMDP 计算 advantage 与 return
→ 固定历史 chain，用当前策略重算每一步 log-prob
→ PPO actor + critic + velocity anchor + reference KL
→ backward、梯度裁剪、optimizer step 与 checkpoint
```

### 2.3 两层状态、决策和反馈

**内层状态**包括固定的视觉—语言条件、机器人状态、当前带噪动作 \(x_t\) 和去噪时间步 \(t\)。内层随机结果是下一带噪状态 \(x_{t-\Delta t}\)，所有去噪步骤结束后得到最终 action chunk。内层没有独立环境 reward 和独立 critic，而是共享对应 action chunk 的外层 advantage。

**外层状态**是当前 LIBERO 多模态 observation。外层决策是生成并执行一个 action chunk；transition 持续时间是实际执行的环境动作数，反馈是 chunk 内累计折扣奖励与执行后的 observation。

---

## 三、Flow Matching 策略概率建模

### 3.1 原始难点

最终 action chunk 由路径

\[
x_1\rightarrow x_{t_1}\rightarrow x_{t_2}\rightarrow\cdots\rightarrow x_0
\]

生成。项目不假设最终 \(x_0\) 具有一个可直接求值的简单 Gaussian 边缘分布，而是在扩展状态空间中建模每一步条件转移：

\[
\pi_\theta(x_{t-\Delta t}\mid x_t,o,t).
\]

这样完整去噪路径的概率可以按链式法则分解。优化对象是生成 trajectory policy，而不是声称直接计算了最终动作的精确边缘 likelihood。

### 3.2 Flow velocity 如何构造转移均值

Pi0.5 action expert 预测：

\[
v_\theta(o,x_t,t).
\]

根据当前状态和 velocity 推导两端估计：

\[
\hat x_0=x_t-t v_\theta,
\qquad
\hat x_1=x_t+(1-t)v_\theta.
\]

令下一时间步为 \(t'=t-\Delta t\)，learned-noise 训练路径使用：

\[
\mu_\theta=(1-t')\hat x_0+t'\hat x_1.
\]

因此准确表述是“基于 velocity 构造 transition mean”，而不是“velocity 本身就是 Gaussian mean”。这保留了预训练 Flow dynamics，PPO 对 mean 的梯度可以沿 \(\mu_\theta\rightarrow v_\theta\) 回传到 `action_out_proj`、action expert 以及允许训练的 VLA 参数。

`pi05_denoising.py` 中的实现步骤为：

1. 将 timestep 和 delta 扩展到与动作 tensor 可广播的维度；
2. 计算 `x0_pred`、`x1_pred` 与 `next_timestep`；
3. 根据 `flow_ode`、`flow_sde`、`flow_cps` 或 `flow_noise` 选择权重和标准差；
4. 使用 `x0_pred * x0_weight + x1_pred * x1_weight` 得到 mean；
5. 将 std 扩展到完整动作 shape。

生产 PPO 路径使用 `flow_noise`；确定性评估可以切换为 `flow_ode`。其他方法保留在通用 helper 中，但不能据此声称正式训练使用了所有采样方法。

### 3.3 状态相关 Gaussian 标准差

新增 Gaussian action head 接收当前去噪步骤的 suffix hidden features，经 MLP、归一化和非线性映射输出各动作维度的 `log_std`：

\[
\log\sigma_\theta=
\operatorname{clip}(\operatorname{Head}(h_t),
\log\sigma_{\min},\log\sigma_{\max}),
\]

\[
\sigma_\theta=\exp(\log\sigma_\theta).
\]

每一步转移成为对角 Gaussian：

\[
x_{t-\Delta t}\sim
\mathcal N\left(\mu_\theta,
\operatorname{diag}(\sigma_\theta^2)\right).
\]

head 直接得到的是标准差，不是方差；方差是 \(\sigma^2\)。对角 Gaussian 避免预测高维完整协方差矩阵，能够对每个去噪步骤、动作时间和动作维度高效采样及计算概率。

初始标准差设置为较小值（默认约 0.1），而不是 1。原因是多步去噪会累计随机扰动，过大的初始噪声可能在 PPO 更新前就破坏预训练 Flow trajectory。较小初始化在保留探索的同时，让初始随机策略更接近预训练动作生成过程。

### 3.4 采样、log-prob 与 entropy

采样形式为：

\[
x_{t-\Delta t}=\mu_\theta+\sigma_\theta\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

每个元素的 log-prob 为：

\[
\log\pi_\theta
=-\log\sigma_\theta-
\frac12\log(2\pi)-
\frac12\left(
\frac{x_{t-\Delta t}-\mu_\theta}{\sigma_\theta}
\right)^2.
\]

实现保留 elementwise log-prob，而不是采样后立即压成标量，使训练阶段既可计算元素/步骤级 PPO，也可聚合为完整路径级 PPO。确定性 ODE 步骤的 std 为 0，helper 会使用安全 std 避免除零，并将对应 log-prob 和 entropy 置零。

Gaussian entropy 为：

\[
H=\frac12\log(2\pi e\sigma^2).
\]

代码支持 entropy regularization，但默认系数为 0，简历只能写“支持”，不能表述为正式训练必然启用。

### 3.5 关键设计取舍

- **不直接增加独立 mean MLP**：避免绕过预训练 Flow velocity field；
- **不直接对最终 action 假设简单 Gaussian**：最终边缘概率需要积分中间路径；
- **使用对角协方差**：控制高维 action chunk 与多步去噪的参数量和概率计算成本；
- **小 std 初始化**：降低随机化对预训练轨迹的初始破坏；
- **rollout 与 recompute 共用 transition helper**：避免行为策略采样公式与更新概率公式不一致。

### 3.6 对应代码

- `src/openpi/rl/pi05_action_head.py`：状态相关 `log_std/std` head；
- `src/openpi/rl/pi05_denoising.py`：时间表、Flow transition mean/std、Gaussian log-prob 与 entropy；
- `src/openpi/rl/pi05_nested_mdp.py`：Pi0.5 velocity 前向、分布构造和内层 rollout；
- `src/openpi/rl/pi05_types.py`：去噪 trajectory 数据结构。

### 3.7 面试回答模板

> Pi0.5 的最终 action chunk 是多步 Flow Matching 去噪得到的，最终动作边缘概率不容易直接计算。因此我把每一步去噪转移建模成条件 Gaussian policy。Pi0.5 仍预测 velocity，我根据当前 \(x_t\)、velocity、timestep 和步长推导 \(\hat x_0\)、\(\hat x_1\)，再构造下一步 transition mean；同时在 suffix hidden states 上增加 log-std head，预测状态相关标准差。这样每一步都可以采样并计算 log-prob，PPO 对 mean 的梯度回传到原 Flow 分支，对 std 的梯度回传到新增 head，既保留预训练 Flow dynamics，又获得可优化的随机策略。

---

## 四、历史去噪轨迹保存与当前策略概率重算

### 4.1 为什么必须保存完整 chain

PPO ratio 要求旧策略和当前策略评估同一个样本：

\[
r(\theta)=
\exp\left(
\log\pi_\theta-\log\pi_{\mathrm{old}}
\right).
\]

对于 Pi0.5，这个样本不仅是最终 action chunk，还包括完整历史路径。若更新时重新采样一条去噪 chain，新旧 log-prob 对应不同随机路径，ratio 不再具有标准 importance sampling 含义。

若有 \(K\) 个 transition，chain 需要保存 \(K+1\) 个状态，因为第 \(k\) 个概率对应相邻状态对 \((x_k,x_{k+1})\)。

### 4.2 rollout 保存的数据

每次生成 action chunk 保存：

- 初始噪声和完整 `chains`；
- behavior policy 的 elementwise `old_logprobs`；
- 各步 Gaussian `means` 与 `stds`；
- `denoise_indices` 与连续 `denoise_timesteps`；
- rollout-time `velocities`；
- 最终 action chunk；
- 外层 observation 的 critic value。

保留 elementwise 数据后，训练器可选择不同的 reduction，而无需重新采样或丢失细粒度信息。

### 4.3 current-policy recompute

更新阶段读取历史相邻状态，并执行：

```text
历史 x_k + 原 observation + 原 timestep
→ 当前模型重新预测 velocity 与 std
→ 使用相同 Flow 公式重算 current mean
→ 评估历史 x_{k+1} 的 current log-prob
```

计算的是：

\[
\log\pi_\theta
(x_{k+1}^{\mathrm{old}}\mid
x_k^{\mathrm{old}},o,t_k),
\]

旧概率则是 rollout 时保存的：

\[
\log\pi_{\mathrm{old}}
(x_{k+1}^{\mathrm{old}}\mid
x_k^{\mathrm{old}},o,t_k).
\]

二者对应同一 behavior transition，才能构造 PPO ratio。该过程也要求采样和重算使用完全一致的 timestep schedule、transition mean 公式、std head 和 mask。

### 4.4 Prefix KV cache

同一个 action chunk 的全部去噪步骤共享图像和语言条件，只有机器人状态、带噪动作和 timestep 变化。实现先计算一次视觉—语言 prefix hidden states 和 Transformer KV cache，后续每步只构造 suffix 并访问缓存 prefix，避免重复进行多模态编码。

这能从代码上证明减少了冗余前向，但没有 benchmark 时不能宣称具体加速倍数。

### 4.5 保存 velocity 的作用

rollout-time velocity 用于：

1. 检查 sample 与 recompute 的 Flow dynamics 是否对齐；
2. 作为 Flow velocity MSE anchor 的 target，限制当前 PPO 更新过度改变预训练速度场。

对应辅助损失为：

\[
L_{\mathrm{velocity}}
=
\frac1N\sum
\|v_\theta(o,x_t,t)-v_{\mathrm{target}}(o,x_t,t)\|_2^2.
\]

### 4.6 CPU on-policy buffer

完整 chain 的 shape 通常包含 rollout batch、去噪步数、action horizon 和 action dimension，长期驻留 GPU 会占用大量显存。因此数据先转到 CPU buffer，生成 minibatch 时再搬回训练设备。

该 buffer 同时保存：

- **内层轨迹**：chain、old log-prob、velocity、timestep/index；
- **外层 transition**：observation、reward、duration、terminated/truncated、value、next value、advantage 和 return。

它不是跨轮复用的 off-policy replay buffer。当前 behavior policy 的数据完成有限次 PPO epoch 后应清空，再由更新后的策略重新采样。

### 4.7 对应代码

- `src/openpi/rl/pi05_nested_mdp.py`：完整去噪采样和 current log-prob recompute；
- `src/openpi/rl/rollout_buffer.py`：内外层数据的 CPU 存储、校验和 minibatch；
- `src/openpi/rl/pi05_trainer.py`：调用 recompute 并组装训练 loss。

### 4.8 面试回答模板

> PPO 要求新旧策略评估同一个 behavior sample。对 Pi0.5 来说，sample 是完整去噪路径，而不只是最终 action。因此 rollout 时我保存 chain、old log-prob、timestep、mean/std 和 velocity。更新时不重新采样，而是固定历史 \(x_k,x_{k+1}\)，由当前模型根据历史 \(x_k\) 重算 mean/std，再评估历史 \(x_{k+1}\) 的 current log-prob。这样 current 和 old log-prob 对应同一条路径。所有去噪步骤共享视觉—语言条件，所以还复用了 prefix KV cache，只更新每步变化的 suffix。

---

## 五、Action-chunk SMDP 与变时长 GAE

### 5.1 为什么不是普通单步 MDP

Pi0.5 一次产生多个连续动作：

\[
[a_t,a_{t+1},\ldots,a_{t+m-1}].
\]

环境执行若干动作后才重新调用模型，一次模型决策跨越多个 LIBERO step。实际时长还会因为任务中途终止或滚动重规划而变化，因此外层 transition 应按 SMDP 建模，而不能把 action chunk 错误地当成一个固定单步 transition。

### 5.2 外层 transition 字段

每条 transition 保存：

```text
observation_t
完整 action-generation trajectory
discounted chunk reward
duration = 实际执行环境动作数
terminated
truncated
episode_end = terminated OR truncated
value = V(observation_t)
next_value = V(observation_{t+m})
```

`duration` 不能写死为模型 action horizon，因为 chunk 可能中途终止，评估也可能只执行前缀。

### 5.3 Chunk 内奖励聚合

若实际执行 \(m\) 步，外层 reward 是：

\[
R_t^{(m)}=
\sum_{j=0}^{m-1}\gamma^j r_{t+j}.
\]

环境端逐步执行动作并根据已执行步数累积折扣 reward。该 reward 已经完成 chunk 内折扣，后续计算 TD residual 时不能再次对整个 reward 乘 \(\gamma^m\)。

### 5.4 SMDP TD residual

下一 observation 位于 \(m\) 个环境步后，因此 bootstrap discount 是 \(\gamma^m\)：

\[
\delta_t=
R_t^{(m)}+
\gamma^m(1-\mathrm{terminated}_t)
V(o_{t+m})-V(o_t).
\]

`returns.py` 对 durations 计算逐 transition 的 `gamma_t = gamma ** durations`，再生成 delta。

### 5.5 Duration-aware GAE

普通 GAE 的 \(\gamma\lambda\) 衰减也要改为变时长版本：

\[
A_t=
\delta_t+
(\gamma\lambda)^m
(1-\mathrm{episode\_end}_t)A_{t+1}.
\]

代码从后向前递推 `running_advantage`，使用 `trace_t = (gamma * gae_lambda) ** durations`。执行 10 步的 chunk 对后续 advantage 的衰减自然大于执行 1 步的 chunk。

最终 critic return 为：

\[
\hat R_t=A_t+V(o_t).
\]

### 5.6 terminated 与 truncated

两者必须分别处理：

- **terminated**：任务真正终止，不使用 next value bootstrap；
- **truncated**：因时间上限或外部限制停止，状态在 MDP 意义上未必终止，可以对截断位置 observation 计算 next value；
- **共同点**：之后都会 reset，因此 GAE trace 都不能跨 reset 连接下一 episode。

所以实现用 `terminated` mask 控制 bootstrap，用 `episode_end = terminated OR truncated` 控制 trace。对于 truncation，必须在 reset 前计算截断 observation 的 value，不能把 reset 后的新 episode 初始 observation 当作当前 transition 的 next state。

### 5.7 数据合法性检查

GAE 和 buffer 对以下条件进行前置校验：

- rewards、values、next values、masks 和 durations shape 一致；
- 输入非空；
- \(\gamma\) 与 \(\lambda\) 位于 \([0,1]\)；
- duration 是正数；
- terminated 与 episode-end 是二值 mask；
- 每个 terminated transition 必须也是 episode end。

这些检查用于防止 shape 广播或终止语义错误导致静默的价值估计偏差。

### 5.8 完整 chunk 与前缀执行

- **训练默认完整执行**：环境 reward 对应完整生成 action chunk 的后果，credit assignment 最直接；
- **中途终止**：仅记录实际执行步数；
- **评估前缀执行**：设置 `execute_horizon=1/2`，执行后获取最新图像并重新采样；未执行 tail actions 丢弃。

训练技术上也能记录前缀 duration，但这会使完整生成路径的概率与实际使用的动作前缀之间出现额外 credit-assignment 取舍，因此当前推荐完整 chunk 训练。

### 5.9 对应代码

- `src/openpi/rl/returns.py`：duration-aware GAE；
- `src/openpi/rl/rollout_collector.py`：本地 chunk 执行、reward、duration 和 next value；
- `src/openpi/rl/run_pi05_libero_rollout.py`：HTTP client 侧 SMDP rollout；
- `src/openpi/rl/rollout_buffer.py`：外层 transition 存储与校验。

### 5.10 面试回答模板

> Pi0.5 一次生成 action chunk，所以一次模型决策会跨越多个 LIBERO step。我将它建模为外层 SMDP，并记录实际 duration。Chunk reward 按 \(\sum_{j=0}^{m-1}\gamma^j r_{t+j}\) 聚合，TD residual 使用 \(\gamma^m\) bootstrap，GAE trace 使用 \((\gamma\lambda)^m\)。我还区分 terminated 和 truncated：真实终止不 bootstrap；时间截断可以 bootstrap，但两者之后都会 reset，所以 advantage trace 都要断开。这保证了环境执行和价值估计使用相同的时间尺度。

---

## 六、多粒度 PPO 目标与训练约束

### 6.1 外层 advantage 如何作用于内层

LIBERO 只返回 action chunk 执行后的环境 reward，不为每个去噪步骤提供单独反馈。项目先在外层 SMDP 上计算 chunk advantage \(A_t\)，再将其广播到生成该 chunk 的所有有效去噪 transition。内层没有人为设计独立 reward，也没有每个去噪步骤独立的 critic。

### 6.2 元素/步骤级 PPO

对 elementwise log-prob 计算：

\[
\Delta\log p_i=
\log p_{\theta,i}-\log p_{\mathrm{old},i},
\qquad
r_i=\exp(\operatorname{clamp}(\Delta\log p_i)).
\]

clipped surrogate 为：

\[
L_{\mathrm{element}}=-\mathbb E_i
\left[
\min\left(
r_iA,
\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A
\right)
\right].
\]

该目标保留去噪步骤、action horizon 和动作维度上的细粒度更新信号，数值上较稳定；但它不等于完整生成路径的严格联合概率 PPO，而且同一 chunk advantage 会广播到多个元素。

### 6.3 完整去噪路径级 PPO

路径概率按链式法则聚合：

\[
\log p_\theta(\tau)=
\sum_k\log p_\theta(x_{k+1}\mid x_k,o,t_k).
\]

若考虑对角 Gaussian 的所有动作时间和维度，严格联合 log-prob 还需在这些有效元素上求和。实现支持：

- **sum reduction**：更接近严格联合路径 log-prob，但高维累加后 ratio 容易极端化；
- **mean reduction**：默认数值稳定近似，约束平均有效元素的 log-prob 变化，但不能称为严格联合概率。

聚合后对每个外层 transition 计算一个 path ratio，并使用同样的 PPO clipped surrogate。简历中的“完整轨迹”特指一个 action chunk 对应的完整去噪 trajectory，不是整个 LIBERO episode。

### 6.4 联合优化模式

训练器支持仅使用 element、仅使用 path，或将二者加权组合。两者侧重点不同：

- path-level 与外层 action-chunk transition 的整体语义更一致；
- element-level 提供更细粒度的局部梯度并缓解高维联合 ratio 的数值问题。

没有消融结果时，只能表述为“实现并支持组合”，不能声称组合一定优于单一目标。

### 6.5 Critic value loss

critic 每个外层 observation 输出一个 scalar：

\[
V_\phi(o_t).
\]

使用 SMDP GAE return 计算：

\[
L_V=
\mathbb E
[(V_\phi(o_t)-\hat R_t)^2].
\]

value 对应 action-chunk transition，而不是为每个去噪步骤预测独立 value。

### 6.6 Flow velocity anchor

当前策略在历史 chain 上重新计算 velocity，并与 rollout-time 或 reference target velocity 计算 MSE：

\[
L_{\mathrm{velocity}}=
\mathbb E\|v_\theta-v_{\mathrm{target}}\|_2^2.
\]

它作为预训练 Flow field 的锚点，限制 PPO reward 信号过度改变原有动作生成方向。在缺少消融实验时，应说“用于约束策略偏移”，而非“已证明提高稳定性”。

### 6.7 Frozen reference policy KL

冻结参考模型在相同 observation 和历史 chain 上计算 reference log-prob，形成 KL-style penalty，限制当前策略长期偏离预训练策略。

需要区分：

- **behavior approximate KL**：比较当前策略与本轮采样策略，用于监控单轮 PPO 更新并触发 early stopping；
- **reference KL**：比较当前策略与固定 pretrained/reference policy，用于限制多轮训练的累计漂移。

PPO clipping 不能完全替代 reference KL，因为每轮都只偏移一点，长期累计后仍可能远离预训练模型。

### 6.8 数值与更新控制

- **log-ratio clamp**：在 `exp` 前裁剪，防止高维 log-prob 聚合后上溢或下溢；
- **PPO ratio clipping**：算法目标的一部分，限制 surrogate objective 中的有效更新幅度；
- **gradient norm clipping**：backward 后裁剪当前模型梯度，避免异常 path ratio 或辅助损失产生梯度爆炸；
- **target-KL early stopping**：若当前 rollout 的 path approximate KL 超阈值，停止该批数据剩余 PPO epoch，不停止整个训练；
- **advantage normalization**：降低不同 batch 的 advantage 尺度差异；
- **entropy regularization**：已支持但默认系数为 0。

### 6.9 总损失

概念上可表示为：

\[
L_{
\mathrm{total}}=
L_{\mathrm{path}}+
\alpha L_{\mathrm{element}}+
\beta_V L_V+
\beta_F L_{\mathrm{velocity}}+
\beta_{KL}L_{\mathrm{refKL}}-
\beta_H H.
\]

具体启用项和权重由配置决定，不能把“代码支持”与“所有实验均启用”混为一谈。

### 6.10 对应代码

- `src/openpi/rl/pi05_losses.py`：element/path PPO、value、velocity anchor 和 reference KL；
- `src/openpi/rl/pi05_trainer.py`：current/reference recompute、loss 组合、backward 和 optimizer step；
- `src/openpi/rl/training_loop.py`：PPO epoch、minibatch 与 target-KL early stopping。

### 6.11 面试回答模板

> 外层 LIBERO reward 只评价 action chunk，因此我先计算 action-chunk SMDP advantage，再将它广播到对应的去噪步骤。Actor 同时支持 element-level 和 path-level PPO。Element-level 保留每个去噪步骤、动作时间和维度的细粒度 ratio；path-level 则聚合完整生成路径的 log-prob。严格联合概率应使用 sum，但高维下容易导致 ratio 极端化，因此代码默认也支持 mean 作为稳定近似。除此之外还有外层 critic loss、velocity MSE anchor、frozen reference KL、log-ratio clamp、梯度裁剪和 target-KL early stopping。PPO KL 控制本轮更新，reference KL 则控制相对预训练模型的长期漂移。

---

## 七、多模态 Critic 与 LIBERO 适配

### 7.1 Critic 网络

Pi0.5 prefix hidden states 编码视觉和语言条件。Value head 使用 prefix padding mask 对有效 token 做 masked mean pooling：

\[
h_{\mathrm{pool}}=
\frac{\sum_i m_i h_i}{\max(\sum_i m_i,1)},
\]

再经 MLP 输出外层状态价值：

\[
V(o)=\operatorname{MLP}(h_{\mathrm{pool}}).
\]

mask 避免 padding token 污染表示。Critic 不读取随机初始噪声或中间去噪状态，因为它服务于外层 SMDP：同一 observation 应对应稳定的 state value，而不是随本次动作生成随机性变化。

### 7.2 LIBERO 多模态 observation

Adapter 提取并转换：

- 第三人称 RGB 图像；
- 腕部 RGB 图像；
- 语言任务描述；
- 机器人状态。

图像统一为 OpenPI 需要的 HWC `uint8` 形式；腕部图像缺失时使用零图补齐。机器人状态优先读取完整 state，缺失时由末端位置、四元数和夹爪状态拼接，再补齐或裁剪到固定维度。

### 7.3 Normalization 与动作反归一化

RL rollout 复用预训练 checkpoint 对应的 normalization statistics 和 OpenPI transforms。若绕过它们，状态和动作尺度会偏离 SFT 阶段，可能造成输入分布错位、动作幅度异常或夹爪控制失真。

动作转换链路为：

```text
Pi0.5 [action_horizon, model_action_dim]
→ OpenPI output transform
→ 使用 checkpoint stats 反归一化
→ 提取 LIBERO 所需有效控制维度
→ 逐步发送 7 维连续动作到 env.step
```

模型动作宽度可以大于环境动作维度，因为模型采用固定 action representation，环境只消费有效的 7 维控制部分。

### 7.4 参数更新范围

当前本地入口和 HTTP server 将当前模型参数交给 Adam，并允许当前模型全参数参与优化；冻结的是 reference policy。Gaussian action head 和 `action_out_proj` 必须可训练，否则策略不能同时通过 std 与 Flow mean 两条路径学习。

因此当前代码路径不是“只训练新增 head”，也不是已经实现 LoRA。是否长期采用全参数更新需要结合 GPU 显存、吞吐和正式实验配置说明。

### 7.5 对应代码

- `src/openpi/rl/value_head.py`：masked pooling 和 critic MLP；
- `src/openpi/rl/pi05_nested_mdp.py`：prefix hidden states、KV cache 和 value 调用；
- `src/openpi/rl/libero_adapter.py`：图像、语言、状态、normalization 和动作转换；
- `src/openpi/rl/run_pi05_libero_rl.py`：当前模型、参考模型、optimizer 与训练组件组装。

### 7.6 面试回答模板

> Critic 复用 Pi0.5 的视觉—语言 prefix hidden states，通过 padding mask 对有效 token 做 masked mean pooling，再用 MLP 输出一个 action-chunk 级 state value。它不读取随机去噪状态，因为外层 GAE 要求同一 observation 对应稳定 value。环境侧将 LIBERO 的第三人称图像、腕部图像、语言和机器人状态转换为 OpenPI 输入，并复用 checkpoint normalization；模型 action chunk 经 output transform 和反归一化后，提取为 LIBERO 的 7 维连续控制动作。当前入口允许整个当前模型优化，冻结的是 reference policy。

---

## 八、训练闭环、HTTP 解耦与滚动重规划

### 8.1 本地 on-policy 训练流程

```text
1. reset 环境并读取 observation
2. 采样完整 denoising chain 和 action chunk
3. 在环境中逐步执行 action chunk
4. 累计 discounted chunk reward 和 duration
5. 获取 terminated、truncated 及 next observation
6. 在 reset 前计算需要 bootstrap 的 next value
7. 将内层轨迹和外层 transition 放入 CPU buffer
8. 收集指定数量 transition 后计算 SMDP advantage/return
9. 打乱数据并生成 minibatch
10. 对同一批 rollout 执行有限个 PPO epoch
11. 记录 policy/value/KL/clip fraction 等指标
12. 按间隔保存模型和 optimizer checkpoint
13. 清空 on-policy buffer并重新采样
```

同一批 rollout 可以执行多个 PPO epoch，但不能跨策略版本长期复用，因此该 buffer 不是 off-policy replay。

### 8.2 HTTP 拆分原因和边界

OpenPI 大模型与 LIBERO 仿真依赖不同的 Python、模型和仿真软件栈。项目将它们拆成独立进程和虚拟环境：

**OpenPI server 负责：**

- 加载当前模型和 frozen reference；
- 动作与去噪轨迹采样；
- critic value；
- buffer、GAE 和 PPO update；
- 训练指标与 checkpoint。

**LIBERO client 负责：**

- 环境 reset/step；
- observation 收集；
- action chunk 执行；
- reward、duration、terminated/truncated；
- rollout 回传。

这是模型端和环境端的进程/依赖解耦，不等同于多 actor、多 learner、多卡参数同步或异步分布式 RL。

### 8.3 `/sample` 与 `/update`

`/sample` 接收 observation，返回 action chunk、完整 chain、old log-prob、mean/std、timestep/index、velocity 和 value。

Client 执行动作后通过 `/update` 提交 observation、历史去噪轨迹、discounted reward、duration、terminated/truncated、value 和 next value。Server 校验 schema，构造 buffer，计算 GAE，并用当前策略重算概率后执行 PPO 更新，最后返回训练指标。

HTTP schema 必须检查 trajectory 字段长度、tensor shape、duration、终止 mask、observation 非空和 velocity 完整性，否则可能出现 reward 与 trajectory 错配、错误广播、非法 bootstrap 或 anchor target 错位。

### 8.4 滚动重规划

训练默认执行完整 chunk，使环境反馈与完整动作序列相对应。评估可设置 `execute_horizon=1/2`：

```text
采样完整 chunk
→ 只执行前 1/2 步
→ 丢弃未执行 tail
→ 读取最新图像
→ 重新采样新 chunk
```

这属于 receding-horizon control，可提高闭环反馈频率，但增加模型调用次数。没有正式对比实验时，只能写“支持滚动重规划评估”，不能声称已提高成功率。

评估禁止向训练 server 提交 rollout，避免污染 on-policy buffer。

### 8.5 Checkpoint 与恢复

Checkpoint 需要覆盖当前模型参数、新增 Gaussian/value head、optimizer state 和训练更新计数。历史 checkpoint 不一定包含 rollout buffer；on-policy PPO 恢复时可以丢弃旧 buffer并用恢复后的策略重新采样，但不能将旧策略 rollout 无条件当作当前 on-policy 数据继续训练。

### 8.6 对应代码

- `src/openpi/rl/training_loop.py`：本地循环、PPO epoch、指标和 checkpoint；
- `src/openpi/rl/openpi_policy_server.py`：HTTP sample/update 与服务端训练；
- `src/openpi/rl/http_protocol.py`：请求、响应和 rollout schema；
- `src/openpi/rl/libero_http_client.py`：HTTP client；
- `src/openpi/rl/run_pi05_openpi_server.py`：模型 server 入口；
- `src/openpi/rl/run_pi05_libero_rollout.py`：LIBERO rollout 与滚动重规划入口。

### 8.7 面试回答模板

> 我实现了完整 on-policy 闭环：采集 action-chunk rollout，把去噪轨迹和 SMDP transition 存入 CPU buffer，计算 duration-aware GAE，再执行 minibatch PPO、指标记录和 checkpoint。由于 OpenPI 和 LIBERO 的依赖环境不同，我通过 HTTP 拆分模型 server 与环境 client。Server 负责采样、value 和 PPO update，client 负责环境执行、reward、duration 和终止信息。这个架构是依赖和进程解耦，不是多 learner 分布式训练。评估还支持只执行 chunk 前缀、根据最新视觉观测重新采样的 receding-horizon control。

---

## 九、高频综合面试问题

### 9.1 项目最难的部分是什么？

> 第一是建立 PPO 概率与 Flow Matching 生成过程的对应关系。最终 action 的边缘概率难求，所以我把每步去噪定义成条件 Gaussian transition，rollout 保存完整 chain，更新时固定历史 chain 重算 current log-prob，保证 PPO ratio 对应同一 behavior trajectory。
>
> 第二是 action chunk 的时间尺度。一次模型决策会执行多个环境 step，因此必须保存 duration，在 reward、bootstrap 和 GAE 中使用 \(\gamma^m\) 与 \((\gamma\lambda)^m\)，并分别处理 termination 和 truncation。

### 9.2 PPO 梯度能回传到原始 Pi0.5 吗？

> 可以。Transition mean 由 Pi0.5 velocity 构造，因此 log-prob 对 mean 的梯度能回传到 `action_out_proj`、action expert 和允许训练的 VLA 参数；对 std 的梯度则回传到 Gaussian head。当前入口允许全模型优化，冻结的是 reference policy。

### 9.3 为什么路径概率可以用于优化最终动作？

> 我没有假设路径概率等于最终 action 的边缘概率。最终边缘概率需要对中间路径积分。项目是在扩展状态空间中优化去噪 trajectory policy，完整路径概率可以分解为各步条件转移概率的乘积，最终环境 reward 作为共享 advantage 作用于该生成路径。

### 9.4 这是 hierarchical RL 吗？

> 它具有内外两层结构，但不完全等于传统 options-based hierarchical RL。外层是 action-chunk 环境 SMDP，内层是固定 observation 下的去噪生成 MDP。内层不单独与环境交互，也没有独立 reward 和 critic，因此更准确的说法是双层 MDP/SMDP 建模。

### 9.5 这是 on-policy 还是 off-policy？

> 是 on-policy PPO。Buffer 保存当前 behavior policy 数据，同一批数据用于有限 PPO epoch，更新后清空并重新采样，不跨策略版本长期 replay。

### 9.6 为什么 PPO clipping 之外还需要 reference KL？

> PPO clipping 和 approximate KL 控制当前策略相对本轮 behavior policy 的单轮变化；即使每轮变化很小，长期累计仍可能远离预训练模型。Frozen reference KL 提供长期锚点，二者约束不同时间尺度的策略漂移。

### 9.7 为什么 path log-prob 默认使用 mean，而不是 sum？

> Sum 更接近严格联合路径概率，但去噪步数、action horizon 和动作维度共同形成高维随机变量，单元素的小变化累加后会让 path log-ratio 极端化。Mean 是数值稳定近似，相当于约束平均有效元素的概率变化。代码保留 sum 选项，因此可以在实验中比较，但不能把默认 mean 描述成严格联合概率。

### 9.8 为什么 critic 不为每个去噪步骤预测 value？

> 环境 reward 只在 action chunk 执行后产生，当前设计的信用分配单位是外层 transition，所以 critic 估计 \(V(o_t)\)。给内层每步建立 value 需要定义内层 reward 或更复杂的 return，目前没有必要，也会增加训练难度。

### 9.9 为什么不只训练新增 head？

> 只训练 std head 只能调整探索尺度，无法根据 reward 改变 Flow mean；要优化动作方向，至少需要让 velocity 输出相关参数可训练。当前入口支持全参数更新，后续可根据显存和稳定性研究只训练 action expert、LoRA 或分阶段解冻，但不能声称当前已经实现这些方案。

### 9.10 是否已经验证成功率提升？

> 当前已完成算法模块、SMDP 语义、关键校验和训练链路实现，但尚无完整多随机种子结果，因此不宣称相对 SFT baseline 已获得确定提升。正式结论需要报告不同 LIBERO suite 的 SFT/PPO 成功率、均值方差、训练曲线，并对 path/element loss、velocity anchor 和 reference KL 做消融。

---

## 十、五分钟项目介绍

> 这个项目是在预训练 Pi0.5 VLA 模型上实现面向 LIBERO 机器人操作任务的 PPO 微调。和普通连续控制 PPO 不同，Pi0.5 不是直接输出单步 Gaussian action，而是从初始噪声开始，通过多步 Flow Matching 去噪生成 action chunk。
>
> 第一个问题是最终动作的边缘概率难以直接计算。我把去噪过程建模成内层 MDP。每个状态包含固定视觉—语言条件、机器人状态、当前带噪动作和 timestep。Pi0.5 action expert 预测 Flow velocity，我根据当前状态、velocity、时间步和步长构造 transition mean；同时在 suffix hidden states 上增加 log-std head，预测状态相关标准差。这样每一步成为条件对角 Gaussian transition，能够采样和计算 log-prob。
>
> Rollout 时我保存完整 denoising chain、old log-prob、mean/std、timestep、index 和 velocity。PPO 更新时不重新采样，而是固定历史相邻状态，用当前模型重新预测 mean/std，再计算历史下一状态的 current log-prob，从而保证新旧策略评估同一 behavior trajectory。因为所有去噪步骤共享图像和语言条件，还复用了 prefix KV cache。
>
> 第二个问题是一次模型决策会连续执行多个 LIBERO step。我把 action chunk 建模为外层 SMDP transition，保存实际执行时长 \(m\)。Chunk reward 按 \(\sum_{j=0}^{m-1}\gamma^j r_{t+j}\) 聚合，TD bootstrap 使用 \(\gamma^m\)，GAE trace 使用 \((\gamma\lambda)^m\)。真实终止不 bootstrap，时间截断可以 bootstrap，但两者都会 reset，所以 advantage trace 都要断开。
>
> Actor loss 支持 element-level 和完整 denoising path-level PPO。外层 advantage 广播到对应去噪步骤；path 的 sum reduction接近联合概率，但高维下 ratio 容易极端，因此也支持 mean 稳定近似。此外还加入外层 critic loss、Flow velocity anchor、frozen reference KL、log-ratio clamp、梯度裁剪和 target-KL early stopping。Entropy 已实现但默认关闭。
>
> Critic 复用视觉—语言 prefix hidden states，通过 masked pooling 和 MLP 输出外层 observation value。环境侧完成了 LIBERO 图像、语言、机器人状态、normalization statistics 和动作反归一化适配，将模型 action chunk 转换为 7 维连续控制动作。
>
> 工程上实现了 CPU on-policy buffer、SMDP GAE、minibatch PPO 和 checkpoint。由于 OpenPI 与 LIBERO 依赖不同，通过 HTTP 拆分模型 server 和环境 client：server 负责采样、value 与更新，client 负责仿真执行和反馈。这是进程依赖解耦，不是分布式 learner。评估支持执行 chunk 前缀后基于最新视觉观测重新规划。目前代码链路已搭建，但没有完整多随机种子成功率结果，因此不会写未经验证的性能提升。

---

## 十一、代码阅读路线

1. `pi05_denoising.py`：先理解 timestep schedule、Flow mean/std、log-prob；
2. `pi05_action_head.py`：理解 hidden features 到 `log_std`、clamp 和初始化；
3. `pi05_nested_mdp.py`：理解 prefix cache、velocity、rollout 和 recompute；
4. `returns.py`：掌握 \(\gamma^m\)、\((\gamma\lambda)^m\) 和两类 mask；
5. `pi05_losses.py`：理解 element/path PPO、sum/mean、anchor 和 reference KL；
6. `pi05_trainer.py`：理解 loss 组合、梯度和 optimizer；
7. `rollout_buffer.py`：理解内层 trajectory 与外层 transition 的数据对齐；
8. `value_head.py`：理解 masked pooling 与 scalar value；
9. `libero_adapter.py`：理解多模态输入、normalization 和动作转换；
10. `openpi_policy_server.py`、`http_protocol.py`：理解 HTTP schema、校验和 server update。

面试回答遵循“问题是什么—为什么普通方法不适用—公式如何修改—代码在哪一层实现—有哪些取舍和边界”。先讲主线，再根据追问展开，不要一开始堆叠所有实现细节。

---

## 十二、简历与实验结论边界

### 代码可以直接支持的表述

- 内层去噪 MDP 与外层 action-chunk SMDP；
- learned Gaussian standard-deviation head；
- 基于 Flow velocity 构造 transition mean；
- 完整 chain 保存与 current-policy log-prob recompute；
- duration-aware reward、bootstrap 和 GAE；
- element/path 及组合 PPO；
- critic、velocity anchor、reference KL、gradient clipping 和 target-KL；
- OpenPI—LIBERO 多模态适配；
- 本地 training loop 与 HTTP client/server；
- 完整 chunk 训练和滚动重规划评估。

### 需要正式实验支持的表述

- PPO 稳定收敛或任务成功率提升；
- 超过 SFT、RLinf 或其他 baseline；
- path + element 组合优于单一目标；
- velocity anchor、reference KL 或 entropy 带来确定收益；
- 具体 GPU 吞吐、显存或推理加速比例；
- 多任务、多随机种子的统计显著性；
- 多 actor、多 learner 或多 GPU 分布式扩展。

完成正式实验后，优先补充：SFT 与 PPO 的任务成功率、覆盖 suite/task 数、随机种子均值和标准差、训练曲线、采样吞吐、显存，以及关键 loss 的消融结果。