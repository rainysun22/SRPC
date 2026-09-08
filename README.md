# SR-PC（自省式预测编码）演化智能系统

Self-Reflective Predictive Coding：从"最小化预测误差"这一条内置规则出发，通过与世界交互在线长出**形成 - 修正 - 组合**能力的演化智能系统原型。对应设计文档 [docs/SRPC_DESIGN.md](docs/SRPC_DESIGN.md)（v1.9，大模型仅作能力评测基准、不进系统构造），后续路线（E / F / G / D）见 [docs/ROADMAP.md](docs/ROADMAP.md)：

- **阶段 A（Phase-0 原型）**：原理自证 —— 形成、修正、组合的最小闭环 + 信用分配早筛（承重墙）
- **阶段 B（规模化与组合）**：深层预测编码（DeepSRPC）、多时间尺度记忆（PrototypeMemory）、ARC-lite 组合基准、顺序学习免遗忘 —— 主验收 3/4 过（learn / forget / combo）；**记忆增益项经 E0 收口重定性为原理性不可达**（ARC-lite 为 i.i.d. 映射，记忆先验无增量信息——b1 零距离耦合/单头遗忘压力消融均证 ≈0%；非架构债务，"能力=记忆·拼合"验证迁移 E3 语言流 / F3 ICL）；B 收尾 W1（符号保真自测 + 预算记账）8/8 过。证据见 [results_phaseB/report_w1.md](results_phaseB/report_w1.md) / [results_phaseB/report_e0.md](results_phaseB/report_e0.md)
- **阶段 C（内在低功耗，软件版）**：出生即结构稀疏（分块/扇入受限 + k-WTA）+ 三口径 MAC 记账 + 大模型能效标尺 + int8 部署就绪 —— 低功耗是架构本身的属性，非训练后裁剪；**已锁定**，硬件化 deferred 至有专用硬件
- **阶段 E（语言化，进行中）**：字节级 UTF-8 词元前端 + 语言长程信用分配 + 缩放梯子 —— **E1 已收口**（主验收 5/5 PASS，assoc 延迟文本关联）；**E2 pilot 已收口**（µPC 参数化 + 独立线性读出头 + iPC 增量调度，1/2/4M 梯子 vs BPTT 孪生 1.20–1.22×，判据 2/3/4 PASS；判据 1 单调性转 GPU 登顶裁决 [docs/GPU_TASKS.md](docs/GPU_TASKS.md) T1/T2）。证据见 [results_e1/report.md](results_e1/report.md) / [results_e2/report.md](results_e2/report.md)

## 核心特性

- **单一原理**：自由能最小化同时驱动感知 / 行动 / 学习
  - 外部预测误差 `e_out = s − ŝ`
  - 自省误差 `e_self = x_self − pred(x_self)`，回传修正自我模型并改变动作倾向
- **无反向传播**：纯 NumPy 局部规则（局部推断 + 局部 Hebbian / LMS），逐样本在线增量更新
- **事件驱动稀疏更新**：|更新| 超过阈值 θ 的节点才更新，事件率随学习下降，能量内生约束
- **自我模型从一开始就在**（不变量 4）：自省环 + 主动推理（动作 = argmin 预期自省误差 + 认识价值探索）
- **规模化（阶段 B）**：可配置 L 层深层预测编码网络，forward（自上而下生成）与 backward_error（自下而上只传误差）方向严格分离
- **多时间尺度记忆（阶段 B）**：fast 工作记忆 + 按任务分组的 slow 长期记忆，WTA 稀疏写入，结构上免遗忘
- **组合泛化（阶段 B）**：ARC-lite 8×8 网格变换基准；保留组合零样本 = 顺序复用两个已学变换片段（条件门控读出头）重组出新变换
- **出生即结构稀疏（阶段 C）**：扇入受限权重掩码（层 1 连续感受野 / 内部随机扇入）+ k-WTA 激活，自出生定型、训练只改已有突触（非训练后裁剪）
- **三口径 MAC 能耗记账（阶段 C）**：事件驱动（活跃单元 × 已有突触）/ 结构（全单元 × 已有突触）/ 稠密等价，硬件无关的架构内在能耗度量，直接对比大模型标尺（N_params × n_tokens）
- **int8 部署就绪（阶段 C）**：训练后对称量化，冻结评估能力无回撤；事件驱动/神经形态芯片本体 deferred 至有专用硬件

## 目录结构

```
srpc/
  config.py      # 全部超参数（模型 / 环境 / 训练轨道 / 阶段 B）
  env.py         # 环境：SourceFieldWorld（Track-1 导航）、SlotWorld（Track-2 组合）
  model.py       # SR-PC 核心（阶段 A）：层级推断、局部 Hebbian、自省环、主动推理
  deepmodel.py   # DeepSRPC（阶段 B）：深层预测编码、条件门控读出头、记忆/条件先验
  memory.py      # PrototypeMemory（阶段 B）：多时间尺度原型记忆、免遗忘结构
  arc.py         # ArcLite（阶段 B）：ARC-lite 组合基准
  runner.py      # 实验编排与验收判定（阶段 A）
  runner_b.py    # 顺序学习免遗忘 + 组合零样本 + 验收判定（阶段 B）
  runner_c.py    # 阶段 C 主流程：稀疏/稠密双臂 + 能耗测量 + C1-C4 验收
  credit.py      # 信用分配早筛（阶段 A 承重墙）：延迟 XOR + 误差/纯相关双臂对照
  lang.py        # ByteTokenizer 字节级 UTF-8 词元前端（阶段 E1）：256 维一热 = 正交基底
  lm.py          # LMPCN 语言模型（阶段 E2）：µPC 参数化 + 独立线性读出头 + iPC 增量调度 + 孪生对照
  lmgpu.py       # LMPCNg：E2 GPU 登顶跑 torch 移植（权重真源 = numpy 同 seed；无 autograd）
  lmgpu_graph.py # LMPCNgG：CUDA-Graph 捕获整训练步（就地更新，~10×；GPU_TASKS T1）
  energy.py      # EnergyLedger 三口径 MAC 记账 + 大模型标尺 llm_task_macs（阶段 C）
  metrics.py     # 指标：感受野对齐、NMI、恢复统计、零样本组合泛化
  plots.py       # 可视化（阶段 A）
  plots_b.py     # 可视化（阶段 B）
  plots_c.py     # 可视化（阶段 C：结构/能耗/能力/活跃率）
scripts/
  run_phase0.py  # 阶段 A 入口：跑全部实验并生成验收报告
  run_phaseB.py  # 阶段 B 入口：顺序学习 + 组合泛化 + Pareto 回归验收
  run_phaseC.py  # 阶段 C 入口：结构稀疏核心双臂 + 能耗/结构/量化验收
  run_e1.py      # 阶段 E1 入口：字节词元质检 + assoc/xorsum 长程信用分配 + 验收
  run_e2.py      # 阶段 E2 入口：锚点网格 + 1/2/4M 梯子 + BPTT 孪生 + iPC 消融 + 验收报告
  run_e2_summit.py        # E2 GPU 登顶跑单档（tinyshakespeare 全 epoch；--resume 断点续跑）
  run_e2_summit_all.py    # E2 登顶全流程编排（5 档 PCN → 5 档孪生 → 自动报告，幂等续跑）
  e2_gpu_parity.py        # 移植 parity（numpy vs torch CPU）+ GPU 冒烟基准
  build_e2_summit_report.py  # E2 登顶裁决报告（全预算单调性/斜率/比值判定）
docs/
  SRPC_DESIGN.md  # 设计文档 v1.9（§7.5 阶段 A 六项验收 / §8.5 信用分配早筛 / §8 里程碑 / §9 开放问题）
  ROADMAP.md      # 后续路线图（E 语言化 / F 知识+持续学习 / G 推理规划 / D 意识向）
  GPU_TASKS.md    # GPU 依赖任务清单（沙箱外执行）
results/          # 阶段 A：自动生成的图表 / metrics.json / report.md
results_phaseB/   # 阶段 B：同上
results_phaseC/   # 阶段 C：同上
results_e1/       # 阶段 E1：词元质检 + 长程信用分配验收
results_e2/       # 阶段 E2：锚点网格 / 梯子 / 孪生 / iPC / 验收报告
results_e2_gpu/   # 阶段 E2 GPU 登顶跑：全 epoch 梯子 / 孪生 / parity / 裁决报告
```

## 快速开始

```bash
pip install numpy matplotlib
python scripts/run_phase0.py              # 阶段 A：默认 3 seeds，约 30–60s
python scripts/run_phaseB.py              # 阶段 B：默认 3 seeds，约 35s
python scripts/run_phaseB.py --regression # 阶段 B + Phase-0 Pareto 回归（A 指标不退化）
python scripts/run_phaseC.py              # 阶段 C：稀疏/稠密双臂 + C1-C4 验收，约 52s
python scripts/run_phaseC.py --steps 200  # 冒烟测试
```

运行后自动生成 `results/report.md`、`results_phaseB/report.md`、`results_phaseC/report.md`（验收报告）与图表。

## 验收结果（阶段 A，3 seeds 均值，每个 seed 都必须通过）

> 五条件验收（对应 v1.5 §7.5 第 1/3/4/5 项 + **信用分配早筛 §2.4/§8.5 承重墙**）；**免遗忘早验（第 6 项）**已由阶段 B 顺序学习实验覆盖（见下）。
> 结构性稀疏（不变量 3：出生即掩码 + k-WTA）已内嵌为默认数学形式，下列数字即稀疏核心的结果。

| 条件 | 结果 | 关键证据 |
|---|---|---|
| 1 自组织层级结构 | PASS | 感受野对齐 0.559 → 0.979，捕获率 100%；概念层 NMI 0.376（随机 0.003） |
| 2 误差随交互下降 | PASS | EMA 斜率 −3.3e−5；首/末十分位误差 0.28 → 0.018 |
| 3 自省环非零增益 | PASS | 扰动重学窗口误差 ON 0.070 vs OFF 0.090（低 22%）；恢复步数 ON 3461 vs OFF 4476 |
| 4 无反传·在线 | PASS | 静态检查无 autograd/反传依赖；逐样本在线更新 |
| 5 信用分配早筛（§8.5 承重墙） | PASS | 延迟 Δ=4：误差驱动 0.890 vs 纯相关 0.512（分离 0.378 ≥ 0.30）；远端权重驱动 0.409（机会 0.20）；Δ=1 对照：误差驱动 0.934 可学（近程对照成立） |

信用早筛 per-seed（每个 seed 全过：acc_pcn ≥ 0.80 / gap ≥ 0.30 / distal ≥ 0.25）：err 0.880/0.837/0.954，hebb 0.528/0.526/0.483，gap 0.352/0.311/0.471，distal 0.480/0.404/0.342。种子间方差修复（0/1 二值 bit 编码 + α=1.5 类拉动 + err 臂深迭代 32 + hebb 臂浅迭代 8 自由推断）见 [srpc/config.py](srpc/config.py) `CreditConfig` 注释。

补充：零样本组合泛化 SR-PC 保留组合误差 0.050 vs 查表基线 1.099；事件驱动更新率 0.45 → 0.06。

## 验收结果（阶段 B，3 seeds，每个 seed 都必须通过）

> 对照 v1.7 §8 阶段 B 关键验收（能力上升 + 免遗忘 + 预算不缺 + 符号保真）。B 收尾 W1 已完成：**符号保真自测**按在线可达阈值（v2，附 LS 上限对比）8/8 过；**迭代/稀疏预算记账**过（B1 口径修正为"≤3 迭代有效，>3 迭代退化"，退化记债务 #2、降优先级）。记忆增益项经 **E0 收口**（五配置矩阵 + 单头消融）重定性为原理性不可达、验证迁移 E3/F3；遗留债务（深迭代退化、在线 RLS 距 LS 差 2.6pp）与全部证据见 [results_phaseB/report_w1.md](results_phaseB/report_w1.md) / [results_phaseB/report_e0.md](results_phaseB/report_e0.md)。
> 结构稀疏（不变量 3：出生即掩码 + k-WTA）已内嵌为 A/B 默认数学形式，下列数字即稀疏核心的结果。

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| 1 能力随交互上升 | PASS | 4 个变换任务任务内误差斜率全部 < 0（均值 −5.2e−2 / −4.9e−2 / −4.9e−2 / −4.9e−2） |
| 2 免遗忘（带记忆） | PASS | 顺序学习后旧任务误差相对回升均值 5.5%（阈值 ≤25%），max 48%（观察项，见 report_w1 §4） |
| 3 记忆增益 | **原理性不可达（E0 收口）** | 带记忆保留误差 2.4% vs 无记忆 2.4%。E0 五配置矩阵 + 单头消融判定：b1（读出源=x_L，零传播距离）增益 −0.25%、单头（遗忘压力最大化）× b1 增益 −1.9%、a/ab（x1 级记忆）−126% 且 S1/S2 回退——ARC-lite retain 为 i.i.d. 确定映射，I(s_out; 记忆原型 \| x_L) ≈ 0，任何耦合设计均不可达；非架构债务，记忆价值验证迁移 E3（语言流）/F3（ICL），见 [results_phaseB/report_e0.md](results_phaseB/report_e0.md) |
| 4 组合零样本 | PASS | 保留组合顺序复合 0.0261 vs 随机条件基线 0.0355（增益 26%，阈值 ≥20%） |
| Pareto 回归 | PASS | Phase-0 快速回归（7.5 四项）全过，阶段 A 指标未退化 |
| W1 符号保真自测（阈值 v2） | PASS | S1 逐格一致率 min 0.927~0.953（≥0.92；LS 上限 0.987）；S2 位置/颜色区分 max 0.630/0.962（≤0.70/0.97）；输入多样性 1.000（≥0.90）；网格级一致率与块结构守恒为记录项（LS 上限 0.555 < 原阈值 0.90，不可达故不设门） |
| W1 预算记账 | PASS | B1：iters=3 达平台精度 95% 以上（≤3 迭代有效，不靠深迭代烧算力；>3 迭代退化 0.96→0.69 记债务 #2）；B2：event/dense = 0.11（≤0.5），结构密度 0.25（≤0.35） |

补充：事件驱动更新率 mem 0.913 vs no-mem 0.839；完整报告见 [results_phaseB/report.md](results_phaseB/report.md)（含 W1 收尾项与架构债务节）。

## 验收结果（阶段 C 软件版，3 seeds，每个 seed 都必须通过）

> 对照 v1.5 §8 里程碑 C 的软件版（无专用硬件，CPU/GPU 验证**架构本身**的低功耗）：出生即稀疏核心从头重训，能力与能效双验收；**阶段 C 已锁定**，事件驱动/神经形态芯片部署本体 deferred 至有专用硬件，不再投入。对照臂 = 同 seeds 同超参的稠密网络（只开记账）。

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| C1 能力无回撤（B 复跑） | PASS | 稀疏核心上阶段 B 四项验收全部复现（组合增益 41%） |
| C2 能效（vs 大模型标尺） | PASS | 每样本推理 8.63e4 MACs（事件驱动）vs LLM-0.5B 2.56e11（最低比率 3e6 倍，阈值 ≥10³） |
| C3 结构由构造保证 | PASS | 权重总密度 0.280（阈值 ≤0.35），掩码自出生不变（学习只改已有突触） |
| C4 int8 部署就绪 | PASS | 量化后冻结误差 0.099 vs fp 0.098；组合 0.103 vs 0.100（无回撤） |

补充：结构口径（架构内在能耗主度量）稀疏核心比稠密对照臂**节省 74%**；与自身稠密等价比节省 89%。事件口径臂间对比不可直接比（稠密臂内部层事件静默、信息流微弱，稀疏核心 k-WTA 保证内部层真实活跃 12–38%）——详见 [results_phaseC/report.md](results_phaseC/report.md) 的口径解读。

## 阶段 E（语言化）进度

> 单一原理在同一套局部规则（误差驱动 PCN）内长出语言能力，缩放曲线对标自训 BPTT 孪生。E0（记忆价值重定性）已收口，E1/E2 见下，E3/F/G 详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

| 子阶段 | 状态 | 关键证据 |
|---|---|---|
| E1 词元前端 + 长程信用分配 | ✅ 收口（5/5 PASS） | ByteTokenizer（字节级 UTF-8，256 维一热，7 组多语/emoji/控制字符往返质检全过）；assoc 延迟文本关联主验收 acc 0.995 / gap 0.818 / Δ=8 外推 0.985（3 seeds）；xorsum 在线不可达归因 parity SQ-hard（BP 在线同样失败，batch 可解），修复排 F1 记忆回放。见 [results_e1/report.md](results_e1/report.md) |
| E2 缩放梯子 + 孪生对照 | ✅ pilot 收口（判据 2/3/4 PASS，判据 1 转 GPU 裁决）→ **GPU 登顶跑执行中（2026-09-08）** | µPC 参数化零调参迁移（锚点 iters=12 / eta_w=0.01，BPC 4.529 下穿 unigram 4.797）+ 独立线性读出头（W_out+bias 承载 unigram 先验）+ iPC 增量调度采纳；梯子 1/2/4M BPC 4.256/4.213/4.238，BP 孪生 3.500/3.464/3.530，PCN:孪生恒 1.20–1.22×；判据 1 单调性 pilot 与孪生同步回退 → 数据量瓶颈定性，**全预算裁决跑（torch/CUDA-Graph 移植 ~10×，全 epoch 1M–15M 梯子 + 孪生）后台进行中**，见 [results_e2/report.md](results_e2/report.md) / [results_e2_gpu/](results_e2_gpu/report.md) 与 [docs/GPU_TASKS.md](docs/GPU_TASKS.md) T1/T2 |

## A/B 实验设计

所有 Track-1 结论均来自**自省环 ON vs OFF** 的受控对照（同种子同环境，仅切换 `self_loop`），扰动测试采用全局感觉重映射（感知维度随机置换），不存在"回避扰动区"的捷径。
阶段 B 免遗忘/记忆增益为**带记忆 vs 无记忆**对照（同种子同环境，仅切换 `memory`）；组合零样本为**正确条件顺序复合 vs 无信息均匀条件多头混合**对照，保留组合在训练中完全不可见。
阶段 C 能效对照为**出生即稀疏核心 vs 同 seeds 同超参稠密网络**（仅结构掩码与 k-WTA 不同）；大模型标尺（N_params × n_tokens）仅为比较基准，不进入系统构造。
