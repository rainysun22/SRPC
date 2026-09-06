# SR-PC（自省式预测编码）演化智能系统

Self-Reflective Predictive Coding：从"最小化预测误差"这一条内置规则出发，通过与世界交互在线长出**形成 - 修正 - 组合**能力的演化智能系统原型。对应设计文档 [docs/SRPC_DESIGN.md](docs/SRPC_DESIGN.md)（v1.4，大模型仅作能力评测基准、不进系统构造）：

- **阶段 A（Phase-0 原型）**：原理自证 —— 形成、修正、组合的最小闭环
- **阶段 B（规模化与组合）**：深层预测编码（DeepSRPC）、多时间尺度记忆（PrototypeMemory）、ARC-lite 组合基准、顺序学习免遗忘

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
  metrics.py     # 指标：感受野对齐、NMI、恢复统计、零样本组合泛化
  plots.py       # 可视化（阶段 A）
  plots_b.py     # 可视化（阶段 B）
scripts/
  run_phase0.py  # 阶段 A 入口：跑全部实验并生成验收报告
  run_phaseB.py  # 阶段 B 入口：顺序学习 + 组合泛化 + Pareto 回归验收
docs/
  SRPC_DESIGN.md  # 设计文档 v1.4（§7.5 阶段 A 六项验收 / §8 里程碑）
results/          # 阶段 A：自动生成的图表 / metrics.json / report.md
results_phaseB/   # 阶段 B：同上
```

## 快速开始

```bash
pip install numpy matplotlib
python scripts/run_phase0.py              # 阶段 A：默认 3 seeds，约 30–60s
python scripts/run_phaseB.py              # 阶段 B：默认 3 seeds，约 35s
python scripts/run_phaseB.py --regression # 阶段 B + Phase-0 Pareto 回归（A 指标不退化）
python scripts/run_phaseB.py --steps 200  # 冒烟测试
```

运行后自动生成 `results/report.md`、`results_phaseB/report.md`（验收报告）与图表。

## 验收结果（阶段 A，3 seeds 均值）

> 按原四条件验收（对应 v1.4 §7.5 第 1/3/4/5 项）；v1.4 新增的**信用分配早筛（§2.4/§8.5，承重墙）**待实现，**免遗忘早验（第 6 项）**已由阶段 B 顺序学习实验覆盖（见下）。

| 条件 | 结果 | 关键证据 |
|---|---|---|
| 1 自组织层级结构 | PASS | 感受野对齐 0.496 → 0.966，捕获率 100%；概念层 NMI 0.398（随机 0.003） |
| 2 误差随交互下降 | PASS | EMA 斜率 −7.5e−5；首/末十分位误差 0.73 → 0.018 |
| 3 自省环非零增益 | PASS | 扰动重学窗口误差 ON 0.058 vs OFF 0.069（低 16%）；扰动后稳态误差低 7% |
| 4 无反传·在线 | PASS | 静态检查无 autograd/反传依赖；逐样本在线更新 |

补充：零样本组合泛化 SR-PC 保留组合误差 0.025 vs 查表基线 1.099；事件驱动更新率 0.37 → 0.07。

## 验收结果（阶段 B，3 seeds，每个 seed 都必须通过）

> 对照 v1.4 §8 阶段 B 关键验收（能力上升 + 免遗忘 + 预算不缺 + 符号保真）：前两项及组合零样本已过；ARC 一热编码即 §7.7 正交基底符号接地，**符号保真自测与迭代/稀疏预算记账**为待办。

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| 1 能力随交互上升 | PASS | 4 个变换任务任务内误差斜率全部 < 0（均值 −0.37 / −0.034 / −0.033 / −0.036） |
| 2 免遗忘（带记忆） | PASS | 顺序学习后旧任务误差相对回升均值 0.7%（阈值 ≤25%），max 5.3% |
| 3 记忆增益 | PASS | 带记忆保留误差 4.3% vs 无记忆 7.3%（改善 41%，阈值 ≥10%） |
| 4 组合零样本 | PASS | 保留组合顺序复合 0.042 vs 随机条件基线 0.121（增益 66%，阈值 ≥20%） |
| Pareto 回归 | PASS | Phase-0 快速回归（7.5 四项）全过，阶段 A 指标未退化 |

补充：事件驱动更新率 mem 0.051 vs no-mem 0.246（记忆先验使编码更稀疏）；完整报告见 [results_phaseB/report.md](results_phaseB/report.md)。

## A/B 实验设计

所有 Track-1 结论均来自**自省环 ON vs OFF** 的受控对照（同种子同环境，仅切换 `self_loop`），扰动测试采用全局感觉重映射（感知维度随机置换），不存在"回避扰动区"的捷径。
阶段 B 免遗忘/记忆增益为**带记忆 vs 无记忆**对照（同种子同环境，仅切换 `memory`）；组合零样本为**正确条件顺序复合 vs 无信息均匀条件多头混合**对照，保留组合在训练中完全不可见。
