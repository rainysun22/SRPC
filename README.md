# SR-PC（自省式预测编码）Phase-0 原型

Self-Reflective Predictive Coding：从"最小化预测误差"这一条内置规则出发，通过与世界交互在线长出**形成 - 修正 - 组合**能力的演化智能系统原型。对应设计文档 [docs/SRPC_DESIGN.md](docs/SRPC_DESIGN.md) 的阶段 A（原理自证）。

## 核心特性

- **单一原理**：自由能最小化同时驱动感知 / 行动 / 学习
  - 外部预测误差 `e_out = s − ŝ`
  - 自省误差 `e_self = x_self − pred(x_self)`，回传修正自我模型并改变动作倾向
- **无反向传播**：纯 NumPy 局部规则（局部推断 + 局部 Hebbian / LMS），逐样本在线增量更新
- **事件驱动稀疏更新**：|更新| 超过阈值 θ 的节点才更新，事件率随学习下降（0.37 → 0.07），能量内生约束
- **自我模型从一开始就在**（不变量 4）：自省环 + 主动推理（动作 = argmin 预期自省误差 + 认识价值探索）

## 目录结构

```
srpc/
  config.py    # 全部超参数（模型 / 环境 / 训练轨道）
  env.py       # 环境：SourceFieldWorld（Track-1 导航）、SlotWorld（Track-2 组合）
  model.py     # SR-PC 核心：层级推断、局部 Hebbian、自省环、工作空间重组、主动推理
  metrics.py   # 指标：感受野对齐、NMI、恢复统计、零样本组合泛化
  plots.py     # 可视化
  runner.py    # 实验编排与验收判定
scripts/
  run_phase0.py   # 入口：跑全部实验并生成验收报告
docs/
  SRPC_DESIGN.md  # 设计文档（含 7.5 验收标准）
results/          # 自动生成的图表 / metrics.json / report.md
```

## 快速开始

```bash
pip install numpy matplotlib
python scripts/run_phase0.py            # 默认 3 seeds，约 30–60s
python scripts/run_phase0.py --seeds 5  # 更多 seeds
```

运行后自动生成 `results/report.md`（验收报告）、`results/metrics.json` 与 5 张图。

## 验收结果（设计文档 7.5，3 seeds 均值）

| 条件 | 结果 | 关键证据 |
|---|---|---|
| 1 自组织层级结构 | PASS | 感受野对齐 0.496 → 0.966，捕获率 100%；概念层 NMI 0.398（随机 0.003） |
| 2 误差随交互下降 | PASS | EMA 斜率 −7.5e−5；首/末十分位误差 0.73 → 0.018 |
| 3 自省环非零增益 | PASS | 扰动重学窗口误差 ON 0.058 vs OFF 0.069（低 16%）；扰动后稳态误差低 7% |
| 4 无反传·在线 | PASS | 静态检查无 autograd/反传依赖；逐样本在线更新 |

补充：零样本组合泛化 SR-PC 保留组合误差 0.025 vs 查表基线 1.099；事件驱动更新率 0.37 → 0.07。

完整报告见 [results/report.md](results/report.md)。

## A/B 实验设计

所有 Track-1 结论均来自**自省环 ON vs OFF** 的受控对照（同种子同环境，仅切换 `self_loop`），扰动测试采用全局感觉重映射（感知维度随机置换），不存在"回避扰动区"的捷径。
