# SR-PC 阶段 B 验收报告（自动生成）

总体结论：**PASS** · 运行耗时 31s · 全程 NumPy 局部规则、免反向传播

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B：

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| 1 能力随交互上升 | PASS | 任务内误差斜率（跨 seed 均值）：['-9.37e-03', '-1.07e-02', '-9.69e-03', '-1.12e-02']，最差 -1.20e-02（全部 < 0） |
| 2 免遗忘（带记忆） | PASS | 末列相对回升均值 1.3%（max 2.6%） |
| 3 记忆增益 | PASS | 带记忆保留误差 9.6% vs 无记忆 12.3%（改善 22%） |
| 4 组合零样本 | PASS | 组合嵌入 0.0973 vs 随机条件 0.1651（增益 41%） |

## 补充指标

- 保留组合逐项零样本误差：{'flip_h_rot90': '0.0991', 'recolor_flip_v': '0.0954'}
- 训练变换冻结误差：{'rot90': '0.0989', 'recolor': '0.1008'}
- 事件驱动更新率（能量代理）：mem 0.956 vs no-mem 0.964

## 图表

![ability](figB1_ability.png)
![forgetting](figB2_forgetting.png)
![composition](figB3_composition.png)
![events](figB4_events.png)
