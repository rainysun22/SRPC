# SR-PC 阶段 B 验收报告（自动生成）

总体结论：**FAIL** · 运行耗时 46s · 全程 NumPy 局部规则、免反向传播

对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B：

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| 1 能力随交互上升 | FAIL | 任务内误差斜率（跨 seed 均值）：['-3.67e-02', '-2.70e-02', '-2.73e-02', '-4.51e-02']，最差 -6.81e-02（全部 < 0） |
| 2 免遗忘（带记忆） | FAIL | 末列相对回升均值 239.2%（max 883.5%） |
| 3 记忆增益 | FAIL | 带记忆保留误差 64.7% vs 无记忆 55.6%（改善 -5%） |
| 4 组合零样本 | FAIL | 组合嵌入 1.4125 vs 随机条件 1.7113（增益 17%） |

## 补充指标

- 保留组合逐项零样本误差：{'flip_h_rot90': '2.5228', 'recolor_flip_v': '0.3022'}
- 训练变换冻结误差：{'rot90': '2.2924', 'recolor': '0.0414'}
- 事件驱动更新率（能量代理）：mem 0.866 vs no-mem 0.833

## 图表

![ability](figB1_ability.png)
![forgetting](figB2_forgetting.png)
![composition](figB3_composition.png)
![events](figB4_events.png)
