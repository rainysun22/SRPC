# SR-PC Phase-0 验收报告（自动生成）

总体结论：**PASS** · 运行耗时 33s · 全程无反向传播、在线增量

对应 docs/SRPC_DESIGN.md 7.5 验收标准：

| 条件 | 结果 | 关键证据 |
|---|---|---|
| 1 自组织层级结构 | PASS | 感受野对齐 0.966（初始 0.496），捕获率 100%；概念层 NMI 0.398（随机 0.003）；组合流特征捕获率 100% |
| 2 误差随交互下降 | PASS | EMA 斜率 -7.48e-05；首/末十分位误差 0.7279 → 0.0185 |
| 3 自省环非零增益 | PASS | 扰动重学窗口误差 ON 0.0583 vs OFF 0.0690（低 16%）；恢复步数 ON 3425 vs OFF 3081；扰动后稳态误差 ON 0.0187 vs OFF 0.0202 |
| 4 无反传·在线 | PASS | 纯 NumPy 局部规则；逐样本在线更新 |

## 补充指标（7.4 组合 / 不变量 3 能量）

- 零样本组合泛化：SR-PC 保留组合误差 0.0251 vs 查表基线 1.0986（训练组合 0.0202 vs 0.0180）
- 片段重组：保留组合 x1 重组余弦 0.999（训练组合 0.998），工作空间重组余弦 0.942
- 新概念新颖性（x2 模式距离比）：0.72
- 少样本适应（保留组合）：ON 增益 0.0465 vs OFF 0.0521
- 事件驱动更新率：前期 0.369 → 末期 0.072

## 图表

![error_curves](fig1_error_curves.png)
![receptive_fields](fig2_receptive_fields.png)
![structure](fig3_structure_nmi.png)
![combination](fig4_combination.png)
![events](fig5_event_rate.png)
