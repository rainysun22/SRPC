"""SR-PC Phase-0 全部超参数（对齐 docs/SRPC_DESIGN.md 第 7 节）。

所有实验的唯一参数来源，保证可复现。
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """SR-PC 核心模型（7.2 模块清单 / 7.3 更新规则）。

    层级（自上而下生成、自下而上只传误差）：
        x_self --Ws2--> x2 --W21--> x1 --W10--> s_hat ~= s
        自省环：Wdyn 预测 x_self(t+1) | x_self(t), a(t)  -->  e_self
    """

    d_obs: int = 32            # 感觉维度 D（低维环境感知）
    n_l1: int = 24             # L1：感觉通路第一层（感受野/部件层）
    n_l2: int = 16             # L2：感觉通路第二层（概念/组合层）
    n_self: int = 16           # x_self：自我模型层（不变量 4：从一开始就在）
    x_max: float = 5.0         # 状态上界（非负稀疏表征）
    inner_iters: int = 3       # 每步局部推断迭代次数
    # --- 7.3 规则 1：局部推断（稀疏/事件驱动） ---
    alpha: float = 0.15        # 系数 a：来自上层的预测误差（拉动 x_l 向上层预测靠拢）
    beta: float = 0.25         # 系数 b：来自下层的误差信号（自下而上只传误差）
    theta_event: float = 0.01  # 事件阈值：|更新|>theta 的节点才更新（不变量 3：能量内生·稀疏）
    # --- 7.3 规则 2：局部 Hebbian 学习（免反传） ---
    eta_w: float = 0.06        # 生成权重（W10/W21/Ws2）学习率
    theta_syn: float = 1e-2    # 突触前活跃门限：仅活跃节点的突触列参与更新（稀疏学习）
    # --- 7.3 规则 3：自省环 ---
    eta_dyn: float = 0.03      # 自我预测器 Wdyn 的局部 LMS 学习率
    dyn_decay: float = 1e-3    # Wdyn 权重衰减（稳定性）
    kappa_boost: float = 1.5   # 精度调制增益：eta_t = eta*(1+kappa*(surprise-1))^+，上限 boost_max
    boost_max: float = 6.0
    ema_self_rate: float = 0.005  # ||e_self|| 的 EMA 基线（慢适应：扰动后 boost 可覆盖重学期）
    u_action_rate: float = 0.10   # 每动作"预期自省误差"EMA 更新率（7.2 模块 5）
    u_regress: float = 0.005      # 久未执行动作的 U 向均值回归率（不确定性回升，保持探索）
    curiosity: float = 3.0        # 主动推理的认识价值项系数：少试的动作按 1/sqrt(n) 折价探索
    # --- 7.2 模块 4：全局工作空间（MVP：高置信收敛信念子集） ---
    ws_k: int = 4              # 广播的 top-k 信念


@dataclass
class FieldConfig:
    """Track-1 环境：SourceFieldWorld（2-D 场地 + 信息源，交互式导航）。"""

    n_sources: int = 5
    source_pos: tuple = ((0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80), (0.50, 0.50))
    kernel_sigma: float = 0.13   # 邻近核宽度（决定单源/组合感受范围）
    move_step: float = 0.12
    move_noise: float = 0.02
    obs_noise: float = 0.03
    pattern_active: int = 7      # 每个源模式的活跃维数（相邻源轻度重叠）
    pattern_stride: int = 5      # 活跃维块间距（< active 表示相邻源共享若干维）
    # 修正测试扰动：全局感觉重映射（感知维度随机置换，无论智能体在哪都被迫重学信念）
    perturb_kind: str = "remap"


@dataclass
class SlotConfig:
    """Track-2 环境：SlotWorld（合成组合流，验证"片段重组出新概念"，7.4 组合）。"""

    n_features: int = 6
    # 15 个特征对中留 4 个作保留组合（训练不可见），每个特征仍在训练对中出现
    train_pairs: tuple = ((0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 5), (3, 4), (4, 5))
    novel_pairs: tuple = ((0, 4), (1, 5), (2, 4), (3, 5))
    single_prob: float = 0.25    # 训练流中单特征上下文占比（提供"纯净部件"片段）
    ctx_min: int = 20            # 一个上下文持续的最少步数（跨时段结构）
    ctx_max: int = 60
    obs_noise: float = 0.03
    pattern_active: int = 7
    pattern_stride: int = 5


@dataclass
class Track1Config:
    """Track-1 实验协议（形成/修正/自省 A/B）。"""

    steps: int = 12000
    perturb_step: int = 6000
    eps_start: float = 0.40      # 主动推理的探索率（探索/利用均衡）
    eps_end: float = 0.20
    eps_decay_steps: int = 4000
    ema_alpha: float = 0.01      # 误差曲线平滑
    recovery_base_win: int = 150
    recovery_mult: float = 1.30  # 恢复阈值 = 扰动前基线 * mult
    recovery_sustain: int = 60
    relearn_lo: int = 300        # 扰动后"重学窗口"起点（跳过 EMA 滞后）
    relearn_hi: int = 2500       # 重学窗口终点


@dataclass
class Track2Config:
    """Track-2 实验协议（组合泛化 + 基线对照 + 自省增益）。"""

    train_steps: int = 6000
    fewshot_steps: int = 500     # 保留组合的少样本在线适应阶段
    probe_samples: int = 30      # 冻结评估时每个上下文的采样数


@dataclass
class AcceptanceConfig:
    """7.5 验收标准的量化阈值（阶段 A 通过条件）。"""

    rf_unit_cos: float = 0.75        # 一个源/特征被"捕获"：存在感受野与其余弦 >= 该值
    rf_captured_frac: float = 0.80   # 至少 80% 的源/特征被捕获
    rf_improve_min: float = 0.15     # 感受野对齐相对初始化的最小提升
    nmi_p_max: float = 0.05          # 概念层-上下文 NMI 的置换检验 p 值上限
    slope_max: float = 0.0           # 预测误差 EMA 斜率必须 < 0（随交互下降）
    self_gain_min_frac: float = 0.10  # 自省环开启后扰动重学误差至少低 10%（A/B 主指标）
