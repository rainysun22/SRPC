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


# ----------------------------------------------------------------------
# 阶段 B（规模化与组合）：DeepSRPC / 多时间尺度记忆 / ARC-lite / 顺序学习
# 对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B
# ----------------------------------------------------------------------

@dataclass
class DeepConfig:
    """阶段 B 深层预测编码网络（规模化：Phase-0 的 3 层 -> 可配置 L 层）。

    dims[0] 为输入维（s），dims[-1] 为自我层维（x_self），内部层数 L = len(dims)-1。
    自上而下生成预测、自下而上只传误差；事件驱动稀疏（不变量 3）；
    自省环 + 记忆先验 + 变换条件经顶层接入（不变量 4 / 阶段 B 里程碑）。
    """

    dims: tuple = (256, 160, 128, 112, 128)  # 输入维(网格一热) + 内部层维 + 自我层维
    x_max: float = 5.0
    inner_iters: int = 3
    # 规则 1：局部推断（稀疏/事件驱动）
    alpha: float = 0.15
    beta: float = 0.25
    theta_event: float = 0.01
    # 规则 2：局部 Hebbian 学习（免反传）
    eta_w: float = 0.05
    theta_syn: float = 1e-2
    # 规则 3：自省环
    eta_dyn: float = 0.03
    dyn_decay: float = 1e-3
    kappa_boost: float = 1.5
    boost_max: float = 6.0
    ema_self_rate: float = 0.005
    # 阶段 B：多时间尺度记忆先验（顶层拉动，按任务分组免遗忘）
    gamma_mem: float = 0.35
    # 阶段 B：ARC 变换条件先验。
    # Uc 为固定分块正交码（非负、不相交支撑 -> 任务编码天然分离，见 set_condition）；
    # 条件直接拉动 x_self 更新（beta_cond 为拉动强度），变换知识由读出层 W_out 学习。
    beta_cond: float = 0.80
    eta_wout: float = 0.08      # 读出层 W_out 学习率（输入->输出映射）
    readout_gate: float = 1e-2  # 读出学习突触前活跃门限
    # --- 阶段 C：结构稀疏（出生即定型，非训练后裁剪；0/False = 稠密旧路径） ---
    fan_in_frac: float = 0.0      # 生成权重每列扇入占比（层1=连续感受野窗口，内部层=随机扇入）
    fan_in_ro_frac: float = 0.0   # 读出头每 self 维扇入占比
    fan_in_dyn_frac: float = 0.0  # 自省动力学 Wdyn 每列扇入占比
    kwta_frac: float = 0.0        # k-WTA：每层保留 top-k 激活占比（结构性稀疏激活）
    trace_energy: bool = False    # 有效 MAC 记账（三口径：事件驱动/结构/稠密等价）


@dataclass
class MemoryConfig:
    """阶段 B 多时间尺度原型记忆（免遗忘结构，不变量 2：能力=记忆·拼合）。

    fast 槽：工作记忆，学习率快，可快速覆盖（当前会话/任务）；
    slow 槽：长期记忆，按任务分组（n_slow_group 个/组），学习率慢，
    只在"稳定/重复"时小幅巩固；顺序学习新任务只写新任务的组，
    旧任务原型固化在旧组 -> 结构上免遗忘（新任务不破坏旧任务）。
    """

    d: int = 12
    n_fast: int = 12
    n_slow_group: int = 2       # 每任务组的长期记忆槽数
    rate_fast: float = 0.15
    rate_slow: float = 0.01
    conf_thresh: float = 0.30   # 自省误差小（stable>=thresh）才巩固


@dataclass
class ArcConfig:
    """阶段 B 组合基准：ARC-lite（8x8 网格变换任务 + 保留组合零样本）。

    训练变换作为顺序学习任务序列；novel_combos 为两个已知变换的组合
    （训练不可见），验证"既有片段重组出新概念"（阶段 B 里程碑）。
    """

    grid: int = 8
    n_colors: int = 4           # 0=空, 1..3 颜色
    n_blocks: int = 3           # 每输入随机放置的块数上限
    max_block: int = 2          # 块最大边长
    train_transforms: tuple = ("flip_h", "flip_v", "rot90", "recolor")
    novel_combos: tuple = (("flip_h", "rot90"), ("recolor", "flip_v"))


@dataclass
class CLConfig:
    """阶段 B 顺序学习（免遗忘）协议。"""

    steps_per_task: int = 800   # 每个变换任务的训练步数
    eval_samples: int = 24      # 冻结评估每任务采样数
    settle: int = 6             # 评估收敛步（跳过瞬态）


@dataclass
class AcceptanceBConfig:
    """阶段 B 可证伪里程碑（第 8 节）量化阈值。"""

    learn_slope_max: float = 0.0        # 能力上升：任务内误差 EMA 斜率 < 0
    forget_rel_max: float = 0.25        # 免遗忘：带记忆时旧任务误差相对回升 <= 25%
    retain_gain_min: float = 0.10       # 记忆增益：保留误差相对无记忆改善 >= 10%
    combo_gain_min: float = 0.20        # 组合嵌入相对随机条件的零样本增益 >= 20%


# ----------------------------------------------------------------------
# 阶段 C（软件版内在化）：结构稀疏核心 + 能耗记账 + 能力复验
# 对应 docs/SRPC_DESIGN.md 不变量 3（能量内生·结构稀疏）与 6.1b（大模型能效对照）；
# 硬件部署本体（事件驱动/低比特芯片）deferred 至有专用硬件时
# ----------------------------------------------------------------------

@dataclass
class PhaseCConfig:
    """阶段 C 实验协议（无专用硬件，CPU/GPU 软件验证）。

    核心：出生即结构稀疏（扇入受限分块权重 + k-WTA）的核心从头重训，
    复跑阶段 B 全部验收（能力无回撤 = 低功耗来自架构本身，非稠密裁剪），
    并以硬件无关的有效 MAC 记账对比大模型标尺。
    """

    yardstick_params: tuple = (0.5e9, 1.5e9, 7e9)   # 大模型标尺参数量（6.1b 对照线）
    yardstick_names: tuple = ("LLM-0.5B", "LLM-1.5B", "LLM-7B")
    tokens_per_task: int = 512      # 大模型单任务推理 token 数（网格序列化+指令+输出）
    quant_bits: int = 8             # 低比特部署就绪检查（训练后权重量化）


@dataclass
class AcceptanceCConfig:
    """阶段 C 可证伪里程碑（第 8 节 C 行，软件版）量化阈值。"""

    llm_ratio_min: float = 1e3          # C2：每样本推理 MACs 比最小标尺低 >= 10^3 倍
    density_max: float = 0.35           # C3：权重总密度上限（出生即稀疏，由构造保证）
    quant_err_ratio_max: float = 1.5    # C4：int8 量化后冻结误差 <= fp 的 1.5 倍
    # C1 能力无回撤 = 稀疏核心上阶段 B 四项验收全部复现（复用 AcceptanceBConfig）
