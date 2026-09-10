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
    # --- 结构性稀疏（不变量 3：阶段 A 起就是核心算子的数学形式，非训练后裁剪） ---
    # 阶段 A 网络小（24/16/16 维），密度 0.75 维持自组织稳定性（3 seeds 自省增益全过）；
    # 阶段 B 网络大（160/128/112），DeepConfig 用 0.25 已验。
    fan_in_frac: float = 0.75     # 生成权重每列扇入占比（层1=连续感受野窗口，内部层=随机扇入）
    fan_in_dyn_frac: float = 0.75 # 自省动力学 Wdyn 每列扇入占比
    kwta_frac: float = 0.5        # k-WTA：每层保留 top-k 激活占比（结构性稀疏激活）


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
class CreditConfig:
    """§8.5 信用分配早筛（承重墙）任务与验收阈值（阶段 A 通过条件第 2 项）。

    任务：延迟 XOR —— 目标 y_t = XOR(bit0(x_{t-Δ}), bit1(x_{t-Δ}))，
    输入为最近 Δ+1 步窗口拼接，远端块（t-Δ）为唯一任务相关块、其余块
    为随机干扰（每步 2 维 = 双峰 bit 对；干扰来自其它时间块）。XOR 目标
    与任何单一输入特征零边际相关（纯相关 Hebbian 必然失败），且线性不可分
    （输出必须为 one-hot 双输出单元）。
    网络与 Phase-0 同一套 7.3 局部规则（出生即稀疏：掩码 + k-WTA）；
    对照 = 纯相关 Hebbian（§2.4：朴素 Hebbian ≠ 误差驱动 PCN）。
    判据：误差驱动显著优于纯相关（误差机制承载信用分配），且误差能量
    回传驱动远端权重（远端块权重变化占比 > 机会水平 1/(Δ+1)）。

    **种子间方差修复（v2，3 seeds 全过）**：
    - bit 编码 0/1（原 0.2/0.8）：低输入强度位型 (0,0) 的远端感知弱是
      seed1 掉队根因（dbg27：位型准确率 0.425 < 随机）；0/1 双峰距离最大，
      低强度位型同样产生可区分表征，XOR 语义不变；
    - alpha=1.5（更强顶层类拉动）：类质心坍缩缓解（x2cos 0.886→分离）；
    - err 臂深迭代收敛（settle_iters=32, eta_inf=0.09）；
    - hebb 臂浅迭代（hebb_settle_iters=8）+ 自由推断训练（hebb_free）：
      纯相关是瞬时联想、无需深迭代；钳制 yoh 会泄漏类信息抬高基线，
      自由推断（仅输入驱动表征）使基线贴近随机（~0.51），gap 判据成立。
    """

    # 任务几何
    d_feat: int = 2                # 每步特征维（前两维为双峰 bit；其余块为随机干扰）
    delay: int = 4                 # 长程延迟 Δ（目标取决于 Δ 步前的输入）
    bit_lo: float = 0.0            # bit 低电平编码（0/1 双峰，原 0.2/0.8 的方差修复）
    bit_hi: float = 1.0
    # 网络（x0 -> x1 -> x2 -> one-hot 输出，与 Phase-0 感官通路同构）
    h1: int = 32                   # L1 隐层维度（时间分块感受野；128/64→32/16 降谱半径）
    h2: int = 16                   # L2 隐层维度（随机扇入，联合特征层）
    x_max: float = 5.0
    settle_iters: int = 32         # 局部推断收敛迭代（误差回传的深度；err 臂）
    hebb_settle_iters: int = 8     # hebb 臂浅迭代：纯相关瞬时联想，深迭代反抬基线
    eta_inf: float = 0.09          # 推断阻尼步长（固定；谱半径 ~8 下的收敛步长）
    # 7.3 规则 1 系数：顶层类拉动 α 必须 >> 底层重建 β（XOR 类条件均值相同，
    # 重建误差无法分位，类分离只能由 top-down 原型拉动提供）
    alpha: float = 1.5
    beta: float = 1.0
    theta_event: float = 0.01
    eta_out: float = 0.1           # 自由输出模式下 x3 的推断步长
    # 7.3 规则 2
    eta_w: float = 0.05            # 权重学习率（信用分配臂调优值）
    theta_syn: float = 1e-2
    # 不变量 3：出生即结构稀疏（阶段 A 密度 0.75 维持自组织稳定性）
    fan_in_frac: float = 0.75
    kwta_frac: float = 0.5
    kwta_on: bool = True
    hebb_free: bool = True         # hebb 臂训练用自由推断（无 yoh 钳制，消除类泄漏）
    predict_mode: str = "compare"  # "compare" 钳制-比较 / "free" 自由推断+读出头
    energy_mode: str = "full"      # 分类比较口径："full"=e0+e1+e2 / "class"=e1+e2（排除类无关重建噪声）
    # 协议
    train_steps: int = 10000
    eval_steps: int = 1500        # 冻结评估样本数（与 verify_final 验证口径一致；500 下种子间噪声大）
    # 验收阈值（§8.5：需量化阈值；承重墙 = 承重墙，早筛不过立即回头）
    acc_pcn_min: float = 0.80      # 误差驱动在长程延迟上准确率 >= 80%（机会 50%）
    acc_hebb_max: float = 0.68     # 纯相关必须 <= 68%（否则判据不具区分力）
    acc_gap_min: float = 0.30      # 误差驱动 - 纯相关 >= 30 个百分点
    distal_share_min: float = 0.25 # 远端块权重变化占比 >= 25%（机会水平 20%）


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
    # B 收尾 W1 实证：alpha=0.4 + kwta=0.8 提升 x1 输入保真度
    # （LS 冻结解 flip_h 色块格正确率 0.82 -> 0.90；kwta=0.5 时 top-50% 剪枝
    # 会丢掉色块细节，见 docs/SRPC_DESIGN.md §8 里程碑 B 收尾记录）。
    alpha: float = 0.40
    beta: float = 0.25
    # B 收尾 W1：推断步长。谱证据（probe_w1z 实测）：eig_max(W1^TW1) 出生
    # 47.97（连续感受野窗口重叠 + 非负元素 -> 列高相关）-> 训练后 3.00
    # （Hebbian + 列归一化去相关）。训练后线性谱已稳，>3 内迭代精度退化
    # （0.96 -> 0.69@8iters）源于非线性环：k-WTA 剪枝 + x_max 截断 + 顶层
    # 先验持续拉动的混沌传播（x2 触截断、x1 差异振荡放大）——架构债务 #2，
    # 由 E0 承接（见 results_phaseB/report_w1.md §3）。
    eta_inf: float = 1.0
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
    # --- 读出头（翻译器）学习规则（B 收尾 W1 实证） ---
    # NLMS（旧默认）在病态 x1 特征上收敛极慢：微缩实验 3000 步后 flip_h 色块格
    # 正确率仅 ~0.40（e 停在 ~1.5-2），远低于 LS 冻结解上限 0.90 —— 病态条件数
    # 拖死梯度类步长。RLS（递归最小二乘，逐样本、逐输出维独立、免反传，标准
    # 自适应滤波）4000 步内逼近 LS 上限（cell 0.993 / 色块格 0.894，kwta0.8+alpha0.4）。
    # 读出头是 §7.7 机械外围翻译器（无自主动力学），RLS 的 P 矩阵为其内部状态，
    # 与核心的 Hebbian 局部规则不冲突（核心仍为误差驱动局部学习）。
    ro_alg: str = "rls"           # 读出头学习规则："rls"（递归最小二乘）/"nlms"（归一化 LMS，旧路径保留对照）
    ro_rls_lam: float = 0.999     # RLS 遗忘因子（1=无穷记忆；小则更快丢弃旧样本）
    eta_wout: float = 0.08      # 读出层 W_out 学习率（输入->输出映射；仅 nlms 使用）
    readout_gate: float = 1e-2  # 读出学习突触前活跃门限（仅 nlms 使用）
    ro_norm: str = "clip"     # 读出权重更新后归一化："full"=列单位 L2（与生成权重一致）/"clip"=列范数超 ro_norm_cap 时投影回该球面（保稳定性+允许大尺度）
    ro_norm_cap: float = 6.0  # clip 模式的列范数上限（LS 解列范数均值 ~1.5、最大 ~4.5；cap=6 留裕量防 runaway）
    # 读出头源模式（E0 记忆-读出耦合实验矩阵，v1.6）：
    #   "recon" = 核心重建 ŝ=W1@x1（默认主路径，符号空间，翻转/旋转/重着色为精确线性映射，
    #             读出头逼近置换矩阵；B 收尾 W1 实证：LS+随机扇入掩码 0.74 vs 稠密 0.99，
    #             随机掩码与置换结构冲突 -> 稠密头）
    #   "self"  = 顶层 x_L（E0-b1：记忆/条件先验直接拉动该层，耦合零传播延迟；
    #             风险 = 符号保真回退，旧配置 cell 0.92/grid 0.0，v2 阈值下重测）
    #   "dual"  = concat[ŝ, x_L]（E0-b2：保真走 ŝ 路径、记忆耦合走 x_L 路径各取所长）
    #   "x1"    = 原始 x1（旧对照路径，随机扇入掩码 + NLMS）
    ro_recon_mode: str = "recon"
    # --- 结构性稀疏（不变量 3：从阶段 A/B 起就是核心算子的数学形式，非训练后裁剪） ---
    # 出生即定型掩码 + k-WTA 激活；学习只更新已有突触（*= mask），结构由构造保证。
    fan_in_frac: float = 0.25     # 生成权重每列扇入占比（层1=连续感受野窗口，内部层=随机扇入）
    fan_in_ro_frac: float = 0.40  # 读出头每 self 维扇入占比
    fan_in_dyn_frac: float = 0.25 # 自省动力学 Wdyn 每列扇入占比
    kwta_frac: float = 0.80       # k-WTA：每层保留 top-k 激活占比（结构性稀疏激活）
    # B 收尾 W1：0.5 -> 0.8（LS 冻结解 flip_h 色块格 0.82 -> 0.90；kwta=0.5 剪掉色块细节）
    trace_energy: bool = False    # 有效 MAC 记账（阶段 C 部署度量；默认关，零开销）


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


# ----------------------------------------------------------------------
# 阶段 E2（规模化语言训练）：µPC 适配 + 1/2/4M 梯子 + BP 孪生对照
# 对应 docs/ROADMAP.md 阶段 E2 / docs/SRPC_DESIGN.md §9-2
# ----------------------------------------------------------------------

@dataclass
class E2Config:
    """E2 规模化语言训练配置（字节级 next-byte 预测，BPC 主指标）。

    **任务**：真实语料（tinyshakespeare，1115394 字节，65 个有效字节值）
    的 next-byte 预测。窗口 = 最近 W 步字节一热（256 维正交基底，E1 前端
    直接复用），目标 = 下一字节。顺序流式单样本在线协议（逐样本增量更新，
    无 batch / 无回放 / 无 shuffle，与阶段 A/B 同协议）。
    训练/验证按位置顺序切分（前 90% / 后 10%，LM 标准口径）。

    **网络（LMPCN，LangPCN 的 LM 化）**：
        x0(W×256 一热) -> x1(h, 时间块感受野+随机扇入) -> x2(h, 随机扇入)
        -> x3(256 类 logits)
    训练钳制 x3=目标 one-hot（误差驱动局部规则，同 7.3）；评估自由推断，
    读收敛态 x3（= min ½||x2−W3·x3||² 的 LS 解，近正交列下≈线性读出），
    softmax(x3/τ) 计 BPC——τ 为读出温度标量（锚点模型校准切片上定，
    随超参一并迁移验证）。

    **µPC 适配（Innocenti et al. 2025, arXiv:2505.13124 的宽度规则锚定化）**：
    µPC 原文：层前乘子进能量（输入 N0^-1/2、隐藏 (NL)^-1/2、输出 N^-1）
    + 权重/活动学习率跨宽度零成本迁移。SR-PC 网络非标准（列归一/k-WTA/
    结构掩码/生成方向），按各信号路径 O(1) 稳定性推导宽度指数，**锚定
    h_ref=768（1M 档）**：锚点处全部超参 = E1 谱系值，宽度偏离仅由显式
    前乘子/步长缩放补偿，迁移有效性由 4M 重调对照实验裁决：
    - s1 = (h_ref/h)^{1/2}：W1 能量前乘子——前向 ŝ0=s1·W1·x1 的活跃
      突触数 ∝ kwta·h，s1 保持 e0 幅度 O(1)（=> η1 不随宽度变）；
    - et1 = eta_inf·(h/h_ref)^{1/2}：x1 活动步长补偿 s1 的识别方向衰减
      （µPC"活动学习率"对应物）；
    - η2 = η3 = eta_w·(h_ref/h)^{1/2}：e1/e2 范数 ∝ √h（列归一下
      ||Δcol|| ∝ η·||e||），保持每样本列旋转角度 O(1)（µP 输出层
      lr 缩放的同族修正）；η1 恒定（||e0|| 由 s1 稳住）；
    - s2 = s3 = 1：W2 双向（√(h2/h1)·√kwta 与 √kwta）、W3 识别方向
      （√kwta）在 h1=h2 下自然 O(1)。

    **梯子**：宽度 h ∈ {768, 1200, 1856}（h1=h2，16 整除保证锚点均衡），
    结构参数 ≈ {0.98M, 1.9M, 3.9M}；W=16 与 d0=4096 固定（隔离宽度轴，
    µP/µPC 迁移的标准设定）。8M/15M 登顶 = GPU_TASKS T1。

    **孪生对照（TwinMLP）**：参数量匹配（= PCN 结构参数）的稠密 2 隐层
    MLP + ReLU + softmax CE + Adam（batch 32，同样本流）——标准反传
    参照，不属 SR-PC 构造（GPU_TASKS 契约第 2 条）。

    **iPC 增量调度（Salvatori et al. ICLR 2024 的上下文课程化）**：
    远端上下文块掩码 4→8→16 三段展开（同一网络，逐步见到更长有效
    上下文），1M 档消融验证是否采纳。

    验收（预注册，ROADMAP E2）：
    1. 单调性：BPC(4M) < BPC(2M) < BPC(1M)；
    2. 斜率平行：PCN 每倍增增益 >= 0.5×孪生每倍增增益（且 > 0）；
    3. 4M 档 BPC <= 1.5×孪生 4M 档 BPC；
    4. µPC 迁移：4M 重调相对迁移超参的增益 <= 0.03 BPC。
    沙箱预算为 pilot 口径（150k 步 ≈ 13% epoch），最终判定以全额预算
    + GPU 登顶跑为准。
    """

    # 语料与协议
    corpus_path: str = "data/tinyshakespeare.txt"
    val_frac: float = 0.10
    context: int = 16                  # W：上下文字节块数（d0 = 16×256 = 4096）
    seed: int = 0
    # 网络几何
    ladder_widths: tuple = (768, 1200, 1856)   # h1=h2（16 整除，锚点均衡）
    h_ref: int = 768                   # µPC 适配锚定宽度（= 1M 档）
    rf_blocks: int = 2                 # L1 单元感受野 = 相邻 rf_blocks 个时间块
    # 不变量 3：出生即结构稀疏
    fan_in_frac: float = 0.75          # W2/W3 随机扇入占比（不变量 3）
    kwta_frac: float = 0.5
    kwta_on: bool = True
    kwta_every_iter: bool = False      # 宽网络默认仅循环末一次（死锁修复）
    w3_scale: float = 1.0              # W3 读出头列范数尺度（调试：x2 幅度失配）
    w3_norm: str = "unit"              # W3 归一化："unit"=列单位范数 / "clip"=列范数上限
    w3_norm_cap: float = 6.0           # clip 模式列范数上限（防发散）
    w1_norm_eps: float = 1e-8          # W1c unit 分支列范数除零下限（默认 1e-8；实验：提高仅推迟发散）
    w1_norm: str = "unit"              # W1c 归一（实验开关，默认 unit=原始行为）：
    #                                   "unit"=列单位范数（能量守恒；大 h 长训下可能失稳，见
    #                                   results_e2_gpu/REPORT_E2_GPU_SUMMIT.md）/
    #                                   "clip"=列范数仅截上限（实验：止崩但性能 ~+0.8 BPC）
    x_max: float = 5.0
    # 稳定性修复：W2 周期谱截断（阶段 E2 GPU 登顶失稳的对因修复，2026-09-08）
    # 根因：超长预算 × 宽网络的 W2 列对齐塌缩，σmax(W2) 由健康 ~3-4 涨到 ~37，
    #  自由推断（无引导）收缩性丢失 -> x2 饱和死锁发散（见
    #  results_e2_gpu/REPORT_E2_GPU_SUMMIT.md §5 与本日研究结论）。
    # 修复：每 w2_cap_every 步对 W2 做主奇异值谱截断至 w2_smax_cap（局部规则 +
    #  结构稀疏不变；健康档 σmax≈3-4 远低于 8，cap 不触发即零影响）。
    w2_cap: bool = False        # 开启周期 W2 谱截断（默认关，保持原行为/校验不变）
    w2_cap_every: int = 2000    # 每 N 步执行一次（>1 周期；=1 每步，配幂迭代法开销可忽略）
    w2_smax_cap: float = 8.0    # σmax(W2) 上限
    w2_pow_iters: int = 6       # 幂迭代次数（估计 W2 顶奇异值，O(n²)，免每步 SVD）
    # 稳定性修复：自由推断收缩步长（阶段 H1 "越深越崩"的对因修复，2026-09-10）
    # 根因：自由推断 x2⇄W3⇄x3 闭环非收缩（σmax(W3)²≈25>>1），深迭代进入 2 周期
    #  符号翻转振荡 + 能量上升发散（EVPE 爆发）→ 读出头 acc→0.01/BPC→16.3。
    #  （诊断证据：deepinfer_trace.json 的 flip_frac→0.99/x2_mean 单调漂升；
    #   文献：Mali et al. 收缩界 step<1/Lipschitz；Ha et al. ICLR'26 Meta-PCN EVPE。）
    # 修复：自由推断期把 x2/x3 更新步长（et2/eta_out）乘 eta_inf_scl 压入收缩界。
    #  已验证 eta_inf_scl=0.5：它ers=12..48 全程 acc 不崩、BPC 单调改善、x2_mean 恒定。
    #  默认 1.0 保持既有 E 档行为不变；深推断协议设 0.5（配合加深深迭代至 16-48）。
    eta_inf_scl: float = 1.0     # 自由推断步长收缩因子（<1 收缩；0.5 为已验证稳定值）
    # 推断（锚点值，E1 谱系）
    settle_iters: int = 12             # 锚点网格 {6,12} 裁定
    eta_inf: float = 0.09
    alpha: float = 1.5
    beta: float = 1.0
    theta_event: float = 0.01
    eta_out: float = 0.1
    tau: float = 0.5                  # 读出温度（训练固定；评估经校准切片定）
    # 独立读出头（阶段 B §7.7 翻译器谱系；E2 类非均匀，W3 LS 读出去
    # 频率先验 => 线性头 W_out + bias，列幅度 ∝ 频率承载 unigram）
    readout_lr: float = 0.05          # 读出头 LMS 学习率
    readout_tau: float = 0.1          # 读出头训练温度（自由推断 x2 判分）
    readout_iters: int = 8            # 读出头自由推断浅迭代（省算力，评估深迭代）
    # H2 对因修复：CE 判别耦合进编码器（2026-09-10）
    # 根因（h2_diag_mech）：free x2 线性可读 acc 仅 ~0.28 vs 孪生 0.49；clamp_x2 1.0
    #  是标签注入的平凡读。孪生=BP，CE 反向把类别信号打进全部特征；SR-PC 的输入
    #  编码器（W1/W2）只在"钳制标签注入类别信息后"间接受类别影响，开环(free)表征
    #  类别可分不足。对应文献：判别式 PC 解钳自由读出低于 BP softmax（Cacioli 2026）；
    #  teacher-forcing→free 泛化塌陷（scheduled sampling/DAgger 族）。
    # 修复：把读出头在 free x2 上的分类(CE)误差经局部误差路径回送 W1/W2——
    #   g2_ce = W_outᵀ·err（R^h），W2 += η·outer(x1, g2_ce)；
    #   g1_ce = W2ᵀ·g2_ce（R^{W·per}），W1c += η·ε(pos) outer(x0rf, g1_ce)，
    #  仍保持局部（post-类别误差 × pre-激活）、免反传、图内确定性可捕获。
    #  语义 = 把类别判别直接写进占算力大头的编码器，使 free x2 更可分。
    #  0 = 关闭（旧模型/旧行为逐位不变）。
    ce_amp: float = 0.0               # CE→编码器耦合强度（0 关；先经 300k 有界挡扫优）
    # H2 对因修复 v2：scheduled sampling（2026-09-10）
    # 诊断（h2_mech）：clamp_x2 探针 1.0 vs free_x2 探针 0.28——类别信息只能靠注入
    #  教师标签进入，开环(free)推断即丢。CE 判别耦合(300k 验证无效) 与 batch 累积
    #  (300k 验证更差，acc 卡 0.133) 均失败，确证根因 = teacher-forcing→free 泛化塌陷。
    # 文献：scheduled sampling（Bengio et al. 2015）/ DAgger（Ross et al.）/
    #  "learning the target" vid-PC（Salvatori et al.）——让钳制目标 = 真值与模型自身
    #  开环预测的凸混合，x2 被拉向"开环可达"的判别态，free 表征类别可分性随之上升。
    # 修复：online 单样本（保持稳定 0.258 轨迹）训练期，先短自由推断得 p̂，
    #  钳制目标 t3 = (1-ss_eps)·yoh + ss_eps·p̂，clamp 推断与 W3 学习都用 t3，
    #  使生成映射（x2→next-byte）与开环读出一致。0 = 关（旧行为逐位不变）。
    ss_eps: float = 0.0               # scheduled-sampling 混合权重（0 纯教师；先有界挡扫优）
    # H2 对因修复 v3：自由沉降输出误差 PC（2026-09-10）
    # 诊断（h2_layerprobe + h2_freebeta，均决定性）：硬 clamp 标签污染状态——训练时
    #  x3=yoh 被钳死沉降，W1/W2 只学会解读"标签污染态"，自由沉降从 x1 起即塌陷
    #  （clamp 1.0 vs free x1 0.32 / free x2 0.24）；放大自底向上 β 探针单调降
    #  （0.242→0.117）证伪"传导力度不足"。CE耦合/batch累积/scheduled sampling
    #  （300k/300k/90k）均失败，根因=标签经硬 clamp 注入只塑造生成轨迹、开环无判别。
    # 修复（free_nudge>0，train_step_free）：不硬钳制。单样本先自由沉降得开环态
    #  x2/free x1，读出头在其上 LMS（同基线），再经局部读头把类别误差
    #  g2=W_out·err 单步软 nudge x2（x2n），以 x2n 为生成目标重建 e1n 更新 W1/W2、
    #  并以标签为目标更新 W3（e2n=x2n−W3·yoh）。类别信号仅在自由态上以软误差进入，
    #  迫使 W1/W2 学会开环判别。0=关（free_nudge=0 走原 clamp train_step，即对照）。
    free_iters: int = 12              # 自由沉降迭代数（train_step_free 用）
    free_nudge: float = 0.0           # 读头类别软误差 nudge 强度（0 关=原 clamp 训练）
    # H2 对因修复 v4：判别式 PC 能量（DPC，Whittington&Bogacz / Salvatori 监督 PC）
    # 诊断（h2_free_sweep，决定性）：自由沉降训练(nudge)三条都逐字节同值 0.133，
    #  远低于 clamp 基线 0.23——标签只在沉降【外】事后 nudge，沉降动力学全程无类别
    #  误差，W 学不到开环判别，自由态坍缩到固定点。v4 修复：把标签回归误差作为
    #  能量一项，在自由沉降【内部】每步把 x2 软性推向正确类（−∂L_CE/∂x2），让
    #  W1/W2 在开环路径下也学到判别编码，再读头 LMS + 局部学习更新。0=关（原 clamp）。
    dpc_amp: float = 0.0              # 沉降内 CE 类别推入强度（0 关=原 clamp 训练）
    dpc_deep: float = 0.0             # 类别误差下沉 x1 的强度（充分监督 PC；0=只推 x2）
    # H2 对因修复 v5：显式识别编码器 PC（recognition-encoder PC，2026-09-10）
    # 诊断（h2_free_sweep + 前四机制全部负结果）：所有"自由沉降中推类别"的机制
    #  都受同一结构性根因拖累——x2 由生成式沉降经 12 步从零自举、开环表征判别不足
    #  （DPC 最优也只 0.275 vs 孪生 0.42）。孪生(BP)是一次前馈即得判别特征，
    #  而 SR-PC 的隐层依赖钳制标签污染态。文献：识别/生成权重孪生绑定是 tPC-RTRL
    #  （Potter & Rhodes'26）、判别式 PC 的标准结构——识别方向 = 生成权重转置。
    # 修复（recog_on，train_step_recog）：x2 不再由沉降自举，而是**一次自底向上前馈
    #   编码**得出 x1 = relu(W1cT·x0rf)、x2 = relu(W2T·x1)，读头在 x2 上 LMS；
    #   再以该识别态为锚用局部规则收紧生成侧（e0c/e1/e2 重建误差更新 W1c/W2/W3），
    #   使生成转置恰好承载判别编码。评估【同一路径】无 teacher-forcing→free 失配。
    #   0=关（走原 clamp 即对照）。结构稀疏+列归一+W2 谱截断全保留。
    recog_on: bool = False           # 显式识别编码器训练开关
    recog_refine: int = 0            # 识别编码后生成侧额外沉降迭代数（0=纯前馈编码）
    recog_act: str = "relu"          # 编码激活："relu"=clamp(0,xmax) / "tanh-like" 备用
    recog_lr: float = 0.01           # 编码器生成侧同步学习率（收敛>光谱稳定）
    # H2 对因修复 v6：识别编码器迭代信用深度（recog_rounds，2026-09-10）
    # 诊断（v5 1856@300k）：识别编码器稳定越基线但仍有 ~0.09 差距（acc 0.33 vs 0.42，
    #   BPC 3.8 vs 2.4）。根因：读头误差只沿识别路径**单趟**回送一遍（W_out→W2→W1c），
    #   编码器与读头未充分对齐判别最优。文献：EO（Error Optimization，Ahn'24）与
    #   Meta-PCN（Ha'26）都把"误差重新沉降/多步信credit"作为解决深层信号衰减的核心——
    #   单步窄 credit 等价于 BP 只反传一层的标志。
    # 修复（recog_rounds>1）：同一训练样本内把【前馈编码 → 读头 LMS → 编码器局部收紧】
    #   重复 R 轮；每轮更新权重后**重新编码**得新 x2r（误差随编码器同步演化，非同一静态
    #   误差复加），再据此重算读头误差、再次收紧 W2/W1c。等价于读头-编码器联合 R 步小步
    #   信用分配：credit depth 放大且逐轮重算误差，比盲目加大 recog_lr 稳定（对治 v5 在
    #   lr>0.001 就发散的痛点）。1=单趟（同 v5）。免反传、结构稀疏+列归一+谱截断保留。
    recog_rounds: int = 1            # 识别编码器每样本 credit 轮数（迭代细化深度）
    # H2 对因修复 v10：Split-FG 局部 BP credit（recog_fg，2026-09-10）
    # 诊断（v9 1856@200k）：沉降式判别 PC 稳定后仅 0.30，略超 recog 基线 0.287 但未闭合
    #   0.33→0.42 与孪生差距；recog(v5/v6) credit 路径缺 relu' 门控（只用 post 激活阈值
    #   掩码），BP-与-局部对 x2→x1 的 credit 差在这一层。文献：Split-FG（Ren'23）网络分
    #   主干+头，头梯度精确 + 主干用雅可比向量积(JVP)/relu' 门控估计，免反传逼近 BP。
    # 修复（recog_fg）：保持 recog 的头梯度精确（g2=W_out^T·(p−y)），把主干 credit 换成
    #   relu'(pre2) 门控的 g2g → dW2 ∝ x1⊗g2g，再经共享 W2 下沉到 x1 并 relu'(pre1) 门控
    #   g1g → dW1c ∝ x0rf⊗g1g；即精确含 relu' 的两层 BP credit，完全局部/免反传。
    #   结构稀疏（k-WTA/事件）+ 列归一 + W2 谱界保留；评估仍走 eval_recog。
    recog_fg: bool = False           # Split-FG：relu' 门控局部 BP credit（替代 post 掩码）
    # H2 对因修复 v7：非线性局部读头（readout-capacity，2026-09-10）
    # 文献（SLL Yin&Corradi'25、Error-Diffusion Yamada'26、Meta-PC Ororbia'25）：
    #   本地/免反传学习逼近 BP 的两根支柱 = 层直接任务可读性 + 读出容量。v5/v6 证实
    #   编码器判别信息过度依赖与【单层线性读头】的共演化（固定特征探针 x2r 可读仅0.13，
    #   远低于在线0.26）；单层线性读头无法把足够任务信号回传给编码器。
    # 修复（recog_mlp_ro）：读头升级为单隐层 ReLU（两层局部 LMS，全程免 autograd），
    #   把判别容量 + 任务 credit 深度（读头内多回传一层到编码器）同时增强。
    #   结构：a = relu(W_r·x2r)，logit = W_out·a；两处误差各自只沿本层权重外积更新。
    #   b_out/W_out 更新取 a 出错，读头隐藏层回传 W_r.t·e_r 再收紧编码器。0=关（单线性读头）。
    recog_mlp_ro: bool = False       # 非线性(单隐层 ReLU)局部读头开关
    recog_ro_h: int = 256            # 读头隐层宽（h_ro；0=复用线性口径）
    recog_ro_lr: float = 0.05        # 读头隐层两处局部学习率（> 底层 recog_lr）
    # H2 对因修复 v8：识别自监督 + meta-PE credit 平衡（2026-09-10）
    # 诊断（h2_recog_probe + v7 全负）：x2r 固定探针可读仅0.13 << 在线0.265 → 编码器
    #   判别信息过度依赖与读头共演化、本身非内在判别；加大读头容量(v7)不增能力且易
    #   EVPE 崩(ro512 w2smax→19)。文献（iPC Salvatori'24、tPC-RTRL Potter&Rhodes'26、
    #   Meta-PCN Ororbia'25）：识别/前向权重应被读头无关的预测目标塑形，而非只被监督头
    #   拉。修复：给识别编码器一个**读头无关的重建教师**——线性重建读头 R 把 x2r 重建
    #   回输入 x0rf（R 由本地 LMS 学），其 credit（g2_r = R^T·(rec−x0rf)）与 CE credit
    #   相加，权重 recog_ss_amp，再对 g2 做 meta-PE（RMS）归一化（recog_g_norm）抑制 EVPE。
    #   于是 W1c/W2 同时受"类别判别(CE)"+"输入保真(重建)"塑形，表征内在结构化且稳定。
    recog_ss: bool = False           # 识别自监督(重建教师,读头无关)开关
    recog_ss_amp: float = 1.0        # 重建 credit g2_r 相对 CE credit 的权重
    recog_ss_lr: float = 0.05        # 重建读头 R 的本地 LMS 学习率
    recog_g_norm: bool = True        # meta-PE：收紧编码器前对 g2 做 RMS 归一化(抑制EVPE)
    recog_g_rms: float = 2.0         # g2 归一化目标 RMS
    # H2 对因修复 v9：完全判别式 PC（µPC/Meta-PCN 风格，2026-09-10）
    # 诊断：v5 识别编码器只用 CE 的 1 层截断 credit（g2=W_out^T·err）收紧编码器 →
    #   表征不内在判别（固定探针 0.13<<在线0.265）；v3-DPC 从零冷启沉降 → 深沉降不稳。
    # 修复（recog_dpc2）：**从前馈识别态非零启动**，在判别能量
    #   E = ½||x2−relu(W2^T x1)||² + ½||x1−relu(W1c^T x0)||² + λ·CE(W_out x2)
    #   下用 relu 门控的完整 credit 沉降 x1/x2（K 步，µP 收缩步长），得到判别沉降态;
    #   再以【预测沉降目标】(target propagation) + meta-PE RMS 平衡，把类别判别目标
    #   下沉到 W2/W1c（ΔW2∝(x2set−W2^T x1)⊗x1，ΔW1c∝(x1set−W1c^T x0)⊗x0，均有
    #   relu 门控），等价逐层把 BP 的全局 credit 吸收进识别权重。
    recog_dpc2: bool = False         # 完全判别式 PC 训练开关
    dpc2_settle: int = 8             # 判别能量沉降步数 K（µP 收缩步长下稳定）
    dpc2_amp: float = 0.8            # 类别 CE 项权重 λ（相对识别残差）
    dpc2_back: float = 0.6           # x1 沉降中把判别目标下沉的回传权重
    # H2 对因修复 v11：局部 Adam + 软化正则（recog_adam，2026-09-10）
    # 诊断（决定性 oracle h2_oraclebp_1856）：识别编码器【同结构】换成真 BP CE + Adam
    #   → 90k acc0.367 / 180k 0.373，完全追平孪生（gap≈0）与局部机制 0.33 的墙无关；
    #   证明墙在【局部 credit 的优化动力学】，不在 kWTA/块紧凑/线性读头/relu' 形式。
    #   v10(Split-FG relu' credit)不应最差(0.255)：credit 路径已是 BP 同款，但被
    #   事件门控 + meta-PE RMS 归一 + 每步 unit 列归一反复抹平 → 更新方向被压爆。
    # 文献（iPC Salvatori'24 增量更新更稳、bPC Oxford'25 分类与 BP 相当标准做法=读头
    # + 自适应优化器）。修复（recog_adam）：保留 v10 的 relu' 门控两层 credit
    #   （pre2→g2g=relu'(pre2)⊙g2；dW2∝x1⊗g2g；g1=W2·g2g；pre1→g1g；dW1c∝x0rf⊗g1g），
    #   但权重更新改成【逐突触自适应矩】（局部 Adam：每突触独立 m/v，免 autograd，
    #   无全局/无反传），并软化破坏性正则：撤事件门控与 meta-PE RMS 归一（credit
    #   原值进动量），W1c 列归一改 clip(≤1) 容许弱列自由变弱，保留 W2 谱界防爆炸。
    recog_adam: bool = False         # 局部自适应矩(免反传局部Adam)训练开关
    recog_adam_lr: float = 3e-4      # 局部 Adam 学习率（对齐 oracle lr=3e-4）
    recog_adam_b1: float = 0.9       # Adam β1
    recog_adam_b2: float = 0.999     # Adam β2
    recog_adam_eps: float = 1e-8     # Adam ε
    # 学习（锚点值）
    eta_w: float = 0.005               # 锚点网格 {0.005,0.01} 裁定（探针：0.005@24iters 最优）
    theta_syn: float = 1e-2
    # 运行预算（pilot）
    anchor_steps: int = 20000          # 锚点网格每配置步数
    ladder_steps: int = 150000         # 梯子每档步数（≈13% epoch）
    control_steps: int = 20000         # 4M 重调对照每配置步数
    ipc_steps: int = 120000            # iPC 消融步数
    eval_every: int = 15000
    eval_windows: int = 800
    eval_stride: int = 70              # 验证窗口间隔（覆盖验证段全程）
    tau_grid: tuple = (0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0)  # 读出温度网格
    tau_cal_windows: int = 200         # τ 校准切片（验证段前 200 窗口）
    # 锚点/重调网格（1M 档裁定；4M 对照同网格）
    anchor_iters: tuple = (12, 24)
    anchor_eta: tuple = (0.005, 0.01)   # 探针证据：0.005@24iters 最优（BPC 5.50@10k）
    # 孪生
    twin_batch: int = 32
    twin_lr: float = 1e-3
    # 验收阈值（预注册）
    slope_frac_min: float = 0.5        # 判据 2
    twin_ratio_max: float = 1.5        # 判据 3
    transfer_gain_max: float = 0.03    # 判据 4


# ----------------------------------------------------------------------
# 阶段 E1（语言化起步）：字节级 UTF-8 词元前端 + 语言长程信用分配小任务
# 对应 docs/ROADMAP.md 阶段 E1 / docs/SRPC_DESIGN.md §9-2（自建词元前端）
# ----------------------------------------------------------------------

@dataclass
class LangConfig:
    """E1 语言长程信用分配（延迟文本关联）任务与阈值（与 §8.5 早筛同谱系）。

    任务族（字母表 A=16 个 ASCII 字母，流经 ByteTokenizer 的 256 维字节一热，
    与 ARC 符号编码 / E2 语言流同前端）：
    - **assoc（延迟关联，主验收任务）**：y_t = π(x_{t-Δ})，π = 固定随机置换。
      π 为双射且流均匀 => y 边际均匀，与任何单一输入符号**零边际相关**
      （同 §8.5 判据）——纯相关 Hebbian 一阶统计无信号，只有误差驱动的
      条件结构学习可解；同时确定性映射信号强，在线单样本可学。
    - **xorsum（延迟成对 XOR，边界记录）**：y_t = x_{t-Δ-1} ⊕ x_{t-Δ}
      （4 bit 逐位 XOR，16 类）。在线单样本学习**不可达**（v2 探针证据链，
      见下），降为在线信用分配边界的记录项，不计入验收。

    **v2 方案调整（2026-09-07，边界探针证据链）**：
    xorsum@16类 在线失败后逐层排查——(1) 超参全排除（credit 同款超参 /
    深收敛 64 iters / 学习率 0.05-1.0 / k-WTA / 掩码 / 事件门控 / 感受野
    窗口 2-3 块均无改善）；(2) **BP 在线对照同样失败**（同形状网络 +
    Adam 单样本 20k 步 ≈ 机会）——parity 为 SQ-hard 高频函数（Kearns &
    Valiant 1989; Blum et al. 1994），在线更新的期望梯度≈0，对称无法
    破缺；(3) **batch 上界 = 1.0**（mini-batch plain SGD 300 epoch 即达）
    ——任务本身可解；(4) PCN+梯度累积/经验回放均失败（k-WTA/clip 等
    非标准组件破坏收敛态≈BP 的等效性）。结论：瓶颈 = 在线协议的样本
    效率，非局部规则原理缺陷；修复路径 = F 阶段记忆回放（慢记忆 ->
    batch 等效协议，生物学对应睡眠重放），不在 E1 解决。
    assoc@Δ=16 亦为边界（远端块表征稀释：层1每单元锚定 1 块，1/17
    覆盖远端，容量不足，非原理性——扩 h1 或感受野可推远）。

    窗口 = 最近 W 步字节一热拼接（延迟线属机械外围序列化，§2.2）；
    W = Δ+1（assoc）/ Δ+2（xorsum）。远端块（窗口最前 1/2 块）为唯一
    任务相关块。网络与 CreditPCN 同一套 7.3 局部规则，多类化
    （x0 -> x1(时间分块感受野) -> x2(随机扇入) -> x3(C 类 one-hot)）。

    阈值预注册（v2，主任务 assoc，Δ=4 主判定）：
    acc_err >= 0.80、acc_hebb <= 0.20（机会 1/16=0.0625）、
    gap >= 0.40、远端块权重变化占比 >= 0.25（机会 1/(Δ+1)=0.20）；
    Δ=8 跨度外推 acc_err >= 0.80。Δ=1 近程对照为记录项。
    """

    # 任务几何
    alphabet: str = "abcdefghijklmnop"   # 16 字母（4 bit 码），i.i.d. 均匀流
    n_classes: int = 16
    delta: int = 4                       # 延迟跨度（与 §8.5 早筛 Δ=4 同起点）
    # 网络（CreditConfig 同谱系；按 256 维字节块放大隐层）
    h1: int = 128                        # L1 隐层（时间分块感受野，每单元锚定一个时间块）
    h2: int = 64                         # L2 隐层（随机扇入，联合特征层）
    x_max: float = 5.0
    settle_iters: int = 24               # err 臂推断收敛迭代（信用回传深度）
    hebb_settle_iters: int = 8           # hebb 臂浅迭代（纯相关瞬时联想）
    eta_inf: float = 0.09                # 推断阻尼步长（块正交掩码下谱半径 ~1）
    alpha: float = 1.5                   # 顶层类拉动 >> 底层重建（同 CreditConfig v2）
    beta: float = 1.0
    theta_event: float = 0.01
    eta_out: float = 0.1                 # 自由输出模式 x3 推断步长
    # 7.3 规则 2
    eta_w: float = 0.05
    theta_syn: float = 1e-2
    # 不变量 3：出生即结构稀疏
    fan_in_frac: float = 0.75
    kwta_frac: float = 0.5
    kwta_on: bool = True
    hebb_free: bool = True               # hebb 臂自由推断训练（消除类泄漏）
    energy_mode: str = "class"           # 钳制-比较能量口径（e1+e2，§8.5 同理）
    # 协议
    train_steps: int = 4000
    eval_steps: int = 600                # 自由推断评估（主口径：acc + BPC）
    compare_steps: int = 120             # 钳制-比较子样本（协议交叉验证）
    # 验收阈值（v2 预注册，主任务 assoc，Δ=4 主判定）
    acc_assoc_min: float = 0.80
    acc_assoc_d8_min: float = 0.80       # Δ=8 跨度外推
    acc_hebb_max: float = 0.20           # 纯相关对照上限（机会 1/16）
    gap_min: float = 0.40                # 误差驱动 - 纯相关
    distal_assoc_min: float = 0.25       # 远端驱动（机会 1/(Δ+1)=0.20）
    # 边界记录项（不设验收阈值）：xorsum@{4,16}（在线组合信用分配边界，
    # SQ-hard + BP 在线对照证据，修复排 F 阶段记忆回放）、assoc@16（远端
    # 表征稀释边界，容量非原理性）。

