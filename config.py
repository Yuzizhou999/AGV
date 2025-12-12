"""
配置文件：环形轨道双工位穿梭车调度系统
"""

# ========== 环境配置 ==========
TRACK_LENGTH = 100.0  # 环形轨道长度
MAX_VEHICLES = 2  # 车辆数量
MAX_SPEED = 5.0  # 最大速度 (单位/秒)
MAX_ACCELERATION = 1.0  # 最大加速度
SAFETY_DISTANCE = 2.0  # 安全距离
ALIGNMENT_TOLERANCE = 1.0  # 对齐容差（增大以便更容易装卸货）

# ========== 上料口配置 ==========
NUM_LOADING_STATIONS = 2  # 上料口数量
LOADING_POSITIONS = [20.0, 30.0]  # 上料口位置
LOADING_STATION_SLOTS = 2  # 每个上料口的工位数（1#, 2#）

# ========== 下料口配置 ==========
NUM_UNLOADING_STATIONS = 3  # 下料口数量
UNLOADING_POSITIONS = [60.0, 70.0, 80.0]  # 下料口位置
UNLOADING_STATION_SLOTS = 2  # 每个下料口的工位数

# ========== 货物配置 ==========
ARRIVAL_INTERVAL_MIN = 5  # 货物最小到达间隔(秒)
ARRIVAL_INTERVAL_MAX = 15  # 货物最大到达间隔(秒)
CARGO_TIMEOUT = 120.0  # 货物超时时间(秒)

# ========== 操作配置 ==========
LOADING_TIME = 15.0  # 单工位上料耗时(秒)
UNLOADING_TIME = 15.0  # 单工位下料耗时(秒)

# ========== 仿真配置 ==========
EPISODE_DURATION_TRAIN = 600  # 训练时使用600秒（10分钟）以加速训练
EPISODE_DURATION_EVAL = 8 * 3600  # 完整评估使用8小时（28800秒）
EPISODE_DURATION = EPISODE_DURATION_TRAIN  # 默认使用训练时长
HIGH_LEVEL_DECISION_INTERVAL = 0.5  # 高层决策时间间隔(秒)
LOW_LEVEL_CONTROL_INTERVAL = 0.5  # 低层控制时间间隔(秒)

# ========== 奖励参数 ==========
REWARD_DELIVERY = 200.0  # 完成卸货奖励
REWARD_PICKUP = 50.0  # 成功取货的奖励
REWARD_TIMEOUT_PICKUP = 20.0  # 取走超时货物的额外奖励（优先级激励）
REWARD_WAIT_PENALTY_COEFF = 0.01  # 等待惩罚系数
REWARD_TIMEOUT_PENALTY = -50.0  # 货物超时惩罚
REWARD_TIMEOUT_WAIT_PENALTY_COEFF = 0.2  # 超时货物的等待惩罚系数
REWARD_SAFETY_VIOLATION = -100.0  # 安全距离违反惩罚
REWARD_SPEED_CHANGE_PENALTY = -0.05  # 速度变化惩罚系数，减轻抖动成本

# ========== 神经网络配置 ==========
HIDDEN_DIM = 128  # 隐层维度
LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 32  # 批大小
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.05  # 最终探索率（提高下限）
EPSILON_DECAY = 0.998  # 探索率衰减（放缓衰减）

# ========== 训练配置 ==========
NUM_EPISODES = 50  # 训练轮数（减少以加速测试）
MAX_STEPS_PER_EPISODE = int(EPISODE_DURATION / LOW_LEVEL_CONTROL_INTERVAL)  # 每轮最大步数
