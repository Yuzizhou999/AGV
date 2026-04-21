"""
配置文件：环形轨道双工位穿梭车调度系统
"""

# ========== 环境配置 ==========
TRACK_LENGTH = 100.0  # 环形轨道长度
MAX_SPEED = 5.0  # 最大速度 (单位/秒)
MAX_ACCELERATION = 1.0  # 最大加速度
SAFETY_DISTANCE = 2.0  # 安全距离
SPEED_TOLERANCE = 0.1  # 对齐判定的速度容差，车辆速度必须≤此值才能开始上下料

# ========== 车辆与卷纸配置 ==========
VEHICLE_SLOT_COUNT = 2  # 每台小车的装载工位数
SMALL_ROLL_DIAMETER_LIMIT = 1.4  # 小卷纸双载的直径上限（米）
SMALL_ROLL_DIAMETER_RANGE = (0.8, 1.7)  # 小卷纸直径采样范围（米）
LARGE_ROLL_DIAMETER_RANGE = (1.6, 2.2)  # 大卷纸直径采样范围（米）
ROLL_TYPE_LARGE = "large"
ROLL_TYPE_SMALL = "small"

# 平滑与安全投影参数：限制加速度变化率，避免 +a 到 -a 瞬间跳变
ACTION_DIRECTION_DEADBAND = 1e-3
MIN_DIRECTION_CHANGE_INTERVAL_STEPS = 3  # 兼容旧字段，当前安全层不再用该参数做硬钳制
MAX_JERK = 1.0  # 每秒允许的最大加速度变化量，dt=0.5 时每步最多变化 0.5
SAFETY_PROJECTION_EPS = 1e-6

# ========== 车辆布局 ==========
VEHICLE_CONFIGS = [
    {"id": 0, "initial_position": 0.0},
    {"id": 1, "initial_position": 20.0},
    {"id": 2, "initial_position": 40.0},
    {"id": 3, "initial_position": 60.0},
]

# ========== 上下料口配置 ==========
# 约定：
# - 上料口集中在左半环（位置 50-100 附近）
# - 下料口集中在右半环（位置 0-50 附近）
LOADING_STATION_CONFIGS = [
    {
        "id": 0,
        "position": 56.0,
        "side": "left",
        "paper_type": ROLL_TYPE_LARGE,
        "primary_unloading_station": 0,
        "alternative_unloading_stations": [1],
    },
    {
        "id": 1,
        "position": 64.0,
        "side": "left",
        "paper_type": ROLL_TYPE_SMALL,
        "primary_unloading_station": 1,
        "alternative_unloading_stations": [0, 2],
    },
    {
        "id": 2,
        "position": 82.0,
        "side": "left",
        "paper_type": ROLL_TYPE_LARGE,
        "primary_unloading_station": 2,
        "alternative_unloading_stations": [1, 3],
    },
    {
        "id": 3,
        "position": 92.0,
        "side": "left",
        "paper_type": ROLL_TYPE_SMALL,
        "primary_unloading_station": 3,
        "alternative_unloading_stations": [2],
    },
]

UNLOADING_STATION_CONFIGS = [
    {"id": 0, "position": 8.0, "side": "right"},
    {"id": 1, "position": 18.0, "side": "right"},
    {"id": 2, "position": 32.0, "side": "right"},
    {"id": 3, "position": 44.0, "side": "right"},
]

MAX_VEHICLES = len(VEHICLE_CONFIGS)  # 车辆数量
NUM_LOADING_STATIONS = len(LOADING_STATION_CONFIGS)  # 上料口数量
NUM_UNLOADING_STATIONS = len(UNLOADING_STATION_CONFIGS)  # 下料口数量
LOADING_POSITIONS = [item["position"] for item in LOADING_STATION_CONFIGS]  # 上料口位置
UNLOADING_POSITIONS = [item["position"] for item in UNLOADING_STATION_CONFIGS]  # 下料口位置
LOADING_STATION_SLOTS = 2  # 每个上料口的工位数（1#, 2#）
UNLOADING_STATION_SLOTS = 2  # 每个下料口的工位数（保留配置，当前不显式建模容量）

# 下料口默认优先固定去向；只有备选口明显更近时才切换
ALTERNATIVE_UNLOADING_SWITCH_DISTANCE = 6.0

# ========== 货物配置 ==========
# 根据 project.md 要求：货物每隔 5-15 秒（随机整数）出现在上料口
ARRIVAL_INTERVAL_MIN = 5  # 货物最小到达间隔(秒)
ARRIVAL_INTERVAL_MAX = 15  # 货物最大到达间隔(秒)
CARGO_TIMEOUT = 120.0  # 货物超时时间(秒)

# ========== 操作配置 ==========
LOADING_TIME = 15.0  # 单工位上料耗时(秒)
UNLOADING_TIME = 15.0  # 单工位下料耗时(秒)

# ========== 仿真配置 ==========
EPISODE_DURATION = 8 * 3600  # 仿真时长 8 小时
HIGH_LEVEL_DECISION_INTERVAL = 1.0  # 高层决策时间间隔(秒)
LOW_LEVEL_CONTROL_INTERVAL = 0.5  # 低层控制时间间隔(秒)

# ========== 奖励参数 ==========
REWARD_DELIVERY = 20.0  # 完成卸货奖励
REWARD_PICKUP = 20.0  # 完成取货奖励
REWARD_ASSIGNMENT = 20.0  # 分配货物给小车的奖励
REWARD_WAIT_PENALTY_COEFF = 0.00002  # 等待惩罚系数
REWARD_TIMEOUT_PENALTY = -5.0  # 超时惩罚
REWARD_HOLDING_PENALTY_COEFF = 0.00002  # 持有货物惩罚系数
REWARD_SAFETY_VIOLATION = -1.0  # 安全距离违反惩罚
REWARD_SPEED_CHANGE_PENALTY = -0.2  # 频繁反向加减速惩罚

# ========== 神经网络配置 ==========
HIDDEN_DIM = 256  # 隐层维度
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 批大小
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.05  # 最终探索率
EPSILON_DECAY = 0.998  # 探索率衰减
REPLAY_BUFFER_SIZE = 100000  # 经验回放缓冲区大小
MIN_REPLAY_SIZE = 1000  # 开始训练前的最小经验数量

# ========== 学习率调度器配置 ==========
LR_SCHEDULER_ENABLED = True  # 是否启用学习率调度器
LR_WARMUP_EPISODES = 1  # 预热阶段的 episode 数
LR_FINAL_VALUE = 1e-5  # 最终学习率
LR_START_WARMUP_VALUE = 1e-6  # 预热起始学习率

# ========== 训练配置 ==========
NUM_EPISODES = 200  # 训练轮数
MAX_STEPS_PER_EPISODE = int(EPISODE_DURATION / LOW_LEVEL_CONTROL_INTERVAL)  # 每轮最大步数
TRAIN_FREQUENCY = 10  # 每 N 步训练一次神经网络
TARGET_UPDATE_FREQUENCY = 100  # 每 N 步更新目标网络
SAVE_FREQUENCY = 50  # 每 N 个 episode 保存一次模型

# ========== 评估配置 ==========
EVAL_SEED = 42  # 固定测试 seed
EVAL_INTERVAL = 10  # 每 N 个 episode 评估一次

# ========== Step-based 训练配置 ==========
STEPS_PER_UPDATE = 2048  # PPO 每 N 步更新一次
