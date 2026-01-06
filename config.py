"""
配置文件：环形轨道双工位穿梭车调度系统
"""

# ========== 环境配置 ==========
TRACK_LENGTH = 100.0  # 环形轨道长度
MAX_VEHICLES = 2  # 车辆数量
MAX_SPEED = 5.0  # 最大速度 (单位/秒)
MAX_ACCELERATION = 1.0  # 最大加速度
SAFETY_DISTANCE = 2.0  # 安全距离
SPEED_TOLERANCE = 0.1  # 对齐判定的速度容差，车辆速度必须≤此值才能开始上下料

# ========== 上料口配置 ==========
NUM_LOADING_STATIONS = 2  # 上料口数量
LOADING_POSITIONS = [20.0, 60.0]  # 上料口位置
LOADING_STATION_SLOTS = 2  # 每个上料口的工位数（1#, 2#）

# ========== 下料口配置 ==========
NUM_UNLOADING_STATIONS = 3  # 下料口数量
UNLOADING_POSITIONS = [30.0, 50.0, 80.0]  # 下料口位置
UNLOADING_STATION_SLOTS = 2  # 每个下料口的工位数

# ========== 货物配置 ==========
# 根据project.md要求：货物每隔5-15秒（随机整数）出现在上料口
ARRIVAL_INTERVAL_MIN = 5  # 货物最小到达间隔(秒)
ARRIVAL_INTERVAL_MAX = 15  # 货物最大到达间隔(秒)
CARGO_TIMEOUT = 120.0  # 货物超时时间(秒)，默认120秒（Tm）

# ========== 操作配置 ==========
LOADING_TIME = 15.0  # 单工位上料耗时(秒)，根据project.md要求
UNLOADING_TIME = 15.0  # 单工位下料耗时(秒)，根据project.md要求

# ========== 仿真配置 ==========
EPISODE_DURATION = 8 * 3600  # 仿真时长8小时，足够产生有意义的训练数据
HIGH_LEVEL_DECISION_INTERVAL = 1.0  # 高层决策时间间隔(秒)
LOW_LEVEL_CONTROL_INTERVAL = 0.5  # 低层控制时间间隔(秒)，0.5秒保证控制精度

# ========== 奖励参数 ==========
REWARD_DELIVERY = 20.0  # 完成卸货奖励，增大以鼓励完成任务
REWARD_PICKUP = 20.0  # 完成取货奖励，鼓励及时取货
REWARD_ASSIGNMENT = 20.0  # 分配货物给小车的奖励，鼓励及时做出分配决策
REWARD_WAIT_PENALTY_COEFF = 0.00002  # 等待惩罚系数，降低以减少负奖励
REWARD_TIMEOUT_PENALTY = -5.0  # 超时惩罚，降低以避免过大的负奖励
REWARD_HOLDING_PENALTY_COEFF = 0.00002  # 持有货物惩罚系数，鼓励快速卸货
REWARD_SAFETY_VIOLATION = -1.0  # 安全距离违反惩罚
REWARD_SPEED_CHANGE_PENALTY = -0.01  # 速度变化惩罚系数，降低

# ========== 神经网络配置 ==========
HIDDEN_DIM = 256  # 隐层维度，平衡表达能力和计算复杂度
LEARNING_RATE = 3e-4  # 学习率，适当降低以稳定训练
BATCH_SIZE = 64  # 批大小，增大以稳定梯度
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.05  # 最终探索率，保留一定探索
EPSILON_DECAY = 0.998  # 探索率衰减，更慢衰减以充分探索
REPLAY_BUFFER_SIZE = 100000  # 经验回放缓冲区大小
MIN_REPLAY_SIZE = 1000  # 开始训练前的最小经验数量

# ========== 学习率调度器配置 ==========
LR_SCHEDULER_ENABLED = True  # 是否启用学习率调度器
LR_WARMUP_EPISODES = 10  # 预热阶段的episode数（lr从0逐渐升到base_value）
LR_FINAL_VALUE = 1e-5  # 最终学习率（训练结束时的lr）
LR_START_WARMUP_VALUE = 1e-6  # 预热起始学习率

# ========== 训练配置 ==========
NUM_EPISODES = 500  # 训练轮数
MAX_STEPS_PER_EPISODE = int(EPISODE_DURATION / LOW_LEVEL_CONTROL_INTERVAL)  # 每轮最大步数
TRAIN_FREQUENCY = 10  # 每N步训练一次神经网络
TARGET_UPDATE_FREQUENCY = 100  # 每N步更新目标网络
SAVE_FREQUENCY = 50  # 每N个episode保存一次模型

# ========== 评估配置 ==========
EVAL_SEED = 42  # 固定测试seed
EVAL_INTERVAL = 10  # 每N个episode评估一次
