"""
新的强化学习底层智能体 - 基于PPO算法
使用连续动作空间进行精细的速度控制
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
from config import *


class RLLowLevelObservation:
    """
    底层智能体观测空间构建器

    观测维度：15维
    - 自车状态 (3维): position, velocity, acceleration
    - 目标信息 (4维): distance, target_velocity, time_to_target, is_loading
    - 前车信息 (3维): distance, velocity, relative_velocity
    - 后车信息 (2维): distance, velocity
    - 任务信息 (3维): has_cargo, has_task, task_urgency
    """

    @staticmethod
    def build(env, vehicle_id: int) -> np.ndarray:
        """
        构建车辆的观测向量

        Args:
            env: 环境实例
            vehicle_id: 车辆ID

        Returns:
            15维观测向量
        """
        vehicle = env.vehicles[vehicle_id]
        obs = []

        # ========== 1. 自车状态 (3维) ==========
        obs.append(vehicle.position / TRACK_LENGTH)
        obs.append(vehicle.velocity / MAX_SPEED)

        # 计算加速度（通过速度变化估算）
        if not hasattr(vehicle, '_prev_velocity'):
            vehicle._prev_velocity = vehicle.velocity
        acceleration = (vehicle.velocity - vehicle._prev_velocity) / LOW_LEVEL_CONTROL_INTERVAL
        vehicle._prev_velocity = vehicle.velocity
        obs.append(np.clip(acceleration / MAX_ACCELERATION, -1.0, 1.0))

        # ========== 2. 目标信息 (4维) ==========
        target_position = None
        target_velocity = 0.0
        is_loading_unloading = float(vehicle.is_loading_unloading)

        # 获取目标位置（从任务队列或自动寻找卸货点）
        if vehicle.assigned_tasks:
            target_position = vehicle.assigned_tasks[0]['target_position']
            target_velocity = 0.0  # 上下料点目标速度为0
        else:
            # 如果车上有货，自动寻找卸货点
            for cargo_id in vehicle.slots:
                if cargo_id is not None:
                    cargo = env.cargos[cargo_id]
                    if cargo.target_unloading_station is not None:
                        station = env.unloading_stations[cargo.target_unloading_station]
                        target_position = station.position
                        target_velocity = 0.0
                        break

        if target_position is not None:
            distance_to_target = vehicle.distance_to(target_position)
            # 预计到达时间（简单估算）
            if vehicle.velocity > 0.1:
                time_to_target = distance_to_target / vehicle.velocity
            else:
                time_to_target = 100.0  # 静止时设为最大值
        else:
            distance_to_target = 0.0
            time_to_target = 0.0

        obs.append(distance_to_target / TRACK_LENGTH)
        obs.append(target_velocity / MAX_SPEED)
        obs.append(min(time_to_target / 100.0, 1.0))
        obs.append(is_loading_unloading)

        # ========== 3. 前车信息 (3维) ==========
        front_vehicle = RLLowLevelObservation._find_front_vehicle(env, vehicle_id)
        if front_vehicle is not None:
            front_distance = vehicle.distance_to(front_vehicle.position)
            front_velocity = front_vehicle.velocity
            relative_velocity = vehicle.velocity - front_vehicle.velocity
        else:
            front_distance = TRACK_LENGTH  # 无前车时设为轨道长度
            front_velocity = vehicle.velocity  # 假设同速
            relative_velocity = 0.0

        obs.append(front_distance / TRACK_LENGTH)
        obs.append(front_velocity / MAX_SPEED)
        obs.append(np.clip(relative_velocity / MAX_SPEED, -1.0, 1.0))

        # ========== 4. 后车信息 (2维) ==========
        rear_vehicle = RLLowLevelObservation._find_rear_vehicle(env, vehicle_id)
        if rear_vehicle is not None:
            rear_distance = rear_vehicle.distance_to(vehicle.position)
            rear_velocity = rear_vehicle.velocity
        else:
            rear_distance = TRACK_LENGTH
            rear_velocity = vehicle.velocity

        obs.append(rear_distance / TRACK_LENGTH)
        obs.append(rear_velocity / MAX_SPEED)

        # ========== 5. 任务信息 (3维) ==========
        has_cargo = float(any(slot is not None for slot in vehicle.slots))
        has_task = float(len(vehicle.assigned_tasks) > 0)

        # 任务紧急度（基于最老货物的等待时间）
        task_urgency = 0.0
        for cargo_id in vehicle.slots:
            if cargo_id is not None:
                cargo = env.cargos[cargo_id]
                wait_time = env.current_time - cargo.arrival_time
                urgency = min(wait_time / CARGO_TIMEOUT, 1.0)
                task_urgency = max(task_urgency, urgency)

        obs.append(has_cargo)
        obs.append(has_task)
        obs.append(task_urgency)

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _find_front_vehicle(env, vehicle_id: int) -> Optional:
        """查找前方最近的车辆"""
        vehicle = env.vehicles[vehicle_id]
        min_distance = float('inf')
        front_vehicle = None

        for vid, v in env.vehicles.items():
            if vid == vehicle_id:
                continue
            distance = vehicle.distance_to(v.position)
            if 0 < distance < min_distance:
                min_distance = distance
                front_vehicle = v

        return front_vehicle

    @staticmethod
    def _find_rear_vehicle(env, vehicle_id: int) -> Optional:
        """查找后方最近的车辆"""
        vehicle = env.vehicles[vehicle_id]
        min_distance = float('inf')
        rear_vehicle = None

        for vid, v in env.vehicles.items():
            if vid == vehicle_id:
                continue
            distance = v.distance_to(vehicle.position)
            if 0 < distance < min_distance:
                min_distance = distance
                rear_vehicle = v

        return rear_vehicle


class RLLowLevelReward:
    """
    底层智能体奖励函数计算器

    奖励组成：
    1. 接近目标奖励（鼓励靠近目标位置）
    2. 速度匹配奖励（接近目标时速度应降低）
    3. 安全距离奖励（避免碰撞）
    4. 平滑控制奖励（避免急刹车）
    5. 对齐成功奖励（成功到达并对齐）
    """

    @staticmethod
    def compute(env, vehicle_id: int, action: float, prev_state: Dict) -> float:
        """
        计算单步奖励

        Args:
            env: 环境实例
            vehicle_id: 车辆ID
            action: 执行的动作（加速度）
            prev_state: 上一步的状态信息

        Returns:
            奖励值
        """
        vehicle = env.vehicles[vehicle_id]
        reward = 0.0

        # 获取目标位置
        target_position = None
        if vehicle.assigned_tasks:
            target_position = vehicle.assigned_tasks[0]['target_position']
        else:
            # 检查是否需要卸货
            for cargo_id in vehicle.slots:
                if cargo_id is not None:
                    cargo = env.cargos[cargo_id]
                    if cargo.target_unloading_station is not None:
                        station = env.unloading_stations[cargo.target_unloading_station]
                        target_position = station.position
                        break

        # 如果有目标位置，计算目标相关奖励
        if target_position is not None:
            distance = vehicle.distance_to(target_position)
            prev_distance = prev_state.get('distance_to_target', distance)

            # 1. 接近目标奖励（距离减少给正奖励）
            distance_reward = (prev_distance - distance) * 0.1
            reward += distance_reward

            # 2. 速度匹配奖励（接近目标时速度应该降低）
            if distance < 5.0:
                # 期望速度：距离越近，期望速度越低
                expected_velocity = min(distance / 5.0 * MAX_SPEED, MAX_SPEED)
                velocity_error = abs(vehicle.velocity - expected_velocity)
                reward += -0.5 * velocity_error

            # 5. 对齐成功奖励
            if vehicle.is_aligned_with(target_position):
                reward += 10.0

        # 3. 安全距离奖励
        front_vehicle = RLLowLevelObservation._find_front_vehicle(env, vehicle_id)
        if front_vehicle is not None:
            front_distance = vehicle.distance_to(front_vehicle.position)
            if front_distance < SAFETY_DISTANCE:
                # 违反安全距离，给予惩罚
                violation = SAFETY_DISTANCE - front_distance
                reward += -10.0 * violation
            elif front_distance < SAFETY_DISTANCE * 2:
                # 距离较近但未违反，小惩罚鼓励保持距离
                reward += -0.5 * (SAFETY_DISTANCE * 2 - front_distance)

        # 4. 平滑控制奖励（惩罚大的加速度变化）
        reward += -0.01 * abs(action)

        # 额外：如果正在上下料，静止奖励
        if vehicle.is_loading_unloading:
            if vehicle.velocity < SPEED_TOLERANCE:
                reward += 0.5  # 上下料时保持静止给小奖励
            else:
                reward += -2.0  # 上下料时移动给惩罚

        return reward


class LowLevelGymEnv(gym.Env):
    """
    将AGV环境包装为Gymnasium环境，用于PPO训练
    单车环境（每个车辆独立训练）
    """

    def __init__(self, base_env, vehicle_id: int):
        """
        初始化Gym环境

        Args:
            base_env: AGV基础环境
            vehicle_id: 控制的车辆ID
        """
        super().__init__()
        self.base_env = base_env
        self.vehicle_id = vehicle_id

        # 定义观测空间（15维连续）
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # 定义动作空间（1维连续加速度）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 状态追踪
        self.prev_state = {}

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置基础环境（如果需要）
        # 这里假设外部会重置base_env

        # 获取初始观测
        obs = RLLowLevelObservation.build(self.base_env, self.vehicle_id)

        # 初始化状态追踪
        vehicle = self.base_env.vehicles[self.vehicle_id]
        if vehicle.assigned_tasks:
            target_pos = vehicle.assigned_tasks[0]['target_position']
            self.prev_state['distance_to_target'] = vehicle.distance_to(target_pos)
        else:
            self.prev_state['distance_to_target'] = 0.0

        return obs, {}

    def step(self, action: np.ndarray):
        """
        执行一步动作

        注意：这个函数不会真正调用base_env.step()
        而是由外部训练循环统一调用环境step
        这里只负责计算单车的奖励和观测
        """
        # 解析动作（反归一化到实际加速度范围）
        acceleration = float(action[0]) * MAX_ACCELERATION

        # 计算奖励
        reward = RLLowLevelReward.compute(
            self.base_env, self.vehicle_id, acceleration, self.prev_state
        )

        # 获取新观测
        obs = RLLowLevelObservation.build(self.base_env, self.vehicle_id)

        # 更新状态追踪
        vehicle = self.base_env.vehicles[self.vehicle_id]
        if vehicle.assigned_tasks:
            target_pos = vehicle.assigned_tasks[0]['target_position']
            self.prev_state['distance_to_target'] = vehicle.distance_to(target_pos)

        # episode结束条件（由外部环境控制）
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        """渲染（可选）"""
        pass


class RLLowLevelAgent:
    """
    基于PPO的底层智能体

    使用Stable-Baselines3的PPO实现
    每个车辆一个独立的智能体
    """

    def __init__(self, base_env, vehicle_id: int, model_path: Optional[str] = None, device='cpu'):
        """
        初始化智能体

        Args:
            base_env: AGV基础环境
            vehicle_id: 车辆ID
            model_path: 模型文件路径（如果要加载已训练模型）
            device: 'cpu' 或 'cuda'
        """
        self.base_env = base_env
        self.vehicle_id = vehicle_id
        self.device = device

        # 创建Gym环境
        self.gym_env = LowLevelGymEnv(base_env, vehicle_id)

        # 创建或加载PPO模型
        if model_path is not None:
            # 加载已训练模型
            self.model = PPO.load(model_path, env=self.gym_env, device=device)
            print(f"✓ 车辆{vehicle_id}加载模型: {model_path}")
        else:
            # 创建新模型
            self.model = PPO(
                "MlpPolicy",
                self.gym_env,
                learning_rate=3e-4,
                n_steps=2048,  # 每次更新收集的步数
                batch_size=64,
                n_epochs=10,  # 每次更新训练的轮数
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # 熵系数，鼓励探索
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256])  # Actor和Critic网络结构
                ),
                verbose=0,
                device=device
            )
            print(f"✓ 车辆{vehicle_id}创建新PPO模型")

    def select_action(self, deterministic=False) -> float:
        """
        选择动作

        Args:
            deterministic: 是否使用确定性策略（训练时False，测试时True）

        Returns:
            加速度值 [-MAX_ACCELERATION, MAX_ACCELERATION]
        """
        obs = RLLowLevelObservation.build(self.base_env, self.vehicle_id)
        action, _ = self.model.predict(obs, deterministic=deterministic)

        # 反归一化到实际加速度范围
        acceleration = float(action[0]) * MAX_ACCELERATION
        return acceleration

    def save(self, save_path: str):
        """保存模型"""
        self.model.save(save_path)

    def learn(self, total_timesteps: int):
        """
        训练模型

        注意：这个方法用于独立训练单车，实际使用时由外部训练循环管理
        """
        self.model.learn(total_timesteps=total_timesteps)


class RLLowLevelController:
    """
    底层控制器：管理所有车辆的RL智能体
    """

    def __init__(self, env, model_path: Optional[str] = None, device='cpu'):
        """
        初始化控制器

        Args:
            env: AGV环境实例
            model_path: 模型路径（如果提供，将加载已训练模型）
            device: 'cpu' 或 'cuda'
        """
        self.env = env
        self.device = device

        # 为每辆车创建智能体
        self.agents = {}
        for vehicle_id in range(MAX_VEHICLES):
            agent_model_path = None
            if model_path is not None:
                # 多车模型路径格式：model_path_v{id}.zip
                agent_model_path = model_path.replace('.zip', f'_v{vehicle_id}.zip')

            self.agents[vehicle_id] = RLLowLevelAgent(
                env, vehicle_id, agent_model_path, device
            )

    def compute_actions(self, deterministic=False) -> Dict[int, float]:
        """
        计算所有车辆的控制动作

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            {vehicle_id: acceleration} 加速度字典
        """
        actions = {}
        for vehicle_id in self.env.vehicles.keys():
            acceleration = self.agents[vehicle_id].select_action(deterministic)
            actions[vehicle_id] = acceleration

        return actions

    def save_models(self, save_dir: str, prefix: str = 'rl_low_level'):
        """
        保存所有车辆的模型

        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        for vehicle_id, agent in self.agents.items():
            save_path = os.path.join(save_dir, f"{prefix}_v{vehicle_id}.zip")
            agent.save(save_path)

        print(f"✓ 所有车辆模型已保存到: {save_dir}")
