"""
使用自定义PPO的底层控制器
不依赖Stable-Baselines3和Gym环境
直接从多智能体环境中学习
"""

import numpy as np
import torch
import os
from typing import Dict, Optional
from config import *
from ppo_agent import PPOAgent


class RLLowLevelObservation:
    """
    底层智能体观测空间构建器
    与原rl_low_level_agent.py中的实现相同
    """

    @staticmethod
    def build(env, vehicle_id: int) -> np.ndarray:
        """构建车辆的观测向量 (15维)"""
        vehicle = env.vehicles[vehicle_id]
        obs = []

        # 1. 自车状态 (3维)
        obs.append(vehicle.position / TRACK_LENGTH)
        obs.append(vehicle.velocity / MAX_SPEED)
        
        if not hasattr(vehicle, '_prev_velocity'):
            vehicle._prev_velocity = vehicle.velocity
        acceleration = (vehicle.velocity - vehicle._prev_velocity) / LOW_LEVEL_CONTROL_INTERVAL
        vehicle._prev_velocity = vehicle.velocity
        obs.append(np.clip(acceleration / MAX_ACCELERATION, -1.0, 1.0))

        # 2. 目标信息 (4维)
        target_position = None
        is_loading_unloading = float(vehicle.is_loading_unloading)

        if vehicle.assigned_tasks:
            target_position = vehicle.assigned_tasks[0]['target_position']
        else:
            for cargo_id in vehicle.slots:
                if cargo_id is not None:
                    cargo = env.cargos[cargo_id]
                    if cargo.target_unloading_station is not None:
                        station = env.unloading_stations[cargo.target_unloading_station]
                        target_position = station.position
                        break

        if target_position is not None:
            distance_to_target = vehicle.distance_to(target_position)
            if vehicle.velocity > 0.1:
                time_to_target = distance_to_target / vehicle.velocity
            else:
                time_to_target = 100.0
        else:
            distance_to_target = 0.0
            time_to_target = 0.0

        obs.append(distance_to_target / TRACK_LENGTH)
        obs.append(0.0)  # target_velocity
        obs.append(min(time_to_target / 100.0, 1.0))
        obs.append(is_loading_unloading)

        # 3. 前车信息 (3维)
        front_vehicle = RLLowLevelObservation._find_front_vehicle(env, vehicle_id)
        if front_vehicle is not None:
            front_distance = vehicle.distance_to(front_vehicle.position)
            front_velocity = front_vehicle.velocity
            relative_velocity = vehicle.velocity - front_vehicle.velocity
        else:
            front_distance = TRACK_LENGTH
            front_velocity = vehicle.velocity
            relative_velocity = 0.0

        obs.append(front_distance / TRACK_LENGTH)
        obs.append(front_velocity / MAX_SPEED)
        obs.append(np.clip(relative_velocity / MAX_SPEED, -1.0, 1.0))

        # 4. 后车信息 (2维)
        rear_vehicle = RLLowLevelObservation._find_rear_vehicle(env, vehicle_id)
        if rear_vehicle is not None:
            rear_distance = rear_vehicle.distance_to(vehicle.position)
            rear_velocity = rear_vehicle.velocity
        else:
            rear_distance = TRACK_LENGTH
            rear_velocity = vehicle.velocity

        obs.append(rear_distance / TRACK_LENGTH)
        obs.append(rear_velocity / MAX_SPEED)

        # 5. 任务信息 (3维)
        has_cargo = float(any(slot is not None for slot in vehicle.slots))
        has_task = float(len(vehicle.assigned_tasks) > 0)

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
    def _find_front_vehicle(env, vehicle_id: int):
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
    def _find_rear_vehicle(env, vehicle_id: int):
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
    底层智能体奖励函数
    与原rl_low_level_agent.py中的实现相同
    """

    @staticmethod
    def compute(env, vehicle_id: int, action: float, prev_state: Dict) -> float:
        """计算单步奖励"""
        vehicle = env.vehicles[vehicle_id]
        reward = 0.0

        # 获取目标位置
        target_position = None
        if vehicle.assigned_tasks:
            target_position = vehicle.assigned_tasks[0]['target_position']
        else:
            for cargo_id in vehicle.slots:
                if cargo_id is not None:
                    cargo = env.cargos[cargo_id]
                    if cargo.target_unloading_station is not None:
                        station = env.unloading_stations[cargo.target_unloading_station]
                        target_position = station.position
                        break

        # 目标相关奖励
        if target_position is not None:
            distance = vehicle.distance_to(target_position)
            prev_distance = prev_state.get('distance_to_target', distance)

            # 接近目标奖励
            distance_reward = (prev_distance - distance) * 0.1
            reward += distance_reward

            # 速度匹配奖励
            if distance < 5.0:
                expected_velocity = min(distance / 5.0 * MAX_SPEED, MAX_SPEED)
                velocity_error = abs(vehicle.velocity - expected_velocity)
                reward += -0.5 * velocity_error

            # 对齐成功奖励
            if vehicle.is_aligned_with(target_position):
                reward += 10.0

        # 安全距离奖励
        front_vehicle = RLLowLevelObservation._find_front_vehicle(env, vehicle_id)
        if front_vehicle is not None:
            front_distance = vehicle.distance_to(front_vehicle.position)
            if front_distance < SAFETY_DISTANCE:
                violation = SAFETY_DISTANCE - front_distance
                reward += -10.0 * violation
            elif front_distance < SAFETY_DISTANCE * 2:
                reward += -0.5 * (SAFETY_DISTANCE * 2 - front_distance)

        # 平滑控制奖励
        reward += -0.01 * abs(action)

        # 上下料静止奖励
        if vehicle.is_loading_unloading:
            if vehicle.velocity < SPEED_TOLERANCE:
                reward += 0.5
            else:
                reward += -2.0

        return reward


class CustomPPOController:
    """
    使用自定义PPO的底层控制器
    直接从环境交互中收集经验并训练
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
        
        # 观测和动作维度
        self.obs_dim = 15  # 观测空间维度
        self.action_dim = 1  # 动作空间维度（加速度）
        
        # 为每辆车创建PPO智能体
        self.agents = {}
        for vehicle_id in range(MAX_VEHICLES):
            self.agents[vehicle_id] = PPOAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=device
            )
            
            # 如果提供了模型路径，加载模型
            if model_path is not None:
                agent_model_path = model_path.replace('.pth', f'_v{vehicle_id}.pth')
                if os.path.exists(agent_model_path):
                    self.agents[vehicle_id].load(agent_model_path)
                    print(f"✓ 车辆{vehicle_id}加载模型: {agent_model_path}")
        
        # 用于存储上一步的状态信息
        self.prev_states = {vid: {} for vid in range(MAX_VEHICLES)}
        
        print(f"✓ 使用自定义PPO底层控制器（不依赖SB3）")

    def compute_actions(self, deterministic=False) -> Dict[int, float]:
        """
        计算所有车辆的控制动作，并存储经验到缓冲区

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            {vehicle_id: acceleration} 加速度字典
        """
        actions = {}
        
        for vehicle_id in self.env.vehicles.keys():
            # 获取观测
            obs = RLLowLevelObservation.build(self.env, vehicle_id)
            
            # 选择动作
            action, value, log_prob = self.agents[vehicle_id].select_action(obs, deterministic)
            
            # 动作是归一化的 [-1, 1]，需要反归一化到实际加速度
            acceleration = float(action[0]) * MAX_ACCELERATION
            actions[vehicle_id] = acceleration
            
            # 计算奖励
            reward = RLLowLevelReward.compute(
                self.env, vehicle_id, acceleration, self.prev_states[vehicle_id]
            )
            
            # 存储经验（在下一步调用时存储，因为需要知道done）
            # 这里先临时保存
            if not hasattr(self, '_temp_transitions'):
                self._temp_transitions = {}
            
            self._temp_transitions[vehicle_id] = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'value': value,
                'log_prob': log_prob
            }
            
            # 更新状态信息
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle.assigned_tasks:
                target_pos = vehicle.assigned_tasks[0]['target_position']
                self.prev_states[vehicle_id]['distance_to_target'] = vehicle.distance_to(target_pos)
            else:
                self.prev_states[vehicle_id]['distance_to_target'] = 0.0
        
        return actions
    
    def store_transitions(self, done=False):
        """
        将临时存储的转移存入各智能体的缓冲区
        
        Args:
            done: episode是否结束
        """
        if not hasattr(self, '_temp_transitions'):
            return
        
        for vehicle_id, transition in self._temp_transitions.items():
            self.agents[vehicle_id].store_transition(
                obs=transition['obs'],
                action=transition['action'],
                reward=transition['reward'],
                value=transition['value'],
                log_prob=transition['log_prob'],
                done=done
            )
        
        self._temp_transitions = {}
    
    def update_policies(self):
        """
        更新所有智能体的策略
        在episode结束后调用
        
        Returns:
            stats: 训练统计信息
        """
        all_stats = {}
        
        for vehicle_id, agent in self.agents.items():
            # 获取最后一个状态的价值（用于GAE计算）
            obs = RLLowLevelObservation.build(self.env, vehicle_id)
            _, next_value, _ = agent.select_action(obs, deterministic=True)
            
            # 更新策略
            stats = agent.update(next_value=next_value)
            all_stats[vehicle_id] = stats
        
        return all_stats

    def save_models(self, save_dir: str, prefix: str = 'custom_ppo'):
        """
        保存所有车辆的模型

        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)

        for vehicle_id, agent in self.agents.items():
            save_path = os.path.join(save_dir, f"{prefix}_v{vehicle_id}.pth")
            agent.save(save_path)

        print(f"✓ 所有车辆模型已保存到: {save_dir}")
