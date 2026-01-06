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
    底层智能体奖励函数（密集奖励版本）
    学习启发式控制器的物理运动规划逻辑，将稀疏奖励转化为密集信号
    """

    @staticmethod
    def _compute_ideal_speed(distance: float, current_velocity: float,
                            target_velocity: float = SPEED_TOLERANCE / 2) -> float:
        """
        计算给定距离下的理想速度（学习启发式的运动规划）

        物理意义：
        - 如果当前速度已经需要很长距离才能停下，说明速度过快，应该减速
        - 如果距离充足，可以加速到最大速度
        - 这模拟了启发式控制器的三段式规划逻辑

        Args:
            distance: 到目标的距离（米）
            current_velocity: 当前速度（米/秒）
            target_velocity: 目标位置期望的最终速度（米/秒）

        Returns:
            ideal_speed: 理想速度（米/秒）
        """
        a = MAX_ACCELERATION
        v_max = MAX_SPEED
        v_target = target_velocity

        # 计算从当前速度减速到目标速度所需的距离
        if current_velocity > v_target:
            decel_distance_needed = (current_velocity ** 2 - v_target ** 2) / (2 * a)
        else:
            decel_distance_needed = 0.0

        # 如果需要的减速距离已经接近或超过剩余距离，说明速度太快
        if decel_distance_needed >= distance * 0.9:  # 留10%安全余量
            # 计算能安全停下的最大速度
            ideal_speed = np.sqrt(max(0, 2 * a * distance + v_target ** 2))
        else:
            # 距离充足，可以保持较高速度
            # 但不能超过物理上能安全停下的速度上限
            v_allowed = np.sqrt(2 * a * distance + v_target ** 2)
            ideal_speed = min(v_max, v_allowed)

        return ideal_speed

    @staticmethod
    def _compute_brake_distance(velocity: float, target_velocity: float = SPEED_TOLERANCE / 2) -> float:
        """
        计算从当前速度减速到目标速度所需的距离

        物理公式：s = (v^2 - v_target^2) / (2*a)

        Args:
            velocity: 当前速度（米/秒）
            target_velocity: 目标速度（米/秒）

        Returns:
            brake_distance: 刹车距离（米）
        """
        if velocity <= target_velocity:
            return 0.0
        return (velocity ** 2 - target_velocity ** 2) / (2 * MAX_ACCELERATION)

    @staticmethod
    def compute(env, vehicle_id: int, action: float, prev_state: Dict) -> float:
        """
        计算单步奖励（密集奖励版本）

        新增密集奖励项：
        1. 全程理想速度匹配奖励（不再局限于距离<5m）
        2. 进度奖励（持续正反馈）
        3. 减速时机奖励（学习何时开始减速）
        4. 速度上限约束奖励（防止超调）
        """
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

        # ============ 目标相关奖励（原有+新增密集奖励） ============
        if target_position is not None:
            distance = vehicle.distance_to(target_position)
            prev_distance = prev_state.get('distance_to_target', distance)
            v_current = vehicle.velocity
            v_target = SPEED_TOLERANCE / 2  # 目标位置期望的最终速度

            # --- 原有奖励1: 接近目标奖励（保持，增加权重） ---
            distance_reward = (prev_distance - distance) * 0.2  # 从0.1提升到0.2
            reward += distance_reward

            # --- 新增密集奖励1: 全程理想速度匹配奖励 ---
            # 物理意义：根据当前距离，车辆应该有一个"理想速度"
            # 远距离时理想速度接近最大速度，近距离时理想速度应该降低
            ideal_speed = RLLowLevelReward._compute_ideal_speed(distance, v_current, v_target)
            speed_error = abs(v_current - ideal_speed)
            ideal_speed_reward = -0.8 * speed_error  # 强信号，鼓励速度匹配
            reward += ideal_speed_reward

            # --- 新增密集奖励2: 进度奖励 ---
            # 物理意义：奖励向目标移动的持续进度，不需要等到对齐才给奖励
            # 将稀疏的"对齐成功+10"分解为连续的进度奖励
            initial_distance = prev_state.get('initial_distance_to_target', distance)
            if distance > 0:
                progress = max(0, 1.0 - distance / max(initial_distance, 1.0))
                prev_progress = prev_state.get('progress', 0.0)
                progress_delta = progress - prev_progress
                progress_reward = progress_delta * 8.0  # 接近对齐时累计可达约8分
                reward += progress_reward
                prev_state['progress'] = progress
                prev_state['initial_distance_to_target'] = initial_distance

            # --- 新增密集奖励3: 减速时机奖励 ---
            # 物理意义：在物理上最优的位置开始减速（学习启发式的阶段判断）
            brake_distance = RLLowLevelReward._compute_brake_distance(v_current, v_target)

            # 如果当前距离接近理想减速点，且速度较高，鼓励减速动作
            # action > 0表示加速，action < 0表示减速（归一化到[-1,1]）
            if distance <= brake_distance + 2.0 and distance >= brake_distance - 1.0:
                # 在减速窗口内
                if v_current > v_target + 0.5:  # 速度还比较高
                    # 动作是归一化的加速度，负值表示减速
                    if action < 0:  # 减速动作
                        brake_timing_reward = 1.0 * abs(action)  # 减速力度越大奖励越高
                    else:  # 加速动作（错误）
                        brake_timing_reward = -0.5 * action  # 惩罚在该减速时加速
                    reward += brake_timing_reward

            # --- 新增密集奖励4: 速度上限约束奖励 ---
            # 物理意义：给定距离，存在一个最大容许速度，超过必然会超调
            v_max_allowed = min(MAX_SPEED, np.sqrt(2 * MAX_ACCELERATION * distance + v_target ** 2))
            if v_current > v_max_allowed:
                overspeed_penalty = -3.0 * (v_current - v_max_allowed) ** 2  # 二次惩罚，超速越多惩罚越重
                reward += overspeed_penalty

            # --- 原有奖励: 对齐成功奖励（保留，作为最终确认奖励） ---
            if vehicle.is_aligned_with(target_position):
                reward += 10.0  # 保持原有的对齐奖励作为最终成功信号

        # ============ 安全距离奖励（保持原有逻辑） ============
        front_vehicle = RLLowLevelObservation._find_front_vehicle(env, vehicle_id)
        if front_vehicle is not None:
            front_distance = vehicle.distance_to(front_vehicle.position)
            if front_distance < SAFETY_DISTANCE:
                violation = SAFETY_DISTANCE - front_distance
                reward += -10.0 * violation
            elif front_distance < SAFETY_DISTANCE * 2:
                reward += -0.5 * (SAFETY_DISTANCE * 2 - front_distance)

        # ============ 平滑控制奖励（保持原有逻辑） ============
        reward += -0.01 * abs(action)

        # ============ 上下料静止奖励（保持原有逻辑） ============
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

    def __init__(self, env, model_path: Optional[str] = None, device='cpu',
                 total_episodes: int = NUM_EPISODES):
        """
        初始化控制器

        Args:
            env: AGV环境实例
            model_path: 模型路径（如果提供，将加载已训练模型）
            device: 'cpu' 或 'cuda'
            total_episodes: 总训练episode数（用于lr scheduler）
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
                device=device,
                total_episodes=total_episodes  # 传递总episode数给scheduler
            )

            # 如果提供了模型路径，加载模型
            if model_path is not None:
                agent_model_path = model_path.replace('.pth', f'_v{vehicle_id}.pth')
                if os.path.exists(agent_model_path):
                    self.agents[vehicle_id].load(agent_model_path)
                    print(f"✓ 车辆{vehicle_id}加载模型: {agent_model_path}")

        # 用于存储上一步的状态信息
        self.prev_states = {vid: {} for vid in range(MAX_VEHICLES)}

        print(f"[OK] 使用自定义PPO底层控制器（不依赖SB3）")
        print(f"  [!] 奖励统一：环境奖励 = 任务奖励 + sum(底层运动奖励)")
        if LR_SCHEDULER_ENABLED:
            print(f"  [!] 学习率调度器已启用: WarmCosine (预热{LR_WARMUP_EPISODES}ep, {LEARNING_RATE:.0e}→{LR_FINAL_VALUE:.0e})")

    def compute_actions(self, deterministic=False) -> Dict[int, float]:
        """
        计算所有车辆的控制动作（仅选择动作，不计算奖励）

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            {vehicle_id: acceleration} 加速度字典
        """
        actions = {}

        # 临时存储，供compute_and_store_rewards()使用
        if not hasattr(self, '_temp_transitions'):
            self._temp_transitions = {}

        for vehicle_id in self.env.vehicles.keys():
            # 获取观测
            obs = RLLowLevelObservation.build(self.env, vehicle_id)

            # 选择动作
            action, value, log_prob = self.agents[vehicle_id].select_action(obs, deterministic)

            # 动作是归一化的 [-1, 1]，需要反归一化到实际加速度
            acceleration = float(action[0]) * MAX_ACCELERATION
            actions[vehicle_id] = acceleration

            # 临时存储obs, action, value, log_prob
            self._temp_transitions[vehicle_id] = {
                'obs': obs,
                'action': action,
                'acceleration': acceleration,
                'value': value,
                'log_prob': log_prob
            }

        return actions
    
    def compute_and_store_rewards(self, done=False, env_task_reward=0.0) -> float:
        """
        计算底层运动奖励，结合环境任务奖励，存储到缓冲区

        职责分离（方案C）：
        - 环境负责：任务奖励（完成货物、等待惩罚等）
        - 底层负责：运动奖励（速度质量、对齐、安全等）
        - 总奖励 = 环境任务奖励 + sum(底层运动奖励)

        Args:
            done: episode是否结束
            env_task_reward: 环境的任务奖励

        Returns:
            total_reward: 总奖励（环境任务奖励 + 所有底层运动奖励之和）
        """
        if not hasattr(self, '_temp_transitions'):
            return env_task_reward

        total_low_level_reward = 0.0

        for vehicle_id, transition in self._temp_transitions.items():
            # 计算底层运动奖励（密集奖励 - RLLowLevelReward）
            low_level_reward = RLLowLevelReward.compute(
                self.env,
                vehicle_id,
                transition['acceleration'],
                self.prev_states[vehicle_id]
            )

            total_low_level_reward += low_level_reward

            # 每辆车的最终奖励 = 底层运动奖励 + 环境任务奖励的平均分配
            # 这样每个车辆都能感知到任务完成的好处
            num_vehicles = len(self.env.vehicles)
            final_reward = low_level_reward + (env_task_reward / max(num_vehicles, 1))

            # 存储到缓冲区
            self.agents[vehicle_id].store_transition(
                obs=transition['obs'],
                action=transition['action'],
                reward=final_reward,
                value=transition['value'],
                log_prob=transition['log_prob'],
                done=done
            )

            # 更新状态信息
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle.assigned_tasks:
                target_pos = vehicle.assigned_tasks[0]['target_position']
                self.prev_states[vehicle_id]['distance_to_target'] = vehicle.distance_to(target_pos)
            else:
                self.prev_states[vehicle_id]['distance_to_target'] = 0.0

        self._temp_transitions = {}

        # 返回总奖励：环境任务奖励 + 所有底层运动奖励
        return env_task_reward + total_low_level_reward
    
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
