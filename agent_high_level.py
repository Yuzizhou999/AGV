"""
上层智能体（High-Level Agent）：任务分配与货物流向决策
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from config import *


class HighLevelNetwork(nn.Module):
    """高层神经网络"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(HighLevelNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class HighLevelAgent:
    """高层智能体"""
    
    def __init__(self, obs_dim: int, action_dim: int, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 神经网络
        self.q_network = HighLevelNetwork(obs_dim, action_dim).to(device)
        self.target_network = HighLevelNetwork(obs_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # 经验回放 - 使用配置的缓冲区大小
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        
        # 探索率
        self.epsilon = EPSILON_START
        self.steps = 0
    
    def select_action(self, observation: np.ndarray) -> int:
        """选择动作（ε-贪心策略）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储过渡"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, batch_size: int = BATCH_SIZE):
        """训练网络"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q学习更新
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """衰减探索率（每个episode结束后调用）"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


class HighLevelController:
    """高层控制器：决策组件
    
    负责任务分配决策，并维护车辆任务队列：
    - 分配上料任务时，将任务加入车辆的 assigned_tasks 队列
    - 分配下料目标时，将任务加入车辆的 assigned_tasks 队列
    - 低层控制器从队列中读取任务，规划运动路径
    - 任务完成后自动从队列移除，实现闭环控制
    """
    
    def __init__(self, agent: HighLevelAgent, env):
        self.agent = agent
        self.env = env
        self.last_action = None
        self.action_sequence = []
    
    def compute_action(self, observation: Dict) -> Optional[Dict]:
        """
        基于观测计算高层动作
        
        Args:
            observation: 环境观测
        
        Returns:
            高层动作字典或None
        """
        # 获取观测向量
        obs_vector = self.env.get_high_level_observation()
        
        # 智能体选择动作
        action_idx = self.agent.select_action(obs_vector)
        
        # 将动作索引映射到具体动作
        action = self._decode_action(action_idx, observation)
        
        return action
    
    def _decode_action(self, action_idx: int, observation: Dict) -> Optional[Dict]:
        """将动作索引解码为具体动作"""
        # 统计可用动作
        available_actions = []
        
        # 检查是否有待取货物（已按优先级排序，超时货物在前）
        waiting_cargos = observation.get('waiting_cargos', [])
        available_vehicles = {v_id: v for v_id, v in enumerate(observation.get('vehicles', []))}
        
        # 获取已分配的车辆工位（避免重复分配）
        assigned_vehicle_slots = set()
        for cargo in self.env.cargos.values():
            if (cargo.assigned_vehicle is not None and 
                cargo.current_location.startswith("IP_")):
                assigned_vehicle_slots.add((cargo.assigned_vehicle, cargo.assigned_vehicle_slot))
        
        # 生成上料任务动作（优先处理超时货物）
        for cargo_info in waiting_cargos:
            cargo_id = cargo_info['id']
            cargo = self.env.cargos[cargo_id]
            # 跳过已分配的货物
            if cargo.assigned_vehicle is not None:
                continue
            priority = cargo_info.get('priority', 0)
            # 查找车辆
            for vehicle_id, vehicle_obs in available_vehicles.items():
                vehicle = self.env.vehicles[vehicle_id]
                # 检查每个工位
                for slot_idx in range(2):
                    if (vehicle.slots[slot_idx] is None and 
                        (vehicle_id, slot_idx) not in assigned_vehicle_slots):
                        available_actions.append({
                            'type': 'assign_loading',
                            'cargo_id': cargo_id,
                            'vehicle_id': vehicle_id,
                            'slot_idx': slot_idx,
                            'priority': priority  # 保留优先级信息
                        })
                        break  # 每个车辆只选择一个空闲工位
        
        # 生成下料目标动作
        for vehicle_id, vehicle in self.env.vehicles.items():
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is not None:
                    cargo = self.env.cargos[cargo_id]
                    if cargo.target_unloading_station is None:
                        for unload_station_id in cargo.allowed_unloading_stations:
                            # TODO: 选择合适的下料工位（这里简单选择第0个）
                            slot_to_use = 0
                            available_actions.append({
                                'type': 'assign_unloading',
                                'cargo_id': cargo_id,
                                'unloading_station_id': unload_station_id,
                                'slot_idx': slot_to_use
                            })
        
        # 选择动作
        if not available_actions:
            return None
        
        action_idx = min(action_idx, len(available_actions) - 1)
        return available_actions[action_idx]
