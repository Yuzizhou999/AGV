"""
下层智能体（Low-Level Agent）：速度和加速度控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from config import *


class LowLevelNetwork(nn.Module):
    """下层神经网络"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(LowLevelNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class LowLevelAgent:
    """下层智能体"""
    
    def __init__(self, obs_dim: int, action_dim: int, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim  # 3: {减速, 保持, 加速}
        
        # 神经网络
        self.q_network = LowLevelNetwork(obs_dim, action_dim).to(device)
        self.target_network = LowLevelNetwork(obs_dim, action_dim).to(device)
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


class LowLevelController:
    """下层控制器：执行具体控制"""
    
    def __init__(self, agent: LowLevelAgent, env):
        self.agent = agent
        self.env = env
        self.agents_dict = {}  # {vehicle_id: agent}
    
    def compute_actions(self) -> Dict[int, int]:
        """
        为所有车辆计算低层控制动作
        
        Returns:
            {vehicle_id: action}
        """
        actions = {}
        
        for vehicle_id in self.env.vehicles.keys():
            # 获取车辆观测
            obs = self.env.get_low_level_observation(vehicle_id)
            
            # 智能体选择动作
            action = self.agent.select_action(obs)
            actions[vehicle_id] = action
        
        return actions
    
    def _action_to_control(self, action: int) -> Tuple[float, float]:
        """
        将动作映射到速度/加速度
        
        Args:
            action: 0=减速, 1=保持, 2=加速
        
        Returns:
            (target_velocity, acceleration)
        """
        if action == 0:  # 减速
            return 0.0, -MAX_ACCELERATION
        elif action == 1:  # 保持
            return 0.0, 0.0
        else:  # 加速
            return MAX_SPEED, MAX_ACCELERATION
