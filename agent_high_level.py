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
    """高层神经网络 - 支持多工位联合决策"""
    
    def __init__(self, input_dim: int, max_decisions: int = 10):
        """
        Args:
            input_dim: 输入维度（全局状态）
            max_decisions: 最大同时决策数量（上料+下料工位数）
        """
        super(HighLevelNetwork, self).__init__()
        self.max_decisions = max_decisions
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # 为每个可能的决策输出分支（上料/下料/无操作）
        # 输出形状: [batch, max_decisions, 3] 
        # 每个决策有3种选择：0=不操作, 1=分配最优车辆, 2=分配次优车辆
        self.decision_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, max_decisions * 3)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            [batch, max_decisions, 3] - Q值矩阵
        """
        features = self.encoder(x)
        decisions = self.decision_head(features)
        return decisions.reshape(-1, self.max_decisions, 3)


class HighLevelAgent:
    """高层智能体 - 多工位联合决策"""
    
    def __init__(self, obs_dim: int, max_decisions: int = 10, device='cpu'):
        """
        Args:
            obs_dim: 观测维度
            max_decisions: 最大同时决策数（上料工位数+下料工位数）
            device: 计算设备
        """
        self.device = device
        self.obs_dim = obs_dim
        self.max_decisions = max_decisions
        
        # 神经网络
        self.q_network = HighLevelNetwork(obs_dim, max_decisions).to(device)
        self.target_network = HighLevelNetwork(obs_dim, max_decisions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 探索率
        self.epsilon = EPSILON_START
        self.steps = 0
        self.last_actions = None  # 记录最后选择的动作
    
    def select_action(self, observation: np.ndarray) -> List[int]:
        """
        选择动作（ε-贪心策略）- 返回多个工位的决策
        
        Returns:
            List[int]: 长度为max_decisions的动作列表，每个元素为0/1/2
                      0=不操作, 1=分配最优方案, 2=分配次优方案
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)  # [1, max_decisions, 3]
        
        actions = []
        for i in range(self.max_decisions):
            if np.random.random() < self.epsilon:
                # 探索：随机选择
                action = np.random.randint(0, 3)
            else:
                # 利用：选择最大Q值
                action = q_values[0, i, :].argmax().item()
            actions.append(action)
        
        self.last_actions = actions  # 保存最后的动作
        return actions
    
    def store_transition(self, state: np.ndarray, actions: List[int], reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储过渡"""
        self.memory.append((state, actions, reward, next_state, done))
    
    def train(self, batch_size: int = BATCH_SIZE):
        """训练网络 - 多工位联合优化"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # [batch, max_decisions]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q学习更新 - 对每个决策点独立计算
        current_q_values = self.q_network(states)  # [batch, max_decisions, 3]
        
        # 收集选择的动作的Q值
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.max_decisions)
        decision_indices = torch.arange(self.max_decisions).unsqueeze(0).expand(batch_size, -1)
        selected_q_values = current_q_values[batch_indices, decision_indices, actions]  # [batch, max_decisions]
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # [batch, max_decisions, 3]
            max_next_q_values = next_q_values.max(dim=2)[0]  # [batch, max_decisions]
            target_q_values = rewards.unsqueeze(1) + GAMMA * max_next_q_values * (1 - dones.unsqueeze(1))
        
        # 计算损失（对所有决策点求平均）
        loss = nn.functional.mse_loss(selected_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps += 1
        
        # 定期更新目标网络
        if self.steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

class HighLevelController:
    """高层控制器：基于RL的多工位联合决策"""
    
    def __init__(self, agent: HighLevelAgent, env):
        self.agent = agent
        self.env = env
        self.last_actions = None
        self.decision_context = {}  # 记录决策上下文
    
    def compute_action(self, observation: Dict) -> List[Dict]:
        """
        基于观测计算高层动作 - 同时为所有工位决策
        
        Args:
            observation: 环境观测
        
        Returns:
            List[Dict]: 多个动作的列表（可能为空）
        """
        # 构建决策上下文：收集所有待决策的工位
        self.decision_context = self._build_decision_context(observation)
        
        if not self.decision_context['decisions']:
            return []  # 没有需要决策的工位
        
        # 提取状态向量
        state_vector = self._extract_state_vector(observation)
        
        actions = self.agent.select_action(state_vector)
        
        # 解码动作为具体的任务分配
        task_assignments = self._decode_actions(actions)
        
        return task_assignments
    
    def _build_decision_context(self, observation: Dict) -> Dict:
        """
        构建决策上下文：收集所有待决策的工位
        强制优先处理超时货物
        
        Returns:
            Dict: {
                'decisions': List[Dict],  # 每个决策点的信息
                'num_loading': int,       # 上料决策点数量
                'num_unloading': int      # 下料决策点数量
            }
        """
        decisions = []
        timeout_decisions = []  # 超时货物的决策（强制优先）
        
        # 1. 收集上料决策点（等待中的货物）
        waiting_cargos = observation.get('waiting_cargos', [])
        for cargo_info in waiting_cargos:
            cargo_id = cargo_info['id']
            cargo = self.env.cargos.get(cargo_id)
            if cargo is None:
                continue
            
            # 找到可用的车辆工位
            available_vehicles = []
            for vehicle_id, vehicle in self.env.vehicles.items():
                slot_idx = vehicle.get_empty_slot_idx()
                if slot_idx is not None and vehicle.slot_operation_end_time[slot_idx] <= 0:
                    station = self.env.loading_stations[cargo.loading_station]
                    distance = vehicle.distance_to(station.position)
                    aligned = vehicle.is_aligned_with(station.position)
                    available_vehicles.append({
                        'vehicle_id': vehicle_id,
                        'slot_idx': slot_idx,
                        'distance': distance,
                        'aligned': aligned,
                        'priority': -distance if aligned else -distance - 100
                    })
            
            if available_vehicles:
                # 按优先级排序
                available_vehicles.sort(key=lambda x: x['priority'], reverse=True)
                decision = {
                    'type': 'loading',
                    'cargo_id': cargo_id,
                    'candidates': available_vehicles[:3],  # 只保留前3个候选
                    'timeout_risk': cargo_info['is_timeout']
                }
                
                # 超时货物强制优先
                if cargo_info['is_timeout']:
                    timeout_decisions.append(decision)
                else:
                    decisions.append(decision)
        
        num_loading = len(timeout_decisions) + len(decisions)
        
        # 2. 收集下料决策点（车上需要分配目标的货物）
        for vehicle_id, vehicle in self.env.vehicles.items():
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is None:
                    continue
                cargo = self.env.cargos[cargo_id]
                if cargo.target_unloading_station is not None:
                    continue  # 已分配目标
                
                # 找到可用的下料工位
                available_stations = []
                for station_id in cargo.allowed_unloading_stations:
                    station = self.env.unloading_stations[station_id]
                    available_slot = station.get_available_slot()
                    if available_slot is not None:
                        distance = vehicle.distance_to(station.position)
                        aligned = vehicle.is_aligned_with(station.position)
                        available_stations.append({
                            'station_id': station_id,
                            'slot_idx': available_slot,
                            'distance': distance,
                            'aligned': aligned,
                            'priority': -distance if aligned else -distance - 100
                        })
                
                if available_stations:
                    available_stations.sort(key=lambda x: x['priority'], reverse=True)
                    decisions.append({
                        'type': 'unloading',
                        'cargo_id': cargo_id,
                        'vehicle_id': vehicle_id,
                        'candidates': available_stations[:3]
                    })
        
        num_unloading = len(decisions) - num_loading
        
        # 将超时决策放在最前面，确保优先处理
        all_decisions = timeout_decisions + decisions
        
        return {
            'decisions': all_decisions,
            'num_loading': num_loading,
            'num_unloading': num_unloading,
            'num_timeout': len(timeout_decisions)  # 记录超时决策数量
        }
    
    def _extract_state_vector(self, observation: Dict) -> np.ndarray:
        """
        从观测中提取状态向量
        
        Returns:
            np.ndarray: 状态向量
        """
        features = []
        
        # 全局特征
        features.append(len(observation.get('waiting_cargos', [])) / 10.0)  # 归一化
        features.append(observation.get('completed_cargos', 0) / 100.0)
        features.append(observation.get('timed_out_cargos', 0) / 10.0)
        
        # 车辆特征
        for vehicle_obs in observation.get('vehicles', []):
            features.append(vehicle_obs.get('position', 0.0))
            features.append(vehicle_obs.get('velocity', 0.0))
            slots = vehicle_obs.get('slots', [None, None])
            features.append(1.0 if slots[0] is None else 0.0)
            features.append(1.0 if slots[1] is None else 0.0)
        
        # 上料口占用情况
        for station_obs in observation.get('loading_stations', []):
            slots = station_obs.get('slots', [None, None])
            features.append(1.0 if slots[0] is None else 0.0)
            features.append(1.0 if slots[1] is None else 0.0)
        
        # 下料口占用情况
        for station_obs in observation.get('unloading_stations', []):
            slots = station_obs.get('slots', [None, None])
            features.append(1.0 if slots[0] is None else 0.0)
            features.append(1.0 if slots[1] is None else 0.0)
        
        # 决策点特征（填充到max_decisions）
        for i, decision in enumerate(self.decision_context.get('decisions', [])):
            if i >= self.agent.max_decisions:
                break
            features.append(1.0 if decision['type'] == 'loading' else 0.0)
            features.append(len(decision['candidates']) / 3.0)
            if decision['type'] == 'loading':
                features.append(1.0 if decision.get('timeout_risk', False) else 0.0)
            else:
                features.append(0.0)
        
        # 填充到固定长度
        target_len = self.agent.obs_dim
        while len(features) < target_len:
            features.append(0.0)
        
        return np.array(features[:target_len], dtype=np.float32)
    
    def _decode_actions(self, actions: List[int]) -> List[Dict]:
        """
        将RL输出的动作解码为具体的任务分配
        
        Args:
            actions: 长度为max_decisions的动作列表，每个元素为0/1/2
        
        Returns:
            List[Dict]: 任务分配列表
        """
        assignments = []
        
        for i, (action, decision) in enumerate(zip(actions, self.decision_context['decisions'])):
            if i >= len(self.decision_context['decisions']):
                break
            
            if action == 0:
                # 不操作
                continue
            
            # action=1: 选择最优候选, action=2: 选择次优候选
            candidate_idx = min(action - 1, len(decision['candidates']) - 1)
            candidate = decision['candidates'][candidate_idx]
            
            if decision['type'] == 'loading':
                assignments.append({
                    'type': 'assign_loading',
                    'cargo_id': decision['cargo_id'],
                    'vehicle_id': candidate['vehicle_id'],
                    'slot_idx': candidate['slot_idx']
                })
            else:  # unloading
                assignments.append({
                    'type': 'assign_unloading',
                    'cargo_id': decision['cargo_id'],
                    'unloading_station_id': candidate['station_id'],
                    'slot_idx': candidate['slot_idx']
                })
        
        return assignments
