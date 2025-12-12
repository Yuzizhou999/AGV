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
    """高层神经网络 - 联合决策：同时选择车辆和下料口"""
    
    def __init__(self, input_dim: int, max_decisions: int = 10, max_vehicle_choices: int = 3, max_unloading_choices: int = 3):
        """
        Args:
            input_dim: 输入维度（全局状态）
            max_decisions: 最大同时决策数量
            max_vehicle_choices: 每个决策的车辆候选数
            max_unloading_choices: 每个决策的下料口候选数
        """
        super(HighLevelNetwork, self).__init__()
        self.max_decisions = max_decisions
        self.max_vehicle_choices = max_vehicle_choices
        self.max_unloading_choices = max_unloading_choices
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # 车辆选择分支: [batch, max_decisions, max_vehicle_choices + 1]
        # +1 for "不操作" 选项
        self.vehicle_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, max_decisions * (max_vehicle_choices + 1))
        )
        
        # 下料口选择分支: [batch, max_decisions, max_unloading_choices]
        self.unloading_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, max_decisions * max_unloading_choices)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            vehicle_q: [batch, max_decisions, max_vehicle_choices + 1]
            unloading_q: [batch, max_decisions, max_unloading_choices]
        """
        features = self.encoder(x)
        
        vehicle_q = self.vehicle_head(features)
        vehicle_q = vehicle_q.reshape(-1, self.max_decisions, self.max_vehicle_choices + 1)
        
        unloading_q = self.unloading_head(features)
        unloading_q = unloading_q.reshape(-1, self.max_decisions, self.max_unloading_choices)
        
        return vehicle_q, unloading_q


class HighLevelAgent:
    """高层智能体 - 联合决策：同时选择车辆和下料口"""
    
    def __init__(self, obs_dim: int, max_decisions: int = 10, max_vehicle_choices: int = 3, 
                 max_unloading_choices: int = 3, device='cpu'):
        """
        Args:
            obs_dim: 观测维度
            max_decisions: 最大同时决策数
            max_vehicle_choices: 每个决策的车辆候选数
            max_unloading_choices: 每个决策的下料口候选数
            device: 计算设备
        """
        self.device = device
        self.obs_dim = obs_dim
        self.max_decisions = max_decisions
        self.max_vehicle_choices = max_vehicle_choices
        self.max_unloading_choices = max_unloading_choices
        
        # 神经网络
        self.q_network = HighLevelNetwork(obs_dim, max_decisions, max_vehicle_choices, max_unloading_choices).to(device)
        self.target_network = HighLevelNetwork(obs_dim, max_decisions, max_vehicle_choices, max_unloading_choices).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 探索率
        self.epsilon = EPSILON_START
        self.steps = 0
        self.last_vehicle_actions = None  # 记录车辆选择
        self.last_unloading_actions = None  # 记录下料口选择
    
    def select_action(self, observation: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        选择动作（ε-贪心策略）- 联合决策：车辆选择 + 下料口选择
        
        Returns:
            vehicle_actions: List[int], 长度为max_decisions，每个元素为0~max_vehicle_choices
                           0=不操作, 1~max_vehicle_choices=选择对应候选车辆
            unloading_actions: List[int], 长度为max_decisions，每个元素为0~max_unloading_choices-1
                             选择对应候选下料口
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vehicle_q, unloading_q = self.q_network(obs_tensor)
            # vehicle_q: [1, max_decisions, max_vehicle_choices + 1]
            # unloading_q: [1, max_decisions, max_unloading_choices]
        
        vehicle_actions = []
        unloading_actions = []
        
        for i in range(self.max_decisions):
            if np.random.random() < self.epsilon:
                # 探索：随机选择
                vehicle_action = np.random.randint(0, self.max_vehicle_choices + 1)
                unloading_action = np.random.randint(0, self.max_unloading_choices)
            else:
                # 利用：选择最大Q值
                vehicle_action = vehicle_q[0, i, :].argmax().item()
                unloading_action = unloading_q[0, i, :].argmax().item()
            
            vehicle_actions.append(vehicle_action)
            unloading_actions.append(unloading_action)
        
        self.last_vehicle_actions = vehicle_actions
        self.last_unloading_actions = unloading_actions
        return vehicle_actions, unloading_actions
    
    def store_transition(self, state: np.ndarray, vehicle_actions: List[int], unloading_actions: List[int],
                        reward: float, next_state: np.ndarray, done: bool):
        """存储过渡"""
        self.memory.append((state, vehicle_actions, unloading_actions, reward, next_state, done))
    
    def train(self, batch_size: int = BATCH_SIZE):
        """训练网络 - 联合决策优化"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, vehicle_actions, unloading_actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        vehicle_actions = torch.LongTensor(vehicle_actions).to(self.device)  # [batch, max_decisions]
        unloading_actions = torch.LongTensor(unloading_actions).to(self.device)  # [batch, max_decisions]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q学习更新 - 对车辆选择和下料口选择分别计算
        current_vehicle_q, current_unloading_q = self.q_network(states)
        
        # 收集选择的动作的Q值
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.max_decisions)
        decision_indices = torch.arange(self.max_decisions).unsqueeze(0).expand(batch_size, -1)
        
        selected_vehicle_q = current_vehicle_q[batch_indices, decision_indices, vehicle_actions]
        selected_unloading_q = current_unloading_q[batch_indices, decision_indices, unloading_actions]
        
        with torch.no_grad():
            next_vehicle_q, next_unloading_q = self.target_network(next_states)
            max_next_vehicle_q = next_vehicle_q.max(dim=2)[0]
            max_next_unloading_q = next_unloading_q.max(dim=2)[0]
            
            # 联合Q值：两个决策的平均
            max_next_q = (max_next_vehicle_q + max_next_unloading_q) / 2.0
            target_q_values = rewards.unsqueeze(1) + GAMMA * max_next_q * (1 - dones.unsqueeze(1))
        
        # 计算损失（两个head的损失之和）
        loss_vehicle = nn.functional.mse_loss(selected_vehicle_q, target_q_values)
        loss_unloading = nn.functional.mse_loss(selected_unloading_q, target_q_values)
        loss = loss_vehicle + loss_unloading
        
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
        
        # 联合决策：同时选择车辆和下料口
        vehicle_actions, unloading_actions = self.agent.select_action(state_vector)
        
        # 解码动作为具体的任务分配
        task_assignments = self._decode_actions(vehicle_actions, unloading_actions)
        
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
            
            # 为该货物收集下料口候选
            available_unloading = []
            for station_id in cargo.allowed_unloading_stations:
                station = self.env.unloading_stations[station_id]
                available_slot = station.get_available_slot()
                if available_slot is not None:
                    # 计算到各下料口的距离（以上料口为参考）
                    loading_station = self.env.loading_stations[cargo.loading_station]
                    # 简化：使用上料口到下料口的距离
                    distance = abs(station.position - loading_station.position)
                    available_unloading.append({
                        'station_id': station_id,
                        'slot_idx': available_slot,
                        'distance': distance,
                        'load': len([s for s in station.slots if s is not None])  # 当前负载
                    })
            
            if available_vehicles and available_unloading:
                # 按优先级排序
                available_vehicles.sort(key=lambda x: x['priority'], reverse=True)
                # 下料口按距离和负载综合排序
                available_unloading.sort(key=lambda x: (x['load'], x['distance']))
                
                decision = {
                    'type': 'loading',
                    'cargo_id': cargo_id,
                    'vehicle_candidates': available_vehicles[:3],  # 保留前3个车辆候选
                    'unloading_candidates': available_unloading[:3],  # 保留前3个下料口候选
                    'timeout_risk': cargo_info['is_timeout']
                }
                
                # 超时货物强制优先
                if cargo_info['is_timeout']:
                    timeout_decisions.append(decision)
                else:
                    decisions.append(decision)
        
        num_loading = len(timeout_decisions) + len(decisions)
        num_unloading = 0  # 下料决策已集成到装货决策中
        
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
            features.append(1.0)  # 都是装货决策
            features.append(len(decision.get('vehicle_candidates', [])) / 3.0)
            features.append(len(decision.get('unloading_candidates', [])) / 3.0)
            features.append(1.0 if decision.get('timeout_risk', False) else 0.0)
        
        # 填充到固定长度
        target_len = self.agent.obs_dim
        while len(features) < target_len:
            features.append(0.0)
        
        return np.array(features[:target_len], dtype=np.float32)
    
    def _decode_actions(self, vehicle_actions: List[int], unloading_actions: List[int]) -> List[Dict]:
        """
        将RL输出的联合动作解码为具体的任务分配
        
        Args:
            vehicle_actions: 车辆选择动作，0=不操作, 1~n=选择对应候选
            unloading_actions: 下料口选择动作，0~n-1=选择对应候选
        
        Returns:
            List[Dict]: 任务分配列表（包含装货和卸货目标）
        """
        assignments = []
        
        for i, decision in enumerate(self.decision_context['decisions']):
            if i >= len(vehicle_actions):
                break
            
            vehicle_action = vehicle_actions[i]
            unloading_action = unloading_actions[i]
            
            # vehicle_action=0 表示不操作
            if vehicle_action == 0:
                continue
            
            # 获取车辆候选（action - 1 转换为索引）
            vehicle_idx = min(vehicle_action - 1, len(decision['vehicle_candidates']) - 1)
            if vehicle_idx < 0:
                continue
                
            vehicle_candidate = decision['vehicle_candidates'][vehicle_idx]
            
            # 获取下料口候选
            unloading_idx = min(unloading_action, len(decision['unloading_candidates']) - 1)
            unloading_candidate = decision['unloading_candidates'][unloading_idx]
            
            # 生成联合决策：同时包含装货和卸货目标
            assignments.append({
                'type': 'assign_loading_with_target',
                'cargo_id': decision['cargo_id'],
                'vehicle_id': vehicle_candidate['vehicle_id'],
                'vehicle_slot_idx': vehicle_candidate['slot_idx'],
                'unloading_station_id': unloading_candidate['station_id'],
                'unloading_slot_idx': unloading_candidate['slot_idx']
            })
        
        return assignments
