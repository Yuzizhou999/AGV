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
    """
    高层神经网络 - 基于固定长度状态向量的决策网络
    不再使用动态列表，使用基于全图的固定特征
    """
    
    def __init__(self, input_dim: int, num_loading_stations: int, num_vehicles: int, num_unloading_stations: int):
        """
        Args:
            input_dim: 输入维度（固定长度的状态向量）
            num_loading_stations: 上料口数量
            num_vehicles: 车辆数量
            num_unloading_stations: 下料口数量
        """
        super(HighLevelNetwork, self).__init__()
        self.num_loading_stations = num_loading_stations
        self.num_vehicles = num_vehicles
        self.num_unloading_stations = num_unloading_stations
        
        # 总工位数（每个上料口2个工位）
        self.num_loading_slots = num_loading_stations * 2
        
        # 共享编码器 - 处理全局状态
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # 为每个上料工位输出：选择哪个车辆
        # 输出维度：[num_loading_slots, num_vehicles + 1] (+1 for "不操作")
        self.vehicle_selection_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.num_loading_slots * (num_vehicles + 1))
        )
        
        # 为每个上料工位输出：选择哪个下料口
        # 输出维度：[num_loading_slots, num_unloading_stations * 2] (每个下料口2个工位)
        self.unloading_selection_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.num_loading_slots * num_unloading_stations * 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - 固定长度的状态向量
        Returns:
            vehicle_q: [batch, num_loading_slots, num_vehicles + 1] - 每个上料工位选择哪个车辆
            unloading_q: [batch, num_loading_slots, num_unloading_stations * 2] - 每个上料工位选择哪个下料工位
        """
        features = self.encoder(x)
        
        vehicle_q = self.vehicle_selection_head(features)
        vehicle_q = vehicle_q.reshape(-1, self.num_loading_slots, self.num_vehicles + 1)
        
        unloading_q = self.unloading_selection_head(features)
        unloading_q = unloading_q.reshape(-1, self.num_loading_slots, self.num_unloading_stations * 2)
        
        return vehicle_q, unloading_q


class HighLevelAgent:
    """
    高层智能体 - 基于固定状态空间的决策
    使用固定长度的状态向量，避免动态列表的问题
    """
    
    def __init__(self, obs_dim: int, num_loading_stations: int, num_vehicles: int, 
                 num_unloading_stations: int, device='cpu'):
        """
        Args:
            obs_dim: 观测维度（固定长度）
            num_loading_stations: 上料口数量
            num_vehicles: 车辆数量
            num_unloading_stations: 下料口数量
            device: 计算设备
        """
        self.device = device
        self.obs_dim = obs_dim
        self.num_loading_stations = num_loading_stations
        self.num_vehicles = num_vehicles
        self.num_unloading_stations = num_unloading_stations
        self.num_loading_slots = num_loading_stations * 2  # 每个上料口2个工位
        
        # 神经网络
        self.q_network = HighLevelNetwork(obs_dim, num_loading_stations, num_vehicles, num_unloading_stations).to(device)
        self.target_network = HighLevelNetwork(obs_dim, num_loading_stations, num_vehicles, num_unloading_stations).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 探索率
        self.epsilon = EPSILON_START
        self.steps = 0
        self.last_vehicle_actions = None  # [num_loading_slots] 每个上料工位选择的车辆
        self.last_unloading_actions = None  # [num_loading_slots] 每个上料工位选择的下料工位
        # 用于训练时的 action masking（存当前决策时刻的mask）
        self.last_vehicle_mask: Optional[np.ndarray] = None   # [num_slots, num_vehicles+1] bool
        self.last_unloading_mask: Optional[np.ndarray] = None # [num_slots, num_unloading_slots] bool

    def select_action(
        self,
        observation: np.ndarray,
        vehicle_mask: Optional[np.ndarray] = None,
        unloading_mask: Optional[np.ndarray] = None,
        slot_order: Optional[List[int]] = None
    ) -> Tuple[List[int], List[int]]:
        """
        选择动作（ε-贪心策略）- 为每个上料工位选择车辆和下料工位
        
        Args:
            observation: [obs_dim] 固定长度的状态向量
        
        Returns:
            vehicle_actions: List[int], 长度为num_loading_slots
                           0=不操作, 1~num_vehicles=选择对应车辆ID
            unloading_actions: List[int], 长度为num_loading_slots
                             0~(num_unloading_stations*2-1)=选择下料工位的全局索引
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vehicle_q, unloading_q = self.q_network(obs_tensor)
            # vehicle_q: [1, num_loading_slots, num_vehicles + 1]
            # unloading_q: [1, num_loading_slots, num_unloading_stations * 2]
        
        # --------- 兼容：未提供mask则保持旧逻辑（不建议训练时继续用） ---------
        if vehicle_mask is None or unloading_mask is None:
            vehicle_actions = []
            unloading_actions = []
            for i in range(self.num_loading_slots):
                if np.random.random() < self.epsilon:
                    vehicle_action = np.random.randint(0, self.num_vehicles + 1)
                    unloading_action = np.random.randint(0, self.num_unloading_stations * 2)
                else:
                    vehicle_action = vehicle_q[0, i, :].argmax().item()
                    unloading_action = unloading_q[0, i, :].argmax().item()
                vehicle_actions.append(int(vehicle_action))
                unloading_actions.append(int(unloading_action))
            self.last_vehicle_mask = None
            self.last_unloading_mask = None
        else:
            # --------- Masked ε-greedy + 同周期冲突消解 ---------
            if slot_order is None:
                slot_order = list(range(self.num_loading_slots))

            vehicle_actions = [0] * self.num_loading_slots
            unloading_actions = [0] * self.num_loading_slots

            used_vehicle_ids = set()     # vehicle_id（0-based）
            used_unloading_idx = set()   # unloading_action 全局索引（0..num_unloading_stations*2-1）

            for i in slot_order:
                # 1) 车选择：剔除本周期已用车
                v_mask = vehicle_mask[i].copy()
                for vid in used_vehicle_ids:
                    a = vid + 1  # action=vehicle_id+1
                    if 0 <= a < v_mask.shape[0]:
                        v_mask[a] = False
                # 至少允许“不操作”
                v_mask[0] = True

                if np.random.random() < self.epsilon:
                    valid_vs = np.flatnonzero(v_mask)
                    v_act = int(np.random.choice(valid_vs))
                else:
                    qv = vehicle_q[0, i, :].detach().cpu().numpy()
                    qv[~v_mask] = -1e9
                    v_act = int(qv.argmax())

                if v_act == 0:
                    vehicle_actions[i] = 0
                    unloading_actions[i] = 0
                    continue

                # 2) OP选择：剔除本周期已用OP工位
                u_mask = unloading_mask[i].copy()
                for ui in used_unloading_idx:
                    if 0 <= ui < u_mask.shape[0]:
                        u_mask[ui] = False

                if not u_mask.any():
                    # 没有可用卸货位：本slot直接不派单，避免“派车无处卸”
                    vehicle_actions[i] = 0
                    unloading_actions[i] = 0
                    continue

                if np.random.random() < self.epsilon:
                    valid_us = np.flatnonzero(u_mask)
                    u_act = int(np.random.choice(valid_us))
                else:
                    qu = unloading_q[0, i, :].detach().cpu().numpy()
                    qu[~u_mask] = -1e9
                    u_act = int(qu.argmax())

                vehicle_actions[i] = v_act
                unloading_actions[i] = u_act

                used_vehicle_ids.add(v_act - 1)
                used_unloading_idx.add(u_act)

            self.last_vehicle_mask = vehicle_mask
            self.last_unloading_mask = unloading_mask
        
        self.last_vehicle_actions = vehicle_actions
        self.last_unloading_actions = unloading_actions
        return vehicle_actions, unloading_actions
    
    def store_transition(self, state: np.ndarray, vehicle_actions: List[int], unloading_actions: List[int],
                        reward: float, next_state: np.ndarray, done: bool,
                        next_vehicle_mask: Optional[np.ndarray] = None,
                        next_unloading_mask: Optional[np.ndarray] = None):
        """存储过渡"""
        self.memory.append((state, vehicle_actions, unloading_actions, reward, next_state, done,
+                            next_vehicle_mask, next_unloading_mask))
    
    def train(self, batch_size: int = BATCH_SIZE):
        """训练网络 - 基于固定状态空间的优化"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        if len(batch[0]) == 6:
            states, vehicle_actions, unloading_actions, rewards, next_states, dones = zip(*batch)
            next_vehicle_masks = None
            next_unloading_masks = None
        else:
            states, vehicle_actions, unloading_actions, rewards, next_states, dones, next_vehicle_masks, next_unloading_masks = zip(*batch)
   
        states = torch.FloatTensor(np.array(states)).to(self.device)
        vehicle_actions = torch.LongTensor(vehicle_actions).to(self.device)  # [batch, num_loading_slots]
        unloading_actions = torch.LongTensor(unloading_actions).to(self.device)  # [batch, num_loading_slots]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q学习更新
        current_vehicle_q, current_unloading_q = self.q_network(states)
        
        # 收集选择的动作的Q值
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.num_loading_slots)
        slot_indices = torch.arange(self.num_loading_slots).unsqueeze(0).expand(batch_size, -1)
        
        selected_vehicle_q = current_vehicle_q[batch_indices, slot_indices, vehicle_actions]
        selected_unloading_q = current_unloading_q[batch_indices, slot_indices, unloading_actions]
        
        with torch.no_grad():
            next_vehicle_q, next_unloading_q = self.target_network(next_states)
            if next_vehicle_masks is not None and next_unloading_masks is not None:
                nvm = torch.BoolTensor(np.array(next_vehicle_masks)).to(self.device)
                num = torch.BoolTensor(np.array(next_unloading_masks)).to(self.device)
                next_vehicle_q = next_vehicle_q.masked_fill(~nvm, -1e9)
                next_unloading_q = next_unloading_q.masked_fill(~num, -1e9)
                
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
    """
    高层控制器：基于固定状态空间的决策
    为每个上料工位直接决策车辆和下料工位
    """
    
    def __init__(self, agent: HighLevelAgent, env):
        self.agent = agent
        self.env = env
        self.last_actions = None
    
    def compute_action(self, observation: Dict) -> List[Dict]:
        """
        基于观测计算高层动作 - 使用固定状态空间
        
        Args:
            observation: 环境观测
        
        Returns:
            List[Dict]: 多个动作的列表（可能为空）
        """
        # 提取固定长度的状态向量
        state_vector = self._extract_state_vector(observation)
        
        # 为每个上料工位选择车辆和下料工位
        vehicle_mask, unloading_mask, slot_order = self._build_action_masks_and_order(observation)
        vehicle_actions, unloading_actions = self.agent.select_action(
            state_vector, vehicle_mask, unloading_mask, slot_order
        )
        
        # 解码动作为具体的任务分配
        task_assignments = self._decode_actions(observation, vehicle_actions, unloading_actions)
        
        return task_assignments
    
    def _build_action_masks_and_order(self, observation: Dict) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        为每个上料工位构建：
        - vehicle_mask: [num_slots, num_vehicles+1] (action=0 代表不操作)
        - unloading_mask: [num_slots, num_unloading_stations*2] (每个OP 2个工位)
        - slot_order: 超时优先，其次等待时间长优先
        """
        num_slots = self.agent.num_loading_slots
        v_mask = np.zeros((num_slots, self.agent.num_vehicles + 1), dtype=bool)
        u_mask = np.zeros((num_slots, self.agent.num_unloading_stations * 2), dtype=bool)

        # 为避免“全False”导致采样/argmax异常：给卸货头一个兜底动作（仅在 vehicle_action==0 时会被用到）
        u_mask[:, 0] = True

        waiting = observation.get('waiting_cargos', [])
        cargo_info = {c['id']: c for c in waiting}

        slot_meta = []  # (slot_idx, has_cargo, is_timeout, wait_time)
        slot_idx = 0
        for _, station in self.env.loading_stations.items():
            for cargo_slot_idx in range(2):
                cargo_id = station.slots[cargo_slot_idx]

                # 不操作永远合法
                v_mask[slot_idx, 0] = True

                if cargo_id is None:
                    slot_meta.append((slot_idx, 0, 0, 0.0))
                    slot_idx += 1
                    continue

                # 车辆合法性：有空位 + 空位不在装卸 + 且未已有 assigned_task（避免覆盖）:contentReference[oaicite:6]{index=6}
                for vid, veh in self.env.vehicles.items():
                    if veh.assigned_task is not None:
                        continue
                    empty_si = veh.get_empty_slot_idx()
                    if empty_si is None:
                        continue
                    if veh.slot_operation_end_time[empty_si] > 0:
                        continue
                    v_mask[slot_idx, vid + 1] = True

                # OP合法性：必须在 cargo.allowed_unloading_stations，且对应工位未reserved，且未占用
                cargo = self.env.cargos.get(cargo_id)
                if cargo is not None:
                    for op_id in cargo.allowed_unloading_stations:
                        op = self.env.unloading_stations.get(op_id)
                        if op is None:
                            continue
                        for us in range(2):
                            if op.slot_reserved[us]:
                                continue
                            if op.slots[us] is not None:
                                continue
                            u_mask[slot_idx, op_id * 2 + us] = True

                info = cargo_info.get(cargo_id, {})
                wt = float(info.get('wait_time', 0.0))
                to = 1 if bool(info.get('is_timeout', False)) else 0
                slot_meta.append((slot_idx, 1, to, wt))
                slot_idx += 1

        # 排序：has_cargo(1优先) -> is_timeout(1优先) -> wait_time(大优先)
        slot_meta.sort(key=lambda x: (-x[1], -x[2], -x[3]))
        slot_order = [x[0] for x in slot_meta]
        return v_mask, u_mask, slot_order
    
    def _extract_state_vector(self, observation: Dict) -> np.ndarray:
        """
        从观测中提取固定长度的状态向量
        
        状态向量设计：
        1. 全局信息 (3个特征)
        2. 上料口状态 (每个上料口2个工位 * 3个特征 = num_loading_stations * 6)
        3. 车辆状态 (每辆车 * 5个特征 = num_vehicles * 5)
        4. 下料口状态 (每个下料口2个工位 * 2个特征 = num_unloading_stations * 4)
        
        Returns:
            np.ndarray: 固定长度的状态向量
        """
        features = []
        
        # ========== 1. 全局特征 (3个) ==========
        global_info = observation.get('global_info', {})
        features.append(global_info.get('current_time', 0.0))  # 归一化时间 [0, 1]
        features.append(len(observation.get('waiting_cargos', [])) / 10.0)  # 等待货物数
        features.append(global_info.get('timed_out_cargos', 0) / 10.0)  # 超时货物数
        
        # ========== 2. 上料口状态 (每个工位3个特征：有货否、等待时间、是否超时) ==========
        waiting_cargos = observation.get('waiting_cargos', [])
        cargo_info_dict = {c['id']: c for c in waiting_cargos}
        
        for station_obs in observation.get('loading_stations', []):
            slots = station_obs.get('slots', [None, None])
            for slot_cargo_id in slots:
                if slot_cargo_id is None:
                    features.extend([0.0, 0.0, 0.0])  # 无货物
                else:
                    cargo_info = cargo_info_dict.get(slot_cargo_id, {})
                    features.append(1.0)  # 有货物
                    features.append(min(cargo_info.get('wait_time', 0.0) / 120.0, 1.0))  # 等待时间归一化
                    features.append(1.0 if cargo_info.get('is_timeout', False) else 0.0)  # 是否超时
        
        # ========== 3. 车辆状态 (每辆车5个特征) ==========
        for vehicle_obs in observation.get('vehicles', []):
            features.append(vehicle_obs.get('position', 0.0))  # 位置 [0, 1]
            features.append(vehicle_obs.get('velocity', 0.0))  # 速度 [-1, 1]
            slot_occupied = vehicle_obs.get('slot_occupied', [False, False])
            features.append(1.0 if not slot_occupied[0] else 0.0)  # 工位1是否空闲
            features.append(1.0 if not slot_occupied[1] else 0.0)  # 工位2是否空闲
            # 距离最近上料口的距离
            min_dist = 1.0
            vehicle_pos = vehicle_obs.get('position', 0.0) * TRACK_LENGTH
            for station_obs in observation.get('loading_stations', []):
                station_pos = station_obs.get('position', 0.0) * TRACK_LENGTH
                d = (station_pos - vehicle_pos) % TRACK_LENGTH
                dist = min(d, TRACK_LENGTH - d) / TRACK_LENGTH
                min_dist = min(min_dist, dist)
            features.append(min_dist)
        
        # ========== 4. 下料口状态 (每个工位2个特征：占用否、预约否) ==========
        for station_id, station in self.env.unloading_stations.items():
            for slot_idx in range(2):
                features.append(1.0 if station.slots[slot_idx] is None else 0.0)  # 是否空闲
                features.append(1.0 if station.slot_reserved[slot_idx] else 0.0)  # 是否被预约
        
        # 填充到固定长度
        target_len = self.agent.obs_dim
        while len(features) < target_len:
            features.append(0.0)
        
        # 截断到固定长度
        return np.array(features[:target_len], dtype=np.float32)
    
    def _decode_actions(self, observation: Dict, vehicle_actions: List[int], 
                       unloading_actions: List[int]) -> List[Dict]:
        """
        将RL输出的动作解码为具体的任务分配
        
        Args:
            observation: 环境观测
            vehicle_actions: [num_loading_slots] 每个上料工位选择的车辆，0=不操作, 1~num_vehicles=车辆ID
            unloading_actions: [num_loading_slots] 每个上料工位选择的下料工位全局索引
        
        Returns:
            List[Dict]: 任务分配列表
        """
        assignments = []
        used_vehicle_ids = set()
        used_unloading_slots = set()  # (op_id, slot)
        
        # 遍历每个上料工位
        slot_idx = 0
        for station_id, station in self.env.loading_stations.items():
            for cargo_slot_idx in range(2):  # 每个上料口2个工位
                # 检查该工位是否有货物
                cargo_id = station.slots[cargo_slot_idx]
                if cargo_id is None:
                    slot_idx += 1
                    continue
                
                # 获取对应的动作
                vehicle_action = vehicle_actions[slot_idx]
                unloading_action = unloading_actions[slot_idx]
                slot_idx += 1
                
                # vehicle_action=0 表示不操作
                if vehicle_action == 0:
                    continue
                
                # 车辆ID (action - 1)
                vehicle_id = vehicle_action - 1
                if vehicle_id < 0 or vehicle_id >= self.agent.num_vehicles:
                    continue
                
                # 检查车辆是否可用
                vehicle = self.env.vehicles.get(vehicle_id)
                if vehicle is None:
                    continue
                # 同周期同车只能接1单；且已有任务的不再接新单（避免覆盖 assigned_task）
                if vehicle_id in used_vehicle_ids:
                    continue
                if vehicle.assigned_task is not None:
                    continue
                
                # 找到车辆的空工位
                vehicle_slot_idx = vehicle.get_empty_slot_idx()
                if vehicle_slot_idx is None:
                    continue
                
                # 不再检查对齐，让规则控制去处理
                # 车辆会自动向目标移动，到达后再执行取货
                
                # 检查工位是否正在操作
                if vehicle.slot_operation_end_time[vehicle_slot_idx] > 0:
                    continue
                
                # 解码下料工位
                # unloading_action 是全局索引：station_id * 2 + slot_idx
                unloading_station_id = unloading_action // 2
                unloading_slot_idx = unloading_action % 2
                
                if unloading_station_id >= self.agent.num_unloading_stations:
                    continue
                
                # 检查货物是否允许去该下料口
                cargo = self.env.cargos.get(cargo_id)
                if cargo is None or unloading_station_id not in cargo.allowed_unloading_stations:
                    continue
                
                # 检查下料工位是否可用
                unloading_station = self.env.unloading_stations.get(unloading_station_id)
                if unloading_station is None:
                    continue
                
                if (unloading_station_id, unloading_slot_idx) in used_unloading_slots:
                    continue
                if unloading_station.slot_reserved[unloading_slot_idx]:
                    continue
                if unloading_station.slots[unloading_slot_idx] is not None:
                    continue
                
                # 生成任务分配
                assignments.append({
                    'type': 'assign_loading_with_target',
                    'cargo_id': cargo_id,
                    'vehicle_id': vehicle_id,
                    'vehicle_slot_idx': vehicle_slot_idx,
                    'unloading_station_id': unloading_station_id,
                    'unloading_slot_idx': unloading_slot_idx
                })
                used_vehicle_ids.add(vehicle_id)
                used_unloading_slots.add((unloading_station_id, unloading_slot_idx))
        
        return assignments
