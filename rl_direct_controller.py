"""
直接控制器
动态生成动作空间、构建语义状态、计算即时奖励
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import *


# 最大动作数：NOOP(1) + 上料分配(MAX_CARGOS * MAX_VEHICLES * 2) + 下料分配(MAX_VEHICLES * 2 * NUM_UNLOADING)
MAX_ACTIONS = 100


class DirectController:
    """直接控制器"""
    
    def __init__(self, env):
        self.env = env
        self.max_actions = MAX_ACTIONS
        
        # 动作映射（每次重建）
        self.action_list = []
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        # 全局特征: 时间(1) + 待取货物数(1) + 超时货物数(1) = 3
        # 车辆特征: 每辆车 位置(1) + 速度(1) + 工位0占用(1) + 工位1占用(1) + 负载(1) = 5 * MAX_VEHICLES
        # 上料口特征: 每个站 有货(1) + 最长等待时间(1) + 是否超时(1) = 3 * NUM_LOADING_STATIONS
        # 分配上下文特征: 15个（预留）
        return 3 + 5 * MAX_VEHICLES + 3 * NUM_LOADING_STATIONS * 2 + 15
    
    def get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        state = []
        
        # 全局特征
        state.append(self.env.current_time / EPISODE_DURATION)
        
        # 待取货物数
        waiting_count = sum(1 for c in self.env.cargos.values()
                           if c.completion_time is None 
                           and c.current_location.startswith("IP_")
                           and c.assigned_vehicle is None)
        state.append(min(waiting_count / 10.0, 1.0))
        
        # 超时货物数
        timeout_count = sum(1 for c in self.env.cargos.values()
                           if c.completion_time is None 
                           and c.current_location.startswith("IP_")
                           and c.is_timeout(self.env.current_time))
        state.append(min(timeout_count / 5.0, 1.0))
        
        # 车辆特征
        for vehicle in self.env.vehicles.values():
            state.append(vehicle.position / TRACK_LENGTH)
            state.append(vehicle.velocity / MAX_SPEED)
            state.append(1.0 if vehicle.slots[0] is not None else 0.0)
            state.append(1.0 if vehicle.slots[1] is not None else 0.0)
            # 负载（包括已分配未取的）
            load = sum(1 for slot in vehicle.slots if slot is not None)
            for cargo in self.env.cargos.values():
                if cargo.assigned_vehicle == vehicle.id and cargo.current_location.startswith("IP_"):
                    load += 1
            state.append(load / 4.0)  # 最大负载4
        
        # 上料口特征
        for station in self.env.loading_stations.values():
            for slot_idx in range(2):
                cargo_id = station.slots[slot_idx]
                if cargo_id is not None and cargo_id in self.env.cargos:
                    cargo = self.env.cargos[cargo_id]
                    is_assigned = cargo.assigned_vehicle is not None
                    state.append(0.0 if is_assigned else 1.0)  # 未分配的货物
                    wait_time = cargo.wait_time(self.env.current_time)
                    state.append(min(wait_time / 300.0, 1.0))
                    state.append(1.0 if cargo.is_timeout(self.env.current_time) else 0.0)
                else:
                    state.extend([0.0, 0.0, 0.0])
        
        # 填充到固定长度
        target_len = self.get_state_dim()
        while len(state) < target_len:
            state.append(0.0)
        
        return np.array(state[:target_len], dtype=np.float32)
    
    def build_actions(self) -> Tuple[List[Dict], np.ndarray]:
        """
        构建当前可用的动作列表和动作掩码
        
        Returns:
            action_list: 动作列表
            action_mask: 动作掩码 [max_actions], 1=有效
        """
        self.action_list = []
        action_mask = np.zeros(self.max_actions, dtype=np.float32)
        
        # 动作0: NOOP（总是有效）
        self.action_list.append({'type': 'noop'})
        action_mask[0] = 1.0
        
        action_idx = 1
        
        # 上料分配动作
        # 获取未分配的货物
        waiting_cargos = []
        for cargo in self.env.cargos.values():
            if (cargo.completion_time is None and 
                cargo.current_location.startswith("IP_") and
                cargo.assigned_vehicle is None):
                waiting_cargos.append(cargo)
        
        # 获取可用车辆工位
        available_slots = []
        for vehicle_id, vehicle in self.env.vehicles.items():
            for slot_idx in range(2):
                if vehicle.slots[slot_idx] is None:
                    # 检查是否已被预分配
                    is_assigned = False
                    for cargo in self.env.cargos.values():
                        if (cargo.assigned_vehicle == vehicle_id and 
                            cargo.assigned_vehicle_slot == slot_idx):
                            is_assigned = True
                            break
                    if not is_assigned:
                        available_slots.append((vehicle_id, slot_idx))
        
        # 生成上料分配动作
        for cargo in waiting_cargos:
            for vehicle_id, slot_idx in available_slots:
                if action_idx >= self.max_actions:
                    break
                self.action_list.append({
                    'type': 'assign_loading',
                    'cargo_id': cargo.id,
                    'vehicle_id': vehicle_id,
                    'slot_idx': slot_idx
                })
                action_mask[action_idx] = 1.0
                action_idx += 1
            if action_idx >= self.max_actions:
                break
        
        # 下料分配动作
        for vehicle_id, vehicle in self.env.vehicles.items():
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is None:
                    continue
                cargo = self.env.cargos.get(cargo_id)
                if cargo is None or cargo.target_unloading_station is not None:
                    continue
                
                # 为该货物生成下料选项
                for station_id in cargo.allowed_unloading_stations:
                    station = self.env.unloading_stations[station_id]
                    available_slot = station.get_available_slot()
                    if available_slot is not None:
                        if action_idx >= self.max_actions:
                            break
                        self.action_list.append({
                            'type': 'assign_unloading',
                            'cargo_id': cargo_id,
                            'unloading_station_id': station_id,
                            'slot_idx': available_slot
                        })
                        action_mask[action_idx] = 1.0
                        action_idx += 1
                if action_idx >= self.max_actions:
                    break
            if action_idx >= self.max_actions:
                break
        
        return self.action_list, action_mask
    
    def decode_action(self, action_idx: int) -> Optional[Dict]:
        """解码动作"""
        if action_idx < len(self.action_list):
            action = self.action_list[action_idx]
            if action['type'] == 'noop':
                return None
            return action
        return None
    
    def compute_reward(self, action_idx: int, prev_completed: int, new_completed: int) -> float:
        """
        计算即时奖励
        
        Args:
            action_idx: 执行的动作
            prev_completed: 执行前完成数
            new_completed: 执行后完成数
        
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 完成货物的大奖励
        completed_delta = new_completed - prev_completed
        reward += completed_delta * 20.0
        
        if action_idx >= len(self.action_list):
            return reward
        
        action = self.action_list[action_idx]
        
        if action['type'] == 'noop':
            # NOOP时检查是否有待处理的任务
            has_waiting = any(c for c in self.env.cargos.values()
                            if c.completion_time is None 
                            and c.current_location.startswith("IP_")
                            and c.assigned_vehicle is None)
            has_available_slot = any(
                vehicle.has_empty_slot() for vehicle in self.env.vehicles.values()
            )
            # 如果有待分配货物且有空位，NOOP应该受到惩罚
            if has_waiting and has_available_slot:
                reward -= 0.5
        
        elif action['type'] == 'assign_loading':
            cargo_id = action['cargo_id']
            vehicle_id = action['vehicle_id']
            
            if cargo_id in self.env.cargos:
                cargo = self.env.cargos[cargo_id]
                vehicle = self.env.vehicles[vehicle_id]
                loading_station = self.env.loading_stations[cargo.loading_station]
                
                # 距离奖励（越近越好）
                distance = vehicle.distance_to(loading_station.position)
                reward += (1.0 - distance / TRACK_LENGTH) * 2.0
                
                # 超时货物优先奖励
                if cargo.is_timeout(self.env.current_time):
                    reward += 5.0
                
                # 等待时间奖励（处理等待久的货物）
                wait_time = cargo.wait_time(self.env.current_time)
                reward += min(wait_time / 60.0, 2.0)  # 最多2分
                
                # 分配成功基础奖励
                reward += 1.0
        
        elif action['type'] == 'assign_unloading':
            # 分配下料成功奖励
            reward += 1.0
            
            cargo_id = action['cargo_id']
            if cargo_id in self.env.cargos:
                cargo = self.env.cargos[cargo_id]
                vehicle = self.env.vehicles.get(cargo.assigned_vehicle)
                if vehicle:
                    station = self.env.unloading_stations[action['unloading_station_id']]
                    distance = vehicle.distance_to(station.position)
                    # 距离奖励
                    reward += (1.0 - distance / TRACK_LENGTH) * 1.0
        
        return reward
