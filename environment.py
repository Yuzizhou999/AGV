"""
环形轨道双工位穿梭车调度系统 - 环境模块
实现车辆、货物、上下料口等环境模型
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from config import *


class SlotState(Enum):
    """工位状态"""
    EMPTY = 0  # 空
    OCCUPIED = 1  # 有货


@dataclass
class Cargo:
    """货物对象"""
    id: int  # 货物唯一ID
    arrival_time: float  # 到达时间
    loading_station: int  # 在哪个上料工位
    loading_slot: int  # 在上料工位的哪个工位 (0: 1#, 1: 2#)
    allowed_unloading_stations: Set[int]  # 允许的下料口集合
    current_location: str  # 当前位置: "IP_{id}_{slot}" 或 "vehicle_{id}_{slot}" 或 "OP_{id}_{slot}"
    target_unloading_station: Optional[int] = None  # 目标下料口
    target_slot: Optional[int] = None  # 目标下料口的工位
    completion_time: Optional[float] = None  # 完成时间
    
    def wait_time(self, current_time: float) -> float:
        """计算等待时间"""
        return current_time - self.arrival_time
    
    def is_timeout(self, current_time: float) -> bool:
        """检查是否超时"""
        return self.wait_time(current_time) > CARGO_TIMEOUT


@dataclass
class Vehicle:
    """车辆对象"""
    id: int  # 车辆ID
    position: float  # 当前位置 [0, L)
    velocity: float  # 当前速度
    slots: List[Optional[int]]  # 两个工位上的货物ID (None表示空)
    slot_operation_end_time: List[float]  # 每个工位的操作结束时间
    
    def __post_init__(self):
        if len(self.slots) != 2:
            self.slots = [None, None]
        if len(self.slot_operation_end_time) != 2:
            self.slot_operation_end_time = [0.0, 0.0]
    
    def has_empty_slot(self) -> bool:
        """是否有空工位"""
        return None in self.slots
    
    def get_empty_slot_idx(self) -> Optional[int]:
        """获取第一个空工位索引"""
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i
        return None
    
    def distance_to(self, position: float) -> float:
        """计算到目标位置的距离(考虑环形轨道)"""
        direct = abs(position - self.position)
        wrap = TRACK_LENGTH - direct
        return min(direct, wrap)
    
    def is_aligned_with(self, station_position: float, tolerance: float = 1.0) -> bool:
        """判断是否与某工位对齐"""
        return self.distance_to(station_position) <= tolerance


class LoadingStation:
    """上料工位"""
    def __init__(self, id: int, position: float):
        self.id = id
        self.position = position
        self.slots: List[Optional[int]] = [None, None]  # 两个工位上的货物ID
    
    def has_empty_slot(self) -> bool:
        """是否有空工位"""
        return None in self.slots
    
    def place_cargo(self, cargo_id: int, slot_idx: int) -> bool:
        """在工位放置货物"""
        if self.slots[slot_idx] is None:
            self.slots[slot_idx] = cargo_id
            return True
        return False
    
    def remove_cargo(self, slot_idx: int) -> Optional[int]:
        """从工位移除货物"""
        cargo_id = self.slots[slot_idx]
        self.slots[slot_idx] = None
        return cargo_id


class UnloadingStation:
    """下料工位"""
    def __init__(self, id: int, position: float):
        self.id = id
        self.position = position
        self.slots: List[Optional[int]] = [None, None]  # 两个工位上的货物ID
        self.slot_reserved: List[bool] = [False, False]  # 是否被预订
    
    def has_empty_slot(self) -> bool:
        """是否有空工位"""
        return None in self.slots and not all(self.slot_reserved)
    
    def get_available_slot(self) -> Optional[int]:
        """获取可用工位"""
        for i in range(2):
            if self.slots[i] is None and not self.slot_reserved[i]:
                return i
        return None
    
    def place_cargo(self, cargo_id: int, slot_idx: int) -> bool:
        """在工位放置货物"""
        if self.slots[slot_idx] is None:
            self.slots[slot_idx] = cargo_id
            return True
        return False
    
    def remove_cargo(self, slot_idx: int) -> Optional[int]:
        """从工位移除货物"""
        cargo_id = self.slots[slot_idx]
        self.slots[slot_idx] = None
        return cargo_id


class Environment:
    """环形轨道双工位穿梭车调度环境"""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        # 车辆
        self.vehicles: Dict[int, Vehicle] = {}
        for i in range(MAX_VEHICLES):
            self.vehicles[i] = Vehicle(
                id=i,
                position=float(i * TRACK_LENGTH / MAX_VEHICLES),
                velocity=0.0,
                slots=[None, None],
                slot_operation_end_time=[0.0, 0.0]
            )
        
        # 上料口
        self.loading_stations: Dict[int, LoadingStation] = {}
        for i in range(NUM_LOADING_STATIONS):
            self.loading_stations[i] = LoadingStation(i, LOADING_POSITIONS[i])
        
        # 下料口
        self.unloading_stations: Dict[int, UnloadingStation] = {}
        for i in range(NUM_UNLOADING_STATIONS):
            self.unloading_stations[i] = UnloadingStation(i, UNLOADING_POSITIONS[i])
        
        # 货物管理
        self.cargos: Dict[int, Cargo] = {}
        self.cargo_counter = 0
        
        # 事件管理
        self.current_time = 0.0
        self.next_arrival_time = np.random.uniform(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX)
        
        # 统计信息
        self.completed_cargos = 0
        self.timed_out_cargos = 0
        self.total_wait_time = 0.0
    
    def reset(self):
        """重置环境"""
        self.__init__(seed=None)
        return self._get_observation()
    
    def _normalize_position(self, pos: float) -> float:
        """规范化位置到[0, L)"""
        return pos % TRACK_LENGTH
    
    def _check_and_generate_cargo(self) -> List[int]:
        """检查是否需要生成新货物，返回生成的货物ID列表"""
        new_cargo_ids = []
        
        while self.current_time >= self.next_arrival_time:
            # 选择随机上料口和工位
            loading_station_id = np.random.randint(0, NUM_LOADING_STATIONS)
            slot_idx = np.random.randint(0, 2)
            
            # 检查工位是否空闲
            station = self.loading_stations[loading_station_id]
            if station.slots[slot_idx] is None:
                # 随机选择允许的下料口（至少选择1个）
                num_allowed = np.random.randint(1, NUM_UNLOADING_STATIONS + 1)
                allowed_stations = set(np.random.choice(
                    NUM_UNLOADING_STATIONS, num_allowed, replace=False
                ))
                
                cargo = Cargo(
                    id=self.cargo_counter,
                    arrival_time=self.current_time,
                    loading_station=loading_station_id,
                    loading_slot=slot_idx,
                    allowed_unloading_stations=allowed_stations,
                    current_location=f"IP_{loading_station_id}_{slot_idx}"
                )
                
                self.cargos[self.cargo_counter] = cargo
                station.slots[slot_idx] = self.cargo_counter
                new_cargo_ids.append(self.cargo_counter)
                self.cargo_counter += 1
            
            # 计划下一次到达
            self.next_arrival_time += np.random.uniform(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX)
        
        return new_cargo_ids
    
    def step(self, high_level_action: Dict, low_level_actions: Dict) -> Tuple[Dict, float, bool]:
        """
        执行一步模拟
        
        Args:
            high_level_action: 高层动作字典
            low_level_actions: 低层动作字典 {vehicle_id: action}
        
        Returns:
            observation, reward, done
        """
        # 更新时间
        self.current_time += LOW_LEVEL_CONTROL_INTERVAL
        done = self.current_time >= EPISODE_DURATION
        
        # 检查是否有新货物到达
        self._check_and_generate_cargo()
        
        # 执行低层控制（更新车辆位置和速度）
        self._execute_low_level_control(low_level_actions)
        
        # 执行高层任务分配
        self._execute_high_level_action(high_level_action)
        
        # 检查完成情况
        completed_ids = self._check_completions()
        
        # 计算奖励
        reward = self._calculate_reward(completed_ids)
        
        # 检查超时货物
        self._check_timeouts()
        
        obs = self._get_observation()
        
        return obs, reward, done
    
    def _execute_low_level_control(self, actions: Dict):
        """执行低层控制：更新车辆位置和速度"""
        for vehicle_id, action in actions.items():
            vehicle = self.vehicles[vehicle_id]
            
            # action: 0=减速, 1=保持, 2=加速
            if action == 0:
                new_velocity = max(-MAX_SPEED, vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
            elif action == 1:
                new_velocity = vehicle.velocity
            else:  # action == 2
                new_velocity = min(MAX_SPEED, vehicle.velocity + MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
            
            # 检查安全距离约束
            if not self._check_safety_distance(vehicle_id, new_velocity):
                new_velocity = 0.0  # 强制停止
            
            vehicle.velocity = new_velocity
            
            # 更新位置
            displacement = vehicle.velocity * LOW_LEVEL_CONTROL_INTERVAL
            vehicle.position = self._normalize_position(vehicle.position + displacement)
            
            # 更新工位操作时间
            for i in range(2):
                if vehicle.slot_operation_end_time[i] > 0:
                    vehicle.slot_operation_end_time[i] -= LOW_LEVEL_CONTROL_INTERVAL
                    if vehicle.slot_operation_end_time[i] < 0:
                        vehicle.slot_operation_end_time[i] = 0
    
    def _check_safety_distance(self, vehicle_id: int, new_velocity: float) -> bool:
        """检查是否满足安全距离约束"""
        vehicle = self.vehicles[vehicle_id]
        new_position = self._normalize_position(vehicle.position + new_velocity * LOW_LEVEL_CONTROL_INTERVAL)
        
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
            
            # 计算距离（沿行驶方向）
            if new_position < other_vehicle.position:
                distance = other_vehicle.position - new_position
            else:
                distance = TRACK_LENGTH - new_position + other_vehicle.position
            
            if distance < SAFETY_DISTANCE:
                return False
        
        return True
    
    def _execute_high_level_action(self, action: Dict):
        """执行高层动作：任务分配和流向决策"""
        if action is None:
            return
        
        action_type = action.get('type')
        
        if action_type == 'assign_loading':
            # 分配上料任务
            cargo_id = action.get('cargo_id')
            vehicle_id = action.get('vehicle_id')
            slot_idx = action.get('slot_idx')
            
            if (cargo_id in self.cargos and vehicle_id in self.vehicles and 
                self.vehicles[vehicle_id].slots[slot_idx] is None):
                self._assign_loading_task(cargo_id, vehicle_id, slot_idx)
        
        elif action_type == 'assign_unloading':
            # 分配下料目标
            cargo_id = action.get('cargo_id')
            unloading_station_id = action.get('unloading_station_id')
            slot_idx = action.get('slot_idx')
            
            if cargo_id in self.cargos:
                cargo = self.cargos[cargo_id]
                if (unloading_station_id in cargo.allowed_unloading_stations and
                    self.unloading_stations[unloading_station_id].has_empty_slot()):
                    cargo.target_unloading_station = unloading_station_id
                    cargo.target_slot = slot_idx
    
    def _assign_loading_task(self, cargo_id: int, vehicle_id: int, slot_idx: int):
        """分配上料任务"""
        cargo = self.cargos[cargo_id]
        vehicle = self.vehicles[vehicle_id]
        loading_station = self.loading_stations[cargo.loading_station]
        
        # 从上料口移除货物
        loading_station.slots[cargo.loading_slot] = None
        
        # 放到车上
        vehicle.slots[slot_idx] = cargo_id
        cargo.current_location = f"vehicle_{vehicle_id}_{slot_idx}"
    
    def _check_completions(self) -> List[int]:
        """检查是否有货物完成"""
        completed_ids = []
        
        # 检查车上的货物是否可以卸货
        for vehicle_id, vehicle in self.vehicles.items():
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is None:
                    continue
                
                cargo = self.cargos[cargo_id]
                
                # 检查是否有卸货目标
                if cargo.target_unloading_station is None:
                    continue
                
                unloading_station = self.unloading_stations[cargo.target_unloading_station]
                
                # 检查是否对齐
                if not vehicle.is_aligned_with(unloading_station.position):
                    continue
                
                # 检查操作是否完成
                if vehicle.slot_operation_end_time[slot_idx] > 0:
                    continue
                
                # 执行卸货
                vehicle.slots[slot_idx] = None
                unloading_station.slots[cargo.target_slot] = cargo_id
                cargo.current_location = f"OP_{cargo.target_unloading_station}_{cargo.target_slot}"
                cargo.completion_time = self.current_time
                
                completed_ids.append(cargo_id)
                self.completed_cargos += 1
        
        return completed_ids
    
    def _check_timeouts(self):
        """检查超时货物"""
        for cargo in self.cargos.values():
            if cargo.completion_time is not None:
                continue
            if cargo.is_timeout(self.current_time):
                self.timed_out_cargos += 1
    
    def _calculate_reward(self, completed_ids: List[int]) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 完成卸货奖励
        reward += len(completed_ids) * REWARD_DELIVERY
        
        # 等待惩罚
        for cargo in self.cargos.values():
            if cargo.completion_time is None and cargo.id not in completed_ids:
                reward -= REWARD_WAIT_PENALTY_COEFF * cargo.wait_time(self.current_time) * LOW_LEVEL_CONTROL_INTERVAL
        
        # 超时惩罚（在_check_timeouts中检查）
        
        return reward
    
    def _get_observation(self) -> Dict:
        """获取观测"""
        # 车辆信息
        vehicle_obs = []
        for vehicle in self.vehicles.values():
            vehicle_obs.append({
                'position': vehicle.position / TRACK_LENGTH,
                'velocity': vehicle.velocity / MAX_SPEED,
                'slots': vehicle.slots.copy(),
                'slot_occupied': [slot is not None for slot in vehicle.slots]
            })
        
        # 上料口信息
        loading_obs = []
        for station in self.loading_stations.values():
            station_obs = {
                'position': station.position / TRACK_LENGTH,
                'slots': station.slots.copy(),
                'slot_occupied': [slot is not None for slot in station.slots]
            }
            loading_obs.append(station_obs)
        
        # 下料口信息
        unloading_obs = []
        for station in self.unloading_stations.values():
            station_obs = {
                'position': station.position / TRACK_LENGTH,
                'slots': station.slots.copy(),
                'slot_occupied': [slot is not None for slot in station.slots]
            }
            unloading_obs.append(station_obs)
        
        # 待取货物信息
        waiting_cargos = []
        for cargo in self.cargos.values():
            if cargo.completion_time is None and cargo.current_location.startswith("IP_"):
                waiting_cargos.append({
                    'id': cargo.id,
                    'wait_time': cargo.wait_time(self.current_time),
                    'is_timeout': cargo.is_timeout(self.current_time)
                })
        
        # 全局信息
        global_info = {
            'current_time': self.current_time / EPISODE_DURATION,
            'total_cargos': len(self.cargos),
            'completed_cargos': self.completed_cargos,
            'waiting_cargos': len(waiting_cargos),
            'timed_out_cargos': self.timed_out_cargos,
            'avg_wait_time': np.mean([c.wait_time(self.current_time) for c in self.cargos.values()]) if self.cargos else 0.0
        }
        
        return {
            'vehicles': vehicle_obs,
            'loading_stations': loading_obs,
            'unloading_stations': unloading_obs,
            'waiting_cargos': waiting_cargos,
            'global_info': global_info
        }
    
    def get_high_level_observation(self) -> np.ndarray:
        """获取高层观测的向量表示"""
        obs_list = []
        
        # 车辆信息
        for vehicle in self.vehicles.values():
            obs_list.extend([
                vehicle.position / TRACK_LENGTH,
                vehicle.velocity / MAX_SPEED,
                float(vehicle.slots[0] is not None),
                float(vehicle.slots[1] is not None)
            ])
        
        # 上料口信息
        for station in self.loading_stations.values():
            obs_list.extend([
                station.position / TRACK_LENGTH,
                float(station.slots[0] is not None),
                float(station.slots[1] is not None)
            ])
        
        # 全局信息
        waiting_count = sum(1 for c in self.cargos.values() 
                          if c.completion_time is None and c.current_location.startswith("IP_"))
        obs_list.extend([
            self.current_time / EPISODE_DURATION,
            waiting_count / max(10, self.cargo_counter)
        ])
        
        return np.array(obs_list, dtype=np.float32)
    
    def get_low_level_observation(self, vehicle_id: int) -> np.ndarray:
        """获取特定车辆的低层观测"""
        vehicle = self.vehicles[vehicle_id]
        obs_list = [
            vehicle.position / TRACK_LENGTH,
            vehicle.velocity / MAX_SPEED
        ]
        
        # 与前车距离
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
            if other_vehicle.position > vehicle.position:
                distance = other_vehicle.position - vehicle.position
            else:
                distance = TRACK_LENGTH - vehicle.position + other_vehicle.position
            obs_list.append(distance / TRACK_LENGTH)
        
        # 目标点距离
        target_found = False
        for cargo_id in vehicle.slots:
            if cargo_id is not None:
                cargo = self.cargos[cargo_id]
                if cargo.target_unloading_station is not None:
                    target_pos = self.unloading_stations[cargo.target_unloading_station].position
                    target_distance = vehicle.distance_to(target_pos)
                    obs_list.append(target_distance / TRACK_LENGTH)
                    target_found = True
                    break
        
        if not target_found:
            obs_list.append(0.0)
        
        return np.array(obs_list, dtype=np.float32)
