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
    assigned_vehicle: Optional[int] = None  # 分配的车辆ID（用于上料任务）
    assigned_vehicle_slot: Optional[int] = None  # 分配的车辆工位
    loading_start_time: Optional[float] = None  # 上料开始时间
    unloading_start_time: Optional[float] = None  # 下料开始时间
    picked_up_time: Optional[float] = None  # 被取走的时间（小车开始取货的时间，用于超时判断）
    
    def wait_time(self, current_time: float) -> float:
        """计算等待时间（从到达到被取走）"""
        # 如果已经被取走（小车开始取货），等待时间就是到被取走的时间
        if self.picked_up_time is not None:
            return self.picked_up_time - self.arrival_time
        # 如果还未被取走，等待时间是到当前时间
        return current_time - self.arrival_time
    
    def is_timeout(self, current_time: float) -> bool:
        """检查是否超时（只在上料前计算）"""
        # 如果已经被取走（小车开始取货了），就不再超时
        if self.picked_up_time is not None:
            return False
        # 否则检查等待时间是否超过阈值
        return self.wait_time(current_time) > CARGO_TIMEOUT


@dataclass
class Vehicle:
    """车辆对象"""
    id: int  # 车辆ID
    position: float  # 当前位置 [0, L)
    velocity: float  # 当前速度
    slots: List[Optional[int]]  # 两个工位上的货物ID (None表示空)
    slot_operation_end_time: List[float]  # 每个工位的操作结束时间
    is_loading_unloading: bool = False  # 是否正在进行上料/下料操作（锁定移动）
    assigned_tasks: List[Dict] = None  # 任务队列：记录高层分配的任务
    
    def __post_init__(self):
        if len(self.slots) != 2:
            self.slots = [None, None]
        if len(self.slot_operation_end_time) != 2:
            self.slot_operation_end_time = [0.0, 0.0]
        if self.assigned_tasks is None:
            self.assigned_tasks = []
    
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
        direct = position - self.position
        if direct < 0:
            direct += TRACK_LENGTH
        return direct
    
    def is_aligned_with(self, station_position: float, tolerance: float = 1.0) -> bool:
        """判断是否与某工位对齐（考虑双向距离和速度限制）
        
        Args:
            station_position: 工位位置
            tolerance: 位置容差
        
        Returns:
            bool: 是否对齐（位置对齐且速度足够低）
        """
        # 计算双向距离，取最小值
        forward_dist = self.distance_to(station_position)  # 顺时针距离
        backward_dist = TRACK_LENGTH - forward_dist  # 逆时针距离
        min_dist = min(forward_dist, backward_dist)
        
        # 位置对齐且速度低于容差
        position_aligned = min_dist <= tolerance
        speed_ok = abs(self.velocity) <= SPEED_TOLERANCE
        
        return position_aligned and speed_ok


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
                slot_operation_end_time=[0.0, 0.0],
                is_loading_unloading=False
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
        # 货物到达间隔使用随机整数（5-15秒）
        self.next_arrival_time = float(np.random.randint(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX + 1))
        
        # 统计信息
        self.completed_cargos = 0
        self.timed_out_cargos = 0
        self.total_wait_time = 0.0
        self.completed_cargo_list = []  # 保存已完成货物的详细信息
    
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
            
            # 计划下一次到达（使用随机整数5-15秒）
            self.next_arrival_time += float(np.random.randint(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX + 1))
        
        return new_cargo_ids
    
    def step(self, high_level_action: Dict, low_level_actions: Dict = None) -> Tuple[Dict, float, bool]:
        """
        执行一步模拟
        
        Args:
            high_level_action: 高层动作字典
            low_level_actions: 低层动作字典 {vehicle_id: action}，如果为None则使用启发式控制
        
        Returns:
            observation, reward, done
        """
        # 更新时间
        self.current_time += LOW_LEVEL_CONTROL_INTERVAL
        done = self.current_time >= EPISODE_DURATION
        
        # 检查是否有新货物到达
        self._check_and_generate_cargo()
        
        # 如果没有提供低层动作，使用启发式控制器
        if low_level_actions is None:
            from heuristic_controller import HeuristicLowLevelController
            if not hasattr(self, 'heuristic_controller'):
                self.heuristic_controller = HeuristicLowLevelController(self)
            low_level_actions = self.heuristic_controller.get_actions()
        
        # 执行低层控制（更新车辆位置和速度）
        self._execute_low_level_control(low_level_actions)
        
        # 执行高层任务分配，获取分配事件
        assigned_ids = self._execute_high_level_action(high_level_action)
        
        # 处理上料和下料操作（需要在位置更新后执行）
        picked_up_ids = self._process_loading_operations()   
        completed_ids = self._process_unloading_operations()
        
        # 计算奖励
        reward = self._calculate_reward(completed_ids, picked_up_ids, assigned_ids)
        
        # 检查超时货物
        self._check_timeouts()
        
        obs = self._get_observation()
        
        return obs, reward, done
    
    def _execute_low_level_control(self, actions: Dict):
        """执行低层控制：更新车辆位置和速度"""
        for vehicle_id, action in actions.items():
            vehicle = self.vehicles[vehicle_id]
            
            # 如果车辆正在进行上料/下料操作，强制锁定不移动
            if vehicle.is_loading_unloading:
                vehicle.velocity = 0.0  # 强制停止
                # 不更新位置，直接跳过
                continue
            
            # action: 0=减速, 1=保持, 2=加速
            # 轨道坐标定义为沿顺时针方向，速度不允许为负（不支持反向行驶）
            if action == 0:
                new_velocity = max(0, vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
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
        """检查是否满足安全距离约束
        
        考虑其他车辆可能的加速/减速情况，检查最坏情况下的距离。
        
        Args:
            vehicle_id: 要检查的车辆ID
            new_velocity: 车辆新的速度
        
        Returns:
            bool: 如果满足安全距离约束返回True，否则返回False
        """
        vehicle = self.vehicles[vehicle_id]
        # 本车在下一时间步的位置
        new_position = self._normalize_position(vehicle.position + new_velocity * LOW_LEVEL_CONTROL_INTERVAL)
        
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
            
            # 计算其他车辆在下一时间步可能的位置范围
            # 最坏情况1：其他车以最大加速度加速
            other_max_velocity = min(MAX_SPEED, other_vehicle.velocity + MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
            other_max_position = self._normalize_position(other_vehicle.position + other_max_velocity * LOW_LEVEL_CONTROL_INTERVAL)
            
            # 最坏情况2：其他车以最大减速度减速（或维持当前位置）
            other_min_velocity = max(0, other_vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
            other_min_position = self._normalize_position(other_vehicle.position + other_min_velocity * LOW_LEVEL_CONTROL_INTERVAL)
            
            # 当前其他车的位置（基准）
            other_current_position = other_vehicle.position
            
            # 计算本车到其他车可能位置的最小距离
            # 考虑三种情况：其他车最大位置、最小位置、当前位置
            min_distance = float('inf')
            
            for other_check_position in [other_current_position, other_max_position, other_min_position]:
                # 计算沿行驶方向的距离（顺时针）
                if new_position <= other_check_position:
                    distance = other_check_position - new_position
                else:
                    distance = TRACK_LENGTH - new_position + other_check_position
                
                min_distance = min(min_distance, distance)
            
            # 检查最坏情况下是否满足安全距离
            if min_distance < SAFETY_DISTANCE:
                return False
        
        return True
    
    def _execute_high_level_action(self, action: Dict) -> List[int]:
        """执行高层动作：任务分配和流向决策
        
        Returns:
            List[int]: 本次新分配的货物ID列表
        """
        assigned_ids = []
        
        if action is None:
            return assigned_ids
        
        action_type = action.get('type')
        
        if action_type == 'assign_loading':
            # 分配上料任务
            cargo_id = action.get('cargo_id')
            vehicle_id = action.get('vehicle_id')
            slot_idx = action.get('slot_idx')
            
            if (cargo_id in self.cargos and vehicle_id in self.vehicles and 
                self.vehicles[vehicle_id].slots[slot_idx] is None):
                cargo = self.cargos[cargo_id]
                # 只有首次分配才记录（避免重复分配奖励）
                if cargo.assigned_vehicle is None:
                    assigned_ids.append(cargo_id)
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
                    
                    # 在车辆任务队列中添加下料任务
                    if cargo.assigned_vehicle is not None:
                        vehicle = self.vehicles[cargo.assigned_vehicle]
                        unloading_station = self.unloading_stations[unloading_station_id]
                        task = {
                            'type': 'unloading',
                            'cargo_id': cargo_id,
                            'target_position': unloading_station.position,
                            'station_id': unloading_station_id,
                            'slot_idx': slot_idx
                        }
                        
                        # 避免重复添加任务
                        task_exists = any(
                            t['type'] == 'unloading' and t['cargo_id'] == cargo_id 
                            for t in vehicle.assigned_tasks
                        )
                        if not task_exists:
                            vehicle.assigned_tasks.append(task)
        
        return assigned_ids
    
    def _assign_loading_task(self, cargo_id: int, vehicle_id: int, slot_idx: int):
        """分配上料任务（只标记任务，实际上料在车辆对齐时执行）"""
        cargo = self.cargos[cargo_id]
        vehicle = self.vehicles[vehicle_id]
        
        # 标记货物任务分配
        cargo.assigned_vehicle = vehicle_id
        cargo.assigned_vehicle_slot = slot_idx
        
        # 在车辆任务队列中添加上料任务
        loading_station = self.loading_stations[cargo.loading_station]
        task = {
            'type': 'loading',
            'cargo_id': cargo_id,
            'target_position': loading_station.position,
            'station_id': cargo.loading_station,
            'slot_idx': slot_idx
        }
        
        # 避免重复添加任务
        task_exists = any(
            t['type'] == 'loading' and t['cargo_id'] == cargo_id 
            for t in vehicle.assigned_tasks
        )
        if not task_exists:
            vehicle.assigned_tasks.append(task)
    
    def _process_loading_operations(self):
        """处理上料操作：检查车辆是否对齐上料口，执行上料
        支持双工位同时上料以提高效率
        
        Returns:
            List[int]: 本次完成取货的货物ID列表
        """
        picked_up_ids = []
        
        for cargo in self.cargos.values():
            # 只处理在上料口等待且已分配车辆的货物
            if (cargo.completion_time is not None or 
                cargo.assigned_vehicle is None or
                not cargo.current_location.startswith("IP_")):
                continue
            
            vehicle = self.vehicles[cargo.assigned_vehicle]
            loading_station = self.loading_stations[cargo.loading_station]
            slot_idx = cargo.assigned_vehicle_slot
            
            # 检查车辆工位是否仍然空闲
            if vehicle.slots[slot_idx] is not None:
                continue
            
            # 如果还未开始上料，检查是否对齐
            if cargo.loading_start_time is None:
                # 检查是否对齐上料口
                if not vehicle.is_aligned_with(loading_station.position):
                    continue  # 未对齐，等待对齐
                
                # 对齐了，开始上料计时
                cargo.loading_start_time = self.current_time
                cargo.picked_up_time = self.current_time  # 记录被取走的时间
                vehicle.is_loading_unloading = True  # 锁定车辆移动
                vehicle.velocity = 0.0  # 立即停止车辆
                
                # 检查是否可以启用双工位同时上料
                # 条件：两个工位都有任务且都在同一个上料口
                for other_cargo in self.cargos.values():
                    if (other_cargo.id != cargo.id and
                        other_cargo.assigned_vehicle == cargo.assigned_vehicle and
                        other_cargo.loading_station == cargo.loading_station and
                        other_cargo.current_location.startswith("IP_") and
                        other_cargo.loading_start_time is None):
                        # 找到另一个工位的货物，同时开始上料
                        other_slot_idx = other_cargo.assigned_vehicle_slot
                        if vehicle.slots[other_slot_idx] is None:
                            other_cargo.loading_start_time = self.current_time
                            other_cargo.picked_up_time = self.current_time
                        break
                
                continue  # 本轮只开始计时，下一轮再检查完成
            
            # 上料已经在进行中，车辆应该保持锁定状态
            # 持续检查：上料过程中车辆应该始终对齐并锁定
            if not vehicle.is_aligned_with(loading_station.position):
                print(f"[ERROR] 车辆{vehicle.id}上料中失去对齐:")
                print(f"  车辆位置: {vehicle.position:.2f}")
                print(f"  上料口位置: {loading_station.position:.2f}")
                print(f"  货物ID: {cargo.id}")
                print(f"  上料开始时间: {cargo.loading_start_time:.2f}")
                print(f"  当前时间: {self.current_time:.2f}")
                assert False, f"车辆{vehicle.id}上料中失去对齐"
            
            if not vehicle.is_loading_unloading:
                print(f"[ERROR] 车辆{vehicle.id}上料中但未锁定:")
                print(f"  车辆位置: {vehicle.position:.2f}")
                print(f"  车辆速度: {vehicle.velocity:.2f}")
                print(f"  货物ID: {cargo.id}")
                print(f"  上料开始时间: {cargo.loading_start_time:.2f}")
                print(f"  当前时间: {self.current_time:.2f}")
                print(f"  上料进行时间: {self.current_time - cargo.loading_start_time:.2f}s")
                print(f"  车辆工位: slot0={vehicle.slots[0]}, slot1={vehicle.slots[1]}")
                assert False, f"车辆{vehicle.id}上料中但未锁定"
            
            # 检查是否完成（耗时15秒）
            if self.current_time - cargo.loading_start_time >= LOADING_TIME:
                # 执行上料：从上料口移除货物，放到车上
                loading_station.slots[cargo.loading_slot] = None
                vehicle.slots[slot_idx] = cargo.id
                cargo.current_location = f"vehicle_{cargo.assigned_vehicle}_{slot_idx}"
                cargo.loading_start_time = None
                # 注意：不在这里解除锁定，统一在函数最后处理
                picked_up_ids.append(cargo.id)  # 记录完成取货的货物
                
                # 从车辆任务队列中移除上料任务
                vehicle.assigned_tasks = [
                    t for t in vehicle.assigned_tasks 
                    if not (t['type'] == 'loading' and t['cargo_id'] == cargo.id)
                ]
        
        # 统一检查所有车辆：如果没有任何货物正在上料或下料，则解除锁定
        for vehicle in self.vehicles.values():
            if vehicle.is_loading_unloading:
                # 检查是否还有货物正在上料
                has_loading = False
                for cargo in self.cargos.values():
                    if (cargo.assigned_vehicle == vehicle.id and 
                        cargo.loading_start_time is not None):
                        has_loading = True
                        break
                
                # 检查是否还有货物正在下料
                has_unloading = False
                for slot_cargo_id in vehicle.slots:
                    if slot_cargo_id is not None and slot_cargo_id in self.cargos:
                        cargo = self.cargos[slot_cargo_id]
                        if cargo.unloading_start_time is not None:
                            has_unloading = True
                            break
                
                # 如果既没有上料也没有下料，解除锁定
                if not has_loading and not has_unloading:
                    vehicle.is_loading_unloading = False
        
        return picked_up_ids
    
    def _process_unloading_operations(self) -> List[int]:
        """处理下料操作：检查车辆是否对齐下料口，执行下料
        支持双工位同时下料以提高效率

        Returns:
            List[int]: 本次完成卸货的货物ID列表
        """
        completed_cargo_ids = []  # 记录本轮完成的货物ID（因为会被删除）
        
        for vehicle_id, vehicle in self.vehicles.items():
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is None:
                    continue
                
                cargo = self.cargos[cargo_id]
                
                # 检查是否有卸货目标
                if cargo.target_unloading_station is None:
                    continue
                
                unloading_station = self.unloading_stations[cargo.target_unloading_station]
                
                # 如果还未开始下料，检查是否对齐
                if cargo.unloading_start_time is None:
                    # 检查是否对齐下料口
                    if not vehicle.is_aligned_with(unloading_station.position):
                        continue  # 未对齐，等待对齐
                    
                    # 对齐了，开始下料计时
                    cargo.unloading_start_time = self.current_time
                    vehicle.is_loading_unloading = True  # 锁定车辆移动
                    vehicle.velocity = 0.0  # 立即停止车辆
                    
                    # 检查是否可以启用双工位同时下料
                    # 条件：两个工位都有货物且都去往同一个下料口
                    other_slot_idx = 1 - slot_idx  # 另一个工位（0->1, 1->0）
                    other_cargo_id = vehicle.slots[other_slot_idx]
                    if other_cargo_id is not None and other_cargo_id in self.cargos:
                        other_cargo = self.cargos[other_cargo_id]
                        if (other_cargo.target_unloading_station == cargo.target_unloading_station and
                            other_cargo.unloading_start_time is None):
                            # 两个工位都去往同一个下料口，同时开始下料
                            other_cargo.unloading_start_time = self.current_time
                    
                    continue  # 本轮只开始计时，下一轮再检查完成
                
                # 下料已经在进行中，车辆应该保持锁定状态
                # 持续检查：下料过程中车辆应该始终对齐并锁定
                if not vehicle.is_aligned_with(unloading_station.position):
                    print(f"[ERROR] 车辆{vehicle.id}下料中失去对齐:")
                    print(f"  车辆位置: {vehicle.position:.2f}")
                    print(f"  下料口位置: {unloading_station.position:.2f}")
                    print(f"  货物ID: {cargo.id}")
                    print(f"  下料开始时间: {cargo.unloading_start_time:.2f}")
                    print(f"  当前时间: {self.current_time:.2f}")
                    assert False, f"车辆{vehicle.id}下料中失去对齐"
                
                if not vehicle.is_loading_unloading:
                    print(f"[ERROR] 车辆{vehicle.id}下料中但未锁定:")
                    print(f"  车辆位置: {vehicle.position:.2f}")
                    print(f"  车辆速度: {vehicle.velocity:.2f}")
                    print(f"  货物ID: {cargo.id}")
                    print(f"  下料开始时间: {cargo.unloading_start_time:.2f}")
                    print(f"  当前时间: {self.current_time:.2f}")
                    print(f"  下料进行时间: {self.current_time - cargo.unloading_start_time:.2f}s")
                    print(f"  车辆工位: slot0={vehicle.slots[0]}, slot1={vehicle.slots[1]}")
                    assert False, f"车辆{vehicle.id}下料中但未锁定"
                
                # 检查是否完成（耗时15秒）
                if self.current_time - cargo.unloading_start_time >= UNLOADING_TIME:
                    # 执行下料：从车上移除货物，货物直接完成任务（不占用下料口slot）
                    vehicle.slots[slot_idx] = None
                    cargo.current_location = f"OP_{cargo.target_unloading_station}_{cargo.target_slot}_completed"
                    cargo.completion_time = self.current_time
                    cargo.unloading_start_time = None
                    self.completed_cargos += 1
                    
                    # 累加总等待时间
                    self.total_wait_time += cargo.wait_time(self.current_time)
                    
                    # 保存已完成货物的详细信息
                    completed_info = {
                        'id': cargo.id,
                        'arrival_time': cargo.arrival_time,
                        'completion_time': cargo.completion_time,
                        'wait_time': cargo.wait_time(self.current_time),
                        'loading_station': cargo.loading_station,
                        'unloading_station': cargo.target_unloading_station,
                        'vehicle_id': cargo.assigned_vehicle
                    }
                    self.completed_cargo_list.append(completed_info)
                    
                    # 下料口的slot保持空闲，不放置货物
                    # unloading_station.slots[cargo.target_slot] 保持为 None
                    
                    # 注意：不在这里解除锁定，统一在函数最后处理
                    # 记录已完成的货物ID
                    completed_cargo_ids.append(cargo_id)
                    
                    # 从车辆任务队列中移除下料任务
                    vehicle.assigned_tasks = [
                        t for t in vehicle.assigned_tasks 
                        if not (t['type'] == 'unloading' and t['cargo_id'] == cargo_id)
                    ]
        
        # 删除已完成的货物（在遍历后删除，避免遍历时修改字典）
        for cargo_id in completed_cargo_ids:
            del self.cargos[cargo_id]
        
        # 统一检查所有车辆：如果没有任何货物正在下料，则解除锁定
        for vehicle in self.vehicles.values():
            if vehicle.is_loading_unloading:
                # 检查是否还有货物正在下料
                has_unloading = False
                for slot_cargo_id in vehicle.slots:
                    if slot_cargo_id is not None and slot_cargo_id in self.cargos:
                        cargo = self.cargos[slot_cargo_id]
                        if cargo.unloading_start_time is not None:
                            has_unloading = True
                            break
                
                # 还需要检查是否有货物正在上料（因为可能同时存在上料和下料）
                has_loading = False
                for cargo in self.cargos.values():
                    if (cargo.assigned_vehicle == vehicle.id and 
                        cargo.loading_start_time is not None):
                        has_loading = True
                        break
                
                # 如果既没有上料也没有下料，解除锁定
                if not has_unloading and not has_loading:
                    vehicle.is_loading_unloading = False

        return completed_cargo_ids
    
    def _check_timeouts(self):
        """统计超时货物数量"""
        timeout_count = sum(1 for c in self.cargos.values()
                           if c.completion_time is None 
                           and c.is_timeout(self.current_time))
        self.timed_out_cargos = timeout_count  # 直接赋值，不累加
    
    def _calculate_reward(self, completed_ids: List[int], picked_up_ids: List[int], assigned_ids: List[int]) -> float:
        """计算奖励
        
        Args:
            completed_ids: 本次完成卸货的货物ID列表
            picked_up_ids: 本次完成取货的货物ID列表
            assigned_ids: 本次分配给小车的货物ID列表
        
        Returns:
            float: 奖励值
        """
        reward = 0.0
        
        # 完成卸货奖励(需要检查是否超时完成,并根据等待时间分级奖励)
        for cargo_id in completed_ids:
            # 从completed_cargo_list中获取刚完成的货物信息
            completed_cargo = next((c for c in self.completed_cargo_list if c['id'] == cargo_id), None)
            if completed_cargo:
                wait_time = completed_cargo['wait_time']
                if wait_time > CARGO_TIMEOUT:
                    # 超时完成:给予净惩罚(改进:从+1.0变为-8.0)
                    reward += REWARD_DELIVERY * 0.1  # 只给10%的完成奖励
                    reward += REWARD_TIMEOUT_PENALTY * 2  # 加倍超时惩罚
                elif wait_time < CARGO_TIMEOUT * 0.5:
                    # 快速完成(少于150秒):给予额外奖励
                    reward += REWARD_DELIVERY * 1.2  # 120%奖励
                else:
                    # 正常完成(150-300秒):给予完整奖励
                    reward += REWARD_DELIVERY
        
        # 完成取货奖励(优先取超时货物给予额外奖励)
        for cargo_id in picked_up_ids:
            cargo = self.cargos.get(cargo_id)
            if cargo:
                wait_time = cargo.wait_time(self.current_time)
                if wait_time > CARGO_TIMEOUT:
                    # 取货的是超时货物,给予额外奖励
                    reward += REWARD_PICKUP * 1.5  # 150%取货奖励
                else:
                    reward += REWARD_PICKUP
        
        # 分配货物奖励(降低以避免过度分配)
        reward += len(assigned_ids) * REWARD_ASSIGNMENT * 0.5
        
        # 等待惩罚（针对在上料口等待的货物）
        for cargo in self.cargos.values():
            if cargo.completion_time is None and cargo.id not in completed_ids:
                # 如果货物还在上料口等待（未被取走）
                if cargo.picked_up_time is None:
                    reward -= REWARD_WAIT_PENALTY_COEFF * cargo.wait_time(self.current_time) * LOW_LEVEL_CONTROL_INTERVAL
                    # 超时货物额外惩罚（优先级提升的体现）
                    if cargo.is_timeout(self.current_time):
                        reward += REWARD_TIMEOUT_PENALTY * LOW_LEVEL_CONTROL_INTERVAL / CARGO_TIMEOUT
        
        # 持有货物惩罚（针对车上的货物，鼓励快速卸货）
        for vehicle in self.vehicles.values():
            # 车辆利用率奖励:如果两个工位都有货物,给予额外奖励
            occupied_slots = sum(1 for slot in vehicle.slots if slot is not None)
            if occupied_slots == 2:
                reward += 0.5 * LOW_LEVEL_CONTROL_INTERVAL  # 每步+0.5的利用率奖励
            
            for cargo_id in vehicle.slots:
                if cargo_id is not None:
                    cargo = self.cargos[cargo_id]
                    if cargo.picked_up_time is not None:
                        # 计算持有时间
                        holding_time = self.current_time - cargo.picked_up_time
                        # 根据持有时间给予惩罚，鼓励尽快卸货
                        reward -= REWARD_HOLDING_PENALTY_COEFF * holding_time * LOW_LEVEL_CONTROL_INTERVAL
        
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
        
        # 待取货物信息（超时货物优先级更高，排在前面）
        waiting_cargos = []
        for cargo in self.cargos.values():
            if cargo.completion_time is None and cargo.current_location.startswith("IP_"):
                waiting_cargos.append({
                    'id': cargo.id,
                    'wait_time': cargo.wait_time(self.current_time),
                    'is_timeout': cargo.is_timeout(self.current_time),
                    'loading_station': cargo.loading_station,
                    'loading_slot': cargo.loading_slot,
                    'priority': 1 if cargo.is_timeout(self.current_time) else 0  # 超时货物优先级提升
                })
        # 按优先级排序，超时货物优先
        waiting_cargos.sort(key=lambda x: (-x['priority'], -x['wait_time']))
        
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
