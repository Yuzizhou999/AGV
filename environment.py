"""
环形轨道双工位穿梭车调度系统 - 环境模块
实现车辆、货物、上下料口等环境模型
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from config import *

@dataclass
class Cargo:
    """货物对象 - 表示系统中的一件货物"""
    id: int  # 货物唯一ID
    arrival_time: float  # 到达时间（秒）
    loading_station: int  # 在哪个上料工位（0或1）
    loading_slot: int  # 在上料工位的哪个工位 (0: 1#工位, 1: 2#工位)
    allowed_unloading_stations: Set[int]  # 允许的下料口集合（货物可以送到的下料口）
    current_location: str  # 当前位置: "IP_{id}_{slot}"(上料口) 或 "vehicle_{id}_{slot}"(车辆) 或 "OP_{id}_{slot}"(下料口)
    target_unloading_station: Optional[int] = None  # 高层决策指定的目标下料口
    target_slot: Optional[int] = None  # 目标下料口的工位（0或1）
    completion_time: Optional[float] = None  # 完成时间（None表示未完成）
    timed_out: bool = False  # 是否已计入超时（防止重复计算超时惩罚）
    
    def wait_time(self, current_time: float) -> float:
        """计算货物已等待时间（秒）"""
        return current_time - self.arrival_time
    
    def is_timeout(self, current_time: float) -> bool:
        """判断货物是否超时"""
        return self.wait_time(current_time) > CARGO_TIMEOUT


@dataclass
class Vehicle:
    """车辆对象 - 表示环形轨道上的一辆穿梭车"""
    id: int  # 车辆唯一ID
    position: float  # 当前位置（米），范围[0, TRACK_LENGTH)
    velocity: float  # 当前速度（米/秒）
    slots: List[Optional[int]]  # 两个工位，存储货物ID（None表示空）
    slot_operation_end_time: List[float]  # 每个工位的装卸操作剩余时间（秒），0表示空闲
    assigned_task: Optional[Dict] = None  # 高层分配的任务（未对齐时先记录，对齐后执行）
    
    def __post_init__(self):
        """初始化后检查，确保工位数量正确"""
        if len(self.slots) != 2:
            self.slots = [None, None]
        if len(self.slot_operation_end_time) != 2:
            self.slot_operation_end_time = [0.0, 0.0]
    
    def has_empty_slot(self) -> bool:
        """判断是否有空工位"""
        return None in self.slots
    
    def get_empty_slot_idx(self) -> Optional[int]:
        """获取第一个空工位的索引"""
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i
        return None
    
    def distance_to(self, position: float) -> float:
        """计算到目标位置的距离（单向，不考虑环形）"""
        return abs(position - self.position)
    
    def is_aligned_with(self, station_position: float, tolerance: float = None) -> bool:
        """判断是否与工位对齐（在允许误差范围内）"""
        if tolerance is None:
            tolerance = ALIGNMENT_TOLERANCE
        return self.distance_to(station_position) <= tolerance


class LoadingStation:
    """上料工位 - 货物到达并等待车辆取走的地方"""
    def __init__(self, id: int, position: float):
        self.id = id  # 上料工位ID（0或1）
        self.position = position  # 在环形轨道上的位置（米）
        self.slots: List[Optional[int]] = [None, None]  # 两个工位，存储等待取走的货物ID
    
    def has_empty_slot(self) -> bool:
        """判断是否有空工位可放置新货物"""
        return None in self.slots
    
    def place_cargo(self, cargo_id: int, slot_idx: int) -> bool:
        """在指定工位放置货物"""
        if self.slots[slot_idx] is None:
            self.slots[slot_idx] = cargo_id
            return True
        return False
    
    def remove_cargo(self, slot_idx: int) -> Optional[int]:
        """从指定工位移除货物（车辆取走时）"""
        cargo_id = self.slots[slot_idx]
        self.slots[slot_idx] = None
        return cargo_id


class UnloadingStation:
    """下料工位 - 车辆将货物卸下的地方"""
    def __init__(self, id: int, position: float):
        self.id = id  # 下料工位ID（0或1）
        self.position = position  # 在环形轨道上的位置（米）
        self.slots: List[Optional[int]] = [None, None]  # 两个工位，存储已卸下的货物ID
        self.slot_reserved: List[bool] = [False, False]  # 工位预留标记（防止冲突）
    
    def has_empty_slot(self) -> bool:
        """判断是否有可用工位（既非空也未被预留）"""
        return None in self.slots and not all(self.slot_reserved)
    
    def get_available_slot(self) -> Optional[int]:
        """获取第一个可用工位的索引"""
        for i in range(2):
            if self.slots[i] is None and not self.slot_reserved[i]:
                return i
        return None
    
    def place_cargo(self, cargo_id: int, slot_idx: int) -> bool:
        """在指定工位放置货物（车辆卸货完成时）"""
        if self.slots[slot_idx] is None:
            self.slots[slot_idx] = cargo_id
            return True
        return False
    
    def remove_cargo(self, slot_idx: int) -> Optional[int]:
        """从指定工位移除货物（货物被处理后）"""
        cargo_id = self.slots[slot_idx]
        self.slots[slot_idx] = None
        return cargo_id


class Environment:
    """环形轨道双工位穿梭车调度系统 - 仿真环境"""
    def __init__(self, seed: int = None):
        """初始化环境
        
        Args:
            seed: 随机种子，用于可重复实验
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化车辆：均匀分布在轨道上
        self.vehicles: Dict[int, Vehicle] = {}
        for i in range(MAX_VEHICLES):
            self.vehicles[i] = Vehicle(
                id=i,
                position=float(i * TRACK_LENGTH / MAX_VEHICLES),  # 均匀分布
                velocity=0.0,
                slots=[None, None],
                slot_operation_end_time=[0.0, 0.0]
            )
        
        # 初始化上料工位
        self.loading_stations: Dict[int, LoadingStation] = {}
        for i in range(NUM_LOADING_STATIONS):
            self.loading_stations[i] = LoadingStation(i, LOADING_POSITIONS[i])
        
        # 初始化下料工位
        self.unloading_stations: Dict[int, UnloadingStation] = {}
        for i in range(NUM_UNLOADING_STATIONS):
            self.unloading_stations[i] = UnloadingStation(i, UNLOADING_POSITIONS[i])
        
        # 货物管理
        self.cargos: Dict[int, Cargo] = {}  # 所有货物（包括进行中和已完成）
        self.cargo_counter = 0  # 货物ID计数器
        
        # 时间管理
        self.current_time = 0.0  # 当前仿真时间（秒）
        self.next_arrival_time = np.random.uniform(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX)  # 下一件货物到达时间
        
        # 统计信息
        self.completed_cargos = 0  # 已完成货物数
        self.timed_out_cargos = 0  # 已超时货物数
        self.total_wait_time = 0.0  # 总等待时间（用于计算平均等待时间）
        self.timeout_pickups = []  # 本步骤取走的超时货物列表（用于额外奖励）
    
    def reset(self):
        self.__init__(seed=None)
        return self._get_observation()
    
    def _normalize_position(self, pos: float) -> float:
        return pos % TRACK_LENGTH
    
    def _check_and_generate_cargo(self) -> List[int]:
        new_cargo_ids = []
        while self.current_time >= self.next_arrival_time:
            loading_station_id = np.random.randint(0, NUM_LOADING_STATIONS)
            slot_idx = np.random.randint(0, 2)
            station = self.loading_stations[loading_station_id]
            if station.slots[slot_idx] is None:
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
            self.next_arrival_time += np.random.uniform(ARRIVAL_INTERVAL_MIN, ARRIVAL_INTERVAL_MAX)
        return new_cargo_ids
    
    def step(self, high_level_actions: List[Dict], low_level_actions: Dict) -> Tuple[Dict, float, bool]:
        """执行一步仿真（时长为LOW_LEVEL_CONTROL_INTERVAL）
        
        执行顺序：
        1. 时间推进
        2. 检查并生成新货物
        3. 规则控制车辆移动
        4. 执行高层决策（分配任务、取货）
        5. 检查卸货完成情况
        6. 检查货物超时
        7. 计算奖励
        
        Args:
            high_level_actions: 高层决策动作列表
            low_level_actions: 低层控制动作（已废弃，使用规则控制）
            
        Returns:
            (observation, reward, done): 观测、奖励、是否结束
        """
        self.current_time += LOW_LEVEL_CONTROL_INTERVAL
        done = self.current_time >= EPISODE_DURATION
        
        self._check_and_generate_cargo()  # 生成新货物
        self._move_vehicles_rule_based()  # 规则控制车辆移动
        pickup_ids = self._execute_high_level_actions(high_level_actions)  # 执行取货任务
        completed_ids = self._check_completions()  # 检查卸货完成
        newly_timed_out = self._check_timeouts()  # 检查新超时货物
        reward = self._calculate_reward(completed_ids, pickup_ids, newly_timed_out)  # 计算奖励
        obs = self._get_observation()  # 获取观测
        return obs, reward, done
    
    def _move_vehicles_rule_based(self):
        """使用规则控制所有车辆的移动
        
        规则优先级：
        1. 如果正在装卸货，停车等待，更新操作时间
        2. 如果有已分配任务（assigned_task），前往取货
        3. 如果车上有货物，前往目标下料口
        4. 如果车辆有空位，前往最近的有货物的上料工位
        5. 否则减速停车
        
        运动控制：
        - 使用P控制：根据距离设定目标速度
        - 考虑加速度限制
        - 安全距离检查
        """
        for vehicle_id, vehicle in self.vehicles.items():
            # 步骤1: 检查是否正在装卸货
            is_loading_or_unloading = any(t > 0 for t in vehicle.slot_operation_end_time)
            if is_loading_or_unloading:
                vehicle.velocity = 0.0  # 装卸时停车
                # 更新操作时间
                for i in range(2):
                    if vehicle.slot_operation_end_time[i] > 0:
                        vehicle.slot_operation_end_time[i] -= LOW_LEVEL_CONTROL_INTERVAL
                        if vehicle.slot_operation_end_time[i] < 0:
                            vehicle.slot_operation_end_time[i] = 0
                continue
            
            # 步骤2: 确定目标位置（按优先级）
            target_pos = None
            
            # 优先级1: 执行已分配的任务（前往取货）
            if vehicle.assigned_task is not None:
                cargo_id = vehicle.assigned_task.get('cargo_id')
                cargo = self.cargos.get(cargo_id)
                if cargo and cargo.current_location.startswith('IP_'):  # 货物还在上料口
                    loading_station = self.loading_stations.get(cargo.loading_station)
                    if loading_station:
                        target_pos = loading_station.position
            
            # 优先级2: 如果车上有货物，前往目标下料口
            if target_pos is None:
                for slot_idx, cargo_id in enumerate(vehicle.slots):
                    if cargo_id is not None:
                        cargo = self.cargos.get(cargo_id)
                        if cargo and cargo.target_unloading_station is not None:
                            unloading_station = self.unloading_stations[cargo.target_unloading_station]
                            target_pos = unloading_station.position
                            break
            
            # 优先级3: 如果有空位，前往最近的有货物的上料工位
            if target_pos is None and vehicle.has_empty_slot():
                min_distance = float('inf')
                for station in self.loading_stations.values():
                    if any(slot is not None for slot in station.slots):  # 工位有货物等待
                        distance = vehicle.distance_to(station.position)
                        if distance < min_distance:
                            min_distance = distance
                            target_pos = station.position
            
            # 优先级4: 没有目标，减速停车
            if target_pos is None:
                new_velocity = max(0.0, vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
                vehicle.velocity = new_velocity
                displacement = vehicle.velocity * LOW_LEVEL_CONTROL_INTERVAL
                vehicle.position = self._normalize_position(vehicle.position + displacement)
                continue
            
            # 步骤3: 根据距离计算目标速度（P控制）
            distance = vehicle.distance_to(target_pos)
            if distance > ALIGNMENT_TOLERANCE * 3:
                # 距离较远：全速前进
                target_velocity = MAX_SPEED
            elif distance > ALIGNMENT_TOLERANCE:
                # 距离适中：线性减速
                target_velocity = MAX_SPEED * (distance / (ALIGNMENT_TOLERANCE * 3))
            else:
                # 已接近目标：停车
                target_velocity = 0.0
            
            # 步骤4: 考虑加速度限制
            velocity_diff = target_velocity - vehicle.velocity
            if abs(velocity_diff) > MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL:
                # 超过加速度限制，分步调整
                if velocity_diff > 0:
                    new_velocity = vehicle.velocity + MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL
                else:
                    new_velocity = vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL
            else:
                # 可直接达到目标速度
                new_velocity = target_velocity
            
            # 步骤5: 安全距离检查
            if not self._check_safety_distance(vehicle_id, new_velocity):
                new_velocity = 0.0  # 与其他车辆距离过近，停车
            
            # 步骤6: 更新速度和位置
            vehicle.velocity = max(0.0, min(MAX_SPEED, new_velocity))
            displacement = vehicle.velocity * LOW_LEVEL_CONTROL_INTERVAL
            vehicle.position = self._normalize_position(vehicle.position + displacement)
    
    def _execute_low_level_control(self, actions: Dict):
        pass
    
    def _check_safety_distance(self, vehicle_id: int, new_velocity: float) -> bool:
        vehicle = self.vehicles[vehicle_id]
        new_position = self._normalize_position(vehicle.position + new_velocity * LOW_LEVEL_CONTROL_INTERVAL)
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
            if new_position < other_vehicle.position:
                distance = other_vehicle.position - new_position
            else:
                distance = TRACK_LENGTH - new_position + other_vehicle.position
            if distance < SAFETY_DISTANCE:
                return False
        return True
    
    def _execute_high_level_actions(self, actions: List[Dict]) -> List[int]:
        """执行高层决策动作
        
        高层决策包括：
        1. 为货物分配车辆和目标下料口
        2. 如果车辆已对齐，立即取货
        3. 如果车辆未对齐，将任务记录在vehicle.assigned_task中，规则控制会引导车辆前往
        
        Args:
            actions: 高层动作列表，每个动作包含货物ID、车辆ID、车辆工位、下料口ID和工位
            
        Returns:
            pickup_ids: 本步骤成功取走的货物ID列表
        """
        pickup_ids: List[int] = []
        if not actions:
            return pickup_ids
        
        for action in actions:
            if action.get('type') != 'assign_loading_with_target':
                continue
            
            # 提取动作参数
            cargo_id = action.get('cargo_id')
            vehicle_id = action.get('vehicle_id')
            vehicle_slot_idx = action.get('vehicle_slot_idx')
            unloading_station_id = action.get('unloading_station_id')
            unloading_slot_idx = action.get('unloading_slot_idx')
            
            # 验证动作有效性
            if (cargo_id in self.cargos and vehicle_id in self.vehicles and
                self.vehicles[vehicle_id].slots[vehicle_slot_idx] is None and
                self.vehicles[vehicle_id].slot_operation_end_time[vehicle_slot_idx] == 0.0):
                
                cargo = self.cargos[cargo_id]
                loading_station = self.loading_stations.get(cargo.loading_station)
                vehicle = self.vehicles[vehicle_id]
                
                if loading_station and loading_station.slots[cargo.loading_slot] == cargo_id:
                    if vehicle.is_aligned_with(loading_station.position):
                        # 车辆已对齐：立即取货
                        # 先设置目标下料口
                        if unloading_station_id in cargo.allowed_unloading_stations:
                            unloading_station = self.unloading_stations[unloading_station_id]
                            if not unloading_station.slot_reserved[unloading_slot_idx]:
                                cargo.target_unloading_station = unloading_station_id
                                cargo.target_slot = unloading_slot_idx
                                unloading_station.slot_reserved[unloading_slot_idx] = True
                        # 执行取货
                        self._assign_loading_task(cargo_id, vehicle_id, vehicle_slot_idx)
                        pickup_ids.append(cargo_id)
                        vehicle.assigned_task = None  # 清除任务
                    else:
                        # 车辆未对齐：记录任务，规则控制会引导车辆前往
                        vehicle.assigned_task = action
        return pickup_ids
    
    def _assign_loading_task(self, cargo_id: int, vehicle_id: int, slot_idx: int):
        """执行装货操作（车辆已对齐上料口）
        
        执行步骤：
        1. 记录超时取货（用于额外奖励）
        2. 从上料口移除货物
        3. 将货物放入车辆工位
        4. 设置装货操作时间
        5. 更新货物位置
        """
        cargo = self.cargos[cargo_id]
        vehicle = self.vehicles[vehicle_id]
        loading_station = self.loading_stations[cargo.loading_station]
        
        # 记录超时取货（用于额外奖励）
        if not hasattr(self, 'timeout_pickups'):
            self.timeout_pickups = []
        if cargo.is_timeout(self.current_time):
            self.timeout_pickups.append(cargo_id)
        
        # 从上料口移除货物
        loading_station.slots[cargo.loading_slot] = None
        # 将货物放入车辆
        vehicle.slots[slot_idx] = cargo_id
        # 设置装货时间（车辆在此期间停车）
        vehicle.slot_operation_end_time[slot_idx] = LOADING_TIME
        # 更新货物位置
        cargo.current_location = f"vehicle_{vehicle_id}_{slot_idx}"
    
    def _check_completions(self) -> List[int]:
        """检查卸货完成情况
        
        处理逻辑：
        1. 检查是否可以同时卸两个工位（同一下料口，两个工位都有货）
        2. 如果不能同时卸，逐个检查每个工位
        3. 车辆对齐下料口 + 操作时间结束 = 开始卸货操作
        4. 卸货完成后，货物从车辆转移到下料口，记为完成
        
        Returns:
            completed_ids: 本步骤完成卸货的货物ID列表
        """
        completed_ids: List[int] = []
        for vehicle_id, vehicle in self.vehicles.items():
            # 检查是否可以同时卸两个工位
            both_slots_ready = self._check_simultaneous_operation(vehicle_id, vehicle)
            if both_slots_ready:
                # 可以同时卸货：启动两个工位的卸货操作
                station_id = both_slots_ready['station_id']
                unloading_station = self.unloading_stations[station_id]
                for slot_info in both_slots_ready['slots']:
                    slot_idx = slot_info['slot_idx']
                    cargo_id = slot_info['cargo_id']
                    cargo = self.cargos[cargo_id]
                    if vehicle.slot_operation_end_time[slot_idx] == 0:
                        # 启动卸货操作
                        vehicle.slot_operation_end_time[slot_idx] = UNLOADING_TIME
                        unloading_station.slot_reserved[cargo.target_slot] = True
            # 逐个检查每个工位的卸货情况
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is None:
                    continue
                cargo = self.cargos[cargo_id]
                if cargo.target_unloading_station is None:
                    continue  # 货物未分配目标下料口
                
                unloading_station = self.unloading_stations[cargo.target_unloading_station]
                if not vehicle.is_aligned_with(unloading_station.position):
                    continue  # 车辆未对齐下料口
                
                current_op_time = vehicle.slot_operation_end_time[slot_idx]
                if current_op_time > 0.1:
                    continue  # 正在卸货，等待
                elif current_op_time > -0.1:  # 接近0，可以启动或完成卸货
                    if not unloading_station.slot_reserved[cargo.target_slot]:
                        # 目标工位未预留：启动卸货操作
                        if both_slots_ready and slot_idx in [s['slot_idx'] for s in both_slots_ready.get('slots', [])]:
                            continue  # 已在同时卸货中处理
                        vehicle.slot_operation_end_time[slot_idx] = UNLOADING_TIME
                        unloading_station.slot_reserved[cargo.target_slot] = True
                        continue
                    else:
                        # 卸货操作完成：货物从车辆转移到下料口
                        vehicle.slots[slot_idx] = None
                        unloading_station.slots[cargo.target_slot] = cargo_id
                        unloading_station.slot_reserved[cargo.target_slot] = False
                        cargo.current_location = f"OP_{cargo.target_unloading_station}_{cargo.target_slot}"
                        cargo.completion_time = self.current_time
                        # 统计信息
                        self.total_wait_time += cargo.wait_time(self.current_time)
                        completed_ids.append(cargo_id)
                        self.completed_cargos += 1
        return completed_ids
    
    def _check_simultaneous_operation(self, vehicle_id: int, vehicle) -> Optional[Dict]:
        if vehicle.slots[0] is None or vehicle.slots[1] is None:
            return None
        cargo_0 = self.cargos.get(vehicle.slots[0])
        cargo_1 = self.cargos.get(vehicle.slots[1])
        if not cargo_0 or not cargo_1:
            return None
        if cargo_0.target_unloading_station is None or cargo_1.target_unloading_station is None:
            return None
        if cargo_0.target_unloading_station != cargo_1.target_unloading_station:
            return None
        station_id = cargo_0.target_unloading_station
        unloading_station = self.unloading_stations[station_id]
        if not vehicle.is_aligned_with(unloading_station.position):
            return None
        if vehicle.slot_operation_end_time[0] > 0 or vehicle.slot_operation_end_time[1] > 0:
            return None
        return {
            'station_id': station_id,
            'slots': [
                {'slot_idx': 0, 'cargo_id': vehicle.slots[0]},
                {'slot_idx': 1, 'cargo_id': vehicle.slots[1]}
            ]
        }
    
    def _check_timeouts(self) -> List[int]:
        """检查货物超时情况
        
        只标记新超时的货物（防止重复计算）
        
        Returns:
            newly_timed_out: 本步骤新超时的货物ID列表
        """
        newly_timed_out: List[int] = []
        for cargo in self.cargos.values():
            if cargo.completion_time is not None or cargo.timed_out:
                continue  # 已完成或已标记超时
            if cargo.is_timeout(self.current_time):
                cargo.timed_out = True
                self.timed_out_cargos += 1
                newly_timed_out.append(cargo.id)
        return newly_timed_out
    
    def _calculate_reward(self, completed_ids: List[int], pickup_ids: List[int], timed_out_ids: List[int]) -> float:
        """计算步骤奖励
        
        奖励组成：
        1. 交付奖励：每完成一件货物 +REWARD_DELIVERY（正奖励）
        2. 取货奖励：每取走一件货物 +REWARD_PICKUP（正奖励）
        3. 超时惩罚：每新超时一件货物 +REWARD_TIMEOUT_PENALTY（负值，一次性）
        4. 超时取货奖励：取走超时货物 +REWARD_TIMEOUT_PICKUP（正奖励，鼓励及时处理）
        5. 等待惩罚：所有未完成且未超时货物按等待时间累积（封顶CARGO_TIMEOUT）
        
        Args:
            completed_ids: 本步骤完成的货物ID列表
            pickup_ids: 本步骤取走的货物ID列表
            timed_out_ids: 本步骤新超时的货物ID列表
            
        Returns:
            reward: 总奖励值
        """
        reward = 0.0
        
        # 1. 交付奖励（完成一件货物）
        reward += len(completed_ids) * REWARD_DELIVERY
        
        # 2. 取货奖励（取走一件货物）
        reward += len(pickup_ids) * REWARD_PICKUP
        
        # 3. 超时惩罚（新超时货物，一次性）
        if timed_out_ids:
            reward += len(timed_out_ids) * REWARD_TIMEOUT_PENALTY  # REWARD_TIMEOUT_PENALTY是负值
        
        # 4. 超时取货奖励（鼓励及时处理超时货物）
        if hasattr(self, 'timeout_pickups') and self.timeout_pickups:
            reward += len(self.timeout_pickups) * REWARD_TIMEOUT_PICKUP
            self.timeout_pickups = []  # 清空列表
        
        # 5. 等待惩罚（所有未完成且未超时的货物）
        for cargo in self.cargos.values():
            if cargo.completion_time is None and not cargo.timed_out:
                # 等待时间封顶，避免超时后继续累积大额惩罚
                wait_time = min(cargo.wait_time(self.current_time), CARGO_TIMEOUT)
                reward -= REWARD_WAIT_PENALTY_COEFF * wait_time * LOW_LEVEL_CONTROL_INTERVAL
        
        return reward
    
    def _get_observation(self) -> Dict:
        vehicle_obs = []
        for vehicle in self.vehicles.values():
            vehicle_obs.append({
                'position': vehicle.position / TRACK_LENGTH,
                'velocity': vehicle.velocity / MAX_SPEED,
                'slots': vehicle.slots.copy(),
                'slot_occupied': [slot is not None for slot in vehicle.slots]
            })
        loading_obs = []
        for station in self.loading_stations.values():
            station_obs = {
                'position': station.position / TRACK_LENGTH,
                'slots': station.slots.copy(),
                'slot_occupied': [slot is not None for slot in station.slots]
            }
            loading_obs.append(station_obs)
        unloading_obs = []
        for station in self.unloading_stations.values():
            station_obs = {
                'position': station.position / TRACK_LENGTH,
                'slots': station.slots.copy(),
                'slot_occupied': [slot is not None for slot in station.slots]
            }
            unloading_obs.append(station_obs)
        waiting_cargos = []
        for cargo in self.cargos.values():
            if cargo.completion_time is None and cargo.current_location.startswith("IP_"):
                waiting_cargos.append({
                    'id': cargo.id,
                    'wait_time': cargo.wait_time(self.current_time),
                    'is_timeout': cargo.is_timeout(self.current_time)
                })
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
        obs_list = []
        for vehicle in self.vehicles.values():
            obs_list.extend([
                vehicle.position / TRACK_LENGTH,
                vehicle.velocity / MAX_SPEED,
                float(vehicle.slots[0] is not None),
                float(vehicle.slots[1] is not None)
            ])
        for station in self.loading_stations.values():
            obs_list.extend([
                station.position / TRACK_LENGTH,
                float(station.slots[0] is not None),
                float(station.slots[1] is not None)
            ])
        waiting_count = sum(1 for c in self.cargos.values() 
                          if c.completion_time is None and c.current_location.startswith("IP_"))
        obs_list.extend([
            self.current_time / EPISODE_DURATION,
            waiting_count / max(10, self.cargo_counter)
        ])
        return np.array(obs_list, dtype=np.float32)
    
    def get_low_level_observation(self, vehicle_id: int) -> np.ndarray:
        vehicle = self.vehicles[vehicle_id]
        obs_list = [
            vehicle.position / TRACK_LENGTH,
            vehicle.velocity / MAX_SPEED
        ]
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
            if other_vehicle.position > vehicle.position:
                distance = other_vehicle.position - vehicle.position
            else:
                distance = TRACK_LENGTH - vehicle.position + other_vehicle.position
            obs_list.append(distance / TRACK_LENGTH)
        target_found = False
        is_aligned = 0.0
        for cargo_id in vehicle.slots:
            if cargo_id is not None:
                cargo = self.cargos[cargo_id]
                if cargo.target_unloading_station is not None:
                    target_pos = self.unloading_stations[cargo.target_unloading_station].position
                    target_distance = vehicle.distance_to(target_pos)
                    obs_list.append(target_distance / TRACK_LENGTH)
                    is_aligned = 1.0 if vehicle.is_aligned_with(target_pos) else 0.0
                    target_found = True
                    break
        if not target_found:
            obs_list.append(0.0)
        obs_list.append(is_aligned)
        return np.array(obs_list, dtype=np.float32)