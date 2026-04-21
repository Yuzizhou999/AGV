"""
环形轨道双工位穿梭车调度系统 - 环境模块
实现车辆、货物、上下料口等环境模型
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from config import *


@dataclass
class Cargo:
    """货物对象"""
    id: int  # 货物唯一ID
    arrival_time: float  # 到达时间
    loading_station: int  # 在哪个上料工位
    loading_slot: int  # 在上料工位的哪个工位 (0: 1#, 1: 2#)
    allowed_unloading_stations: Set[int]  # 允许的下料口集合
    target_unloading_station: Optional[int] = None  # 目标下料口
    completion_time: Optional[float] = None  # 完成时间
    assigned_vehicle: Optional[int] = None  # 分配的车辆ID（用于上料任务）
    assigned_vehicle_slot: Optional[int] = None  # 分配的车辆工位
    loading_start_time: Optional[float] = None  # 上料开始时间
    unloading_start_time: Optional[float] = None  # 下料开始时间
    picked_up_time: Optional[float] = None  # 被取走的时间（小车开始取货的时间，用于超时判断）
    paper_type: str = ROLL_TYPE_SMALL
    diameter_m: float = SMALL_ROLL_DIAMETER_LIMIT
    required_slots: int = 1
    primary_unloading_station: Optional[int] = None
    alternative_unloading_stations: Tuple[int, ...] = field(default_factory=tuple)
    assigned_vehicle_slots: Tuple[int, ...] = field(default_factory=tuple)
    
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
    is_loading_unloading: bool = False  # 是否正在进行上料/下料操作（锁定移动）
    last_applied_acceleration: float = 0.0
    last_nonzero_accel_sign: int = 0
    steps_since_direction_change: int = MIN_DIRECTION_CHANGE_INTERVAL_STEPS
    
    def __post_init__(self):
        if len(self.slots) != VEHICLE_SLOT_COUNT:
            self.slots = [None] * VEHICLE_SLOT_COUNT
    
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
    def __init__(
        self,
        id: int,
        position: float,
        paper_type: str,
        side: str,
        primary_unloading_station: int,
        alternative_unloading_stations: List[int],
    ):
        self.id = id
        self.position = position
        self.paper_type = paper_type
        self.side = side
        self.primary_unloading_station = primary_unloading_station
        self.alternative_unloading_stations = tuple(alternative_unloading_stations)
        self.slots: List[Optional[int]] = [None] * LOADING_STATION_SLOTS
    
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
    def __init__(self, id: int, position: float, side: str):
        self.id = id
        self.position = position
        self.side = side
        # 下料口按需求假定“无限接收”：卸货完成后不占用下料口工位。
        # 因此不建模 slots/reserved 等容量状态，只保留位置用于对齐判断。


class Environment:
    """环形轨道双工位穿梭车调度环境"""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        # 车辆
        self.vehicles: Dict[int, Vehicle] = {}
        for vehicle_cfg in VEHICLE_CONFIGS:
            vehicle_id = int(vehicle_cfg["id"])
            self.vehicles[vehicle_id] = Vehicle(
                id=vehicle_id,
                position=float(vehicle_cfg["initial_position"]),
                velocity=0.0,
                slots=[None] * VEHICLE_SLOT_COUNT,
                is_loading_unloading=False
            )
        
        # 上料口
        self.loading_stations: Dict[int, LoadingStation] = {}
        for station_cfg in LOADING_STATION_CONFIGS:
            station_id = int(station_cfg["id"])
            self.loading_stations[station_id] = LoadingStation(
                id=station_id,
                position=float(station_cfg["position"]),
                paper_type=str(station_cfg["paper_type"]),
                side=str(station_cfg.get("side", "left")),
                primary_unloading_station=int(station_cfg["primary_unloading_station"]),
                alternative_unloading_stations=list(station_cfg.get("alternative_unloading_stations", [])),
            )
        
        # 下料口
        self.unloading_stations: Dict[int, UnloadingStation] = {}
        for station_cfg in UNLOADING_STATION_CONFIGS:
            station_id = int(station_cfg["id"])
            self.unloading_stations[station_id] = UnloadingStation(
                id=station_id,
                position=float(station_cfg["position"]),
                side=str(station_cfg.get("side", "right")),
            )
        
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
        self.safety_violations = []  # 记录本次step的安全违例车辆ID
        self.action_switch_violations = []  # 记录被钳制的反向加减速动作
        self.alerts = []
        self.safety_warning_count = 0
        self.collision_alert_count = 0

    def is_cargo_at_loading_station(self, cargo: Cargo) -> bool:
        """判断货物是否仍在其上料口工位（单一真相：以站点/车辆slots为准）。"""
        station = self.loading_stations.get(cargo.loading_station)
        if station is None:
            return False
        if cargo.loading_slot < 0 or cargo.loading_slot >= LOADING_STATION_SLOTS:
            return False
        return station.slots[cargo.loading_slot] == cargo.id

    def is_cargo_on_vehicle(self, cargo: Cargo) -> bool:
        """判断货物是否已装载在某辆车的某个工位（单一真相：以车辆slots为准）。"""
        # 优先检查分配车辆（通常就是装载车辆），避免全量扫描。
        if cargo.assigned_vehicle is not None:
            vehicle = self.vehicles.get(cargo.assigned_vehicle)
            if vehicle is not None and cargo.id in vehicle.slots:
                return True
        # 兜底：扫描所有车辆，避免异常状态下误判。
        for vehicle in self.vehicles.values():
            if cargo.id in vehicle.slots:
                return True
        return False

    def _assigned_slot_indices(self, cargo: Cargo) -> Tuple[int, ...]:
        """返回货物被分配到车辆上的工位集合。"""
        if cargo.assigned_vehicle_slots:
            return tuple(int(slot_idx) for slot_idx in cargo.assigned_vehicle_slots)
        if cargo.assigned_vehicle_slot is not None:
            return (int(cargo.assigned_vehicle_slot),)
        return ()

    def _vehicle_unique_cargo_ids(self, vehicle: Vehicle) -> List[int]:
        """返回车辆上不重复的货物 ID。"""
        seen = set()
        cargo_ids: List[int] = []
        for cargo_id in vehicle.slots:
            if cargo_id is None or cargo_id in seen:
                continue
            seen.add(cargo_id)
            cargo_ids.append(cargo_id)
        return cargo_ids

    def _vehicle_available_slot_groups(self, vehicle_id: int, cargo: Cargo) -> List[Tuple[int, ...]]:
        """返回车辆对某个货物可行的工位分配方案。"""
        vehicle = self.vehicles[vehicle_id]
        empty_slots = [
            slot_idx
            for slot_idx, slot in enumerate(vehicle.slots)
            if slot is None and not self._is_vehicle_slot_reserved(vehicle_id, slot_idx)
        ]

        if cargo.required_slots <= 1:
            return [(slot_idx,) for slot_idx in empty_slots]

        # 大卷纸 / 超径小卷纸一律占满整车，不允许和其他卷纸混装。
        if len(empty_slots) == len(vehicle.slots):
            return [tuple(empty_slots)]
        return []

    def vehicle_can_accept_cargo(self, vehicle_id: int, cargo: Cargo) -> bool:
        """判断车辆是否还能装下指定货物。"""
        if cargo.assigned_vehicle == vehicle_id:
            return True
        return bool(self._vehicle_available_slot_groups(vehicle_id, cargo))

    def _required_slots_for_station(self, station: LoadingStation, diameter_m: float) -> int:
        """根据上料口类型与卷径计算占用工位数。"""
        if station.paper_type == ROLL_TYPE_LARGE:
            return VEHICLE_SLOT_COUNT
        if diameter_m > SMALL_ROLL_DIAMETER_LIMIT:
            return VEHICLE_SLOT_COUNT
        return 1

    def has_vehicle_task(self, vehicle_id: int) -> bool:
        """判断车辆是否存在“待完成的任务”。

        单一真相口径：
        - 不依赖 vehicle.assigned_tasks（已移除），而是从 cargo/slots 的真实状态推导。
        - 任务定义：需要去某个上料口取货，或需要去某个下料口卸货。
        """
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return False

        # 1) 卸货任务：车上有货物，且已分配下料口
        for cargo_id in self._vehicle_unique_cargo_ids(vehicle):
            if cargo_id is None or cargo_id not in self.cargos:
                continue
            cargo = self.cargos[cargo_id]
            if cargo.target_unloading_station is not None:
                return True

        # 2) 取货任务：存在分配给该车且仍在上料口等待的货物
        for cargo in self.cargos.values():
            if cargo.completion_time is not None:
                continue
            if cargo.assigned_vehicle != vehicle_id:
                continue
            if not self.is_cargo_at_loading_station(cargo):
                continue
            slot_indices = self._assigned_slot_indices(cargo)
            if slot_indices and all(vehicle.slots[slot_idx] is None for slot_idx in slot_indices):
                return True

        return False

    def get_vehicle_target_position(self, vehicle_id: int) -> Optional[float]:
        """推导车辆当前应该前往的目标位置（顺时针距离最短）。"""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return None

        # 优先：卸货（已有货物先送达，减少持有时间）
        best_position = None
        best_distance = float('inf')
        for cargo_id in self._vehicle_unique_cargo_ids(vehicle):
            if cargo_id is None or cargo_id not in self.cargos:
                continue
            cargo = self.cargos[cargo_id]
            if cargo.target_unloading_station is None:
                continue
            station = self.unloading_stations.get(cargo.target_unloading_station)
            if station is None:
                continue
            dist = vehicle.distance_to(station.position)
            if dist < best_distance:
                best_distance = dist
                best_position = station.position
        if best_position is not None:
            return best_position

        # 其次：取货（已分配且仍在上料口等待的货物）
        for cargo in self.cargos.values():
            if cargo.completion_time is not None:
                continue
            if cargo.assigned_vehicle != vehicle_id:
                continue
            if not self.is_cargo_at_loading_station(cargo):
                continue
            slot_indices = self._assigned_slot_indices(cargo)
            if not slot_indices:
                continue
            if any(vehicle.slots[slot_idx] is not None for slot_idx in slot_indices):
                continue
            station = self.loading_stations.get(cargo.loading_station)
            if station is None:
                continue
            dist = vehicle.distance_to(station.position)
            if dist < best_distance:
                best_distance = dist
                best_position = station.position

        return best_position
    
    def reset(self, seed: int = None):
        """重置环境，支持可选seed

        Args:
            seed: 随机种子，None时使用随机值

        Returns:
            初始观测
        """
        if seed is None:
            seed = np.random.randint(0, 100000)
        self.__init__(seed=seed)
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
            slot_idx = np.random.randint(0, LOADING_STATION_SLOTS)
            
            # 检查工位是否空闲
            station = self.loading_stations[loading_station_id]
            if station.slots[slot_idx] is None:
                if station.paper_type == ROLL_TYPE_LARGE:
                    paper_type = ROLL_TYPE_LARGE
                    diameter_m = float(np.random.uniform(*LARGE_ROLL_DIAMETER_RANGE))
                else:
                    paper_type = ROLL_TYPE_SMALL
                    diameter_m = float(np.random.uniform(*SMALL_ROLL_DIAMETER_RANGE))

                required_slots = self._required_slots_for_station(station, diameter_m)
                alternative_stations = tuple(int(station_id) for station_id in station.alternative_unloading_stations)
                allowed_stations = {
                    int(station.primary_unloading_station),
                    *alternative_stations,
                }
                
                cargo = Cargo(
                    id=self.cargo_counter,
                    arrival_time=self.current_time,
                    loading_station=loading_station_id,
                    loading_slot=slot_idx,
                    allowed_unloading_stations=allowed_stations,
                    paper_type=paper_type,
                    diameter_m=diameter_m,
                    required_slots=required_slots,
                    primary_unloading_station=int(station.primary_unloading_station),
                    alternative_unloading_stations=alternative_stations,
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
        self._update_proximity_alerts()

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
        """
        执行低层控制：更新车辆位置和速度

        支持两种动作格式：
        1. 离散动作（int）: 0=减速, 1=保持, 2=加速（用于启发式和DQN）
        2. 连续动作（float）: 加速度值 [-MAX_ACCELERATION, MAX_ACCELERATION]（用于PPO）
        """
        # 清空上一次的违例记录
        self.safety_violations = []
        self.action_switch_violations = []

        for vehicle_id, action in actions.items():
            vehicle = self.vehicles[vehicle_id]

            # 如果车辆正在进行上料/下料操作，强制锁定不移动
            if vehicle.is_loading_unloading:
                vehicle.velocity = 0.0  # 强制停止
                vehicle.last_applied_acceleration = 0.0
                # 不更新位置，直接跳过
                continue

            # 判断动作类型并计算加速度
            if isinstance(action, (int, np.integer)):
                # 离散动作: 0=减速, 1=保持, 2=加速
                if action == 0:
                    acceleration = -MAX_ACCELERATION
                elif action == 1:
                    acceleration = 0.0
                else:  # action == 2
                    acceleration = MAX_ACCELERATION
            else:
                # 连续动作: 直接使用加速度值
                acceleration = float(action)
                # 限制加速度范围
                acceleration = np.clip(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION)

            acceleration = self._apply_acceleration_smoothing(vehicle_id, acceleration)

            old_velocity = vehicle.velocity
            acceleration = self._project_acceleration_for_safety(vehicle_id, old_velocity, acceleration)

            # 计算新速度
            # 轨道坐标定义为沿顺时针方向，速度不允许为负（不支持反向行驶）
            new_velocity = old_velocity + acceleration * LOW_LEVEL_CONTROL_INTERVAL
            new_velocity = np.clip(new_velocity, 0.0, MAX_SPEED)

            # 更新位置（使用平均速度计算位移，物理精确）
            vehicle.velocity = new_velocity
            vehicle.last_applied_acceleration = (
                (new_velocity - old_velocity) / LOW_LEVEL_CONTROL_INTERVAL
            )
            displacement = (old_velocity + new_velocity) / 2 * LOW_LEVEL_CONTROL_INTERVAL
            vehicle.position = self._normalize_position(vehicle.position + displacement)

    def _acceleration_sign(self, acceleration: float) -> int:
        if acceleration > ACTION_DIRECTION_DEADBAND:
            return 1
        if acceleration < -ACTION_DIRECTION_DEADBAND:
            return -1
        return 0

    def _apply_acceleration_smoothing(self, vehicle_id: int, acceleration: float) -> float:
        """限制加速度变化率，避免瞬时猛冲猛刹。"""
        vehicle = self.vehicles[vehicle_id]
        prev_acceleration = vehicle.last_applied_acceleration
        max_delta = MAX_JERK * LOW_LEVEL_CONTROL_INTERVAL
        smoothed_acceleration = np.clip(
            acceleration,
            prev_acceleration - max_delta,
            prev_acceleration + max_delta
        )

        if abs(smoothed_acceleration - acceleration) > ACTION_DIRECTION_DEADBAND:
            self.action_switch_violations.append(vehicle_id)

        requested_sign = self._acceleration_sign(smoothed_acceleration)
        if requested_sign != 0 and requested_sign != vehicle.last_nonzero_accel_sign:
            vehicle.last_nonzero_accel_sign = requested_sign
            vehicle.steps_since_direction_change = 0
        else:
            vehicle.steps_since_direction_change += 1

        return float(np.clip(smoothed_acceleration, -MAX_ACCELERATION, MAX_ACCELERATION))
    
    def _project_acceleration_for_safety(
        self,
        vehicle_id: int,
        old_velocity: float,
        acceleration: float
    ) -> float:
        """将期望加速度投影到满足前向安全距离的可执行范围内。"""
        vehicle = self.vehicles[vehicle_id]
        dt = LOW_LEVEL_CONTROL_INTERVAL
        max_safe_acceleration = MAX_ACCELERATION

        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue

            headway = self._forward_distance(vehicle.position, other_vehicle.position)
            free_distance = headway - SAFETY_DISTANCE

            # 位移约束：v*dt + 0.5*a*dt^2 <= free_distance。
            safe_acceleration_bound = (
                2.0 * (free_distance - old_velocity * dt) / (dt * dt)
            )
            max_safe_acceleration = min(max_safe_acceleration, safe_acceleration_bound)

        projected_acceleration = min(acceleration, max_safe_acceleration)
        projected_acceleration = np.clip(
            projected_acceleration,
            -MAX_ACCELERATION,
            MAX_ACCELERATION
        )

        if projected_acceleration < acceleration - SAFETY_PROJECTION_EPS:
            self.safety_violations.append(vehicle_id)

        return float(projected_acceleration)
    
    def _forward_distance(self, from_pos: float, to_pos: float) -> float:
        """计算从 from_pos 到 to_pos 的顺时针距离（环形轨道）
        
        Args:
            from_pos: 起始位置
            to_pos: 目标位置
        
        Returns:
            float: 顺时针方向的距离
        """
        if to_pos >= from_pos:
            return to_pos - from_pos
        else:
            return TRACK_LENGTH - from_pos + to_pos

    def _circular_distance(self, pos_a: float, pos_b: float) -> float:
        """计算环形轨道上两点之间的最小圆周距离"""
        forward = self._forward_distance(pos_a, pos_b)
        return min(forward, TRACK_LENGTH - forward)

    def _record_alert(self, level: str, vehicle_ids: tuple, distance: float) -> None:
        self.alerts.append({
            'time': self.current_time,
            'level': level,
            'vehicle_ids': list(vehicle_ids),
            'distance': distance,
        })
        if level == 'collision':
            self.collision_alert_count += 1
        else:
            self.safety_warning_count += 1

    def _update_proximity_alerts(self, collision_threshold: float = 0.1) -> None:
        """记录车辆间安全距离预警与碰撞告警"""
        vehicle_ids = sorted(self.vehicles.keys())
        for index, vehicle_id in enumerate(vehicle_ids):
            for other_id in vehicle_ids[index + 1:]:
                vehicle = self.vehicles[vehicle_id]
                other_vehicle = self.vehicles[other_id]
                distance = self._circular_distance(vehicle.position, other_vehicle.position)
                if distance <= collision_threshold:
                    self._record_alert('collision', (vehicle_id, other_id), distance)
                elif distance < SAFETY_DISTANCE:
                    self._record_alert('warning', (vehicle_id, other_id), distance)

    def _is_vehicle_slot_reserved(self, vehicle_id: int, slot_idx: int) -> bool:
        """检查车辆某个工位是否已经被“未完成任务”的货物预占。

        说明：
        - 车辆工位有两层状态：物理占用( vehicle.slots ) 与 任务预占( cargo.assigned_vehicle/slot )。
        - 如果只看 vehicle.slots，会允许“同一工位被分配多个上料任务”的幽灵状态。
        - 这里用简单规则硬性禁止：同一 (vehicle_id, slot_idx) 同时只能对应一个未完成货物。
        """
        for cargo in self.cargos.values():
            if cargo.completion_time is not None:
                continue
            if cargo.assigned_vehicle != vehicle_id:
                continue
            if slot_idx in self._assigned_slot_indices(cargo):
                return True
        return False
    
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
            slot_indices = action.get('slot_indices')
            slot_idx = action.get('slot_idx')
            if slot_indices is None and slot_idx is not None:
                slot_indices = (int(slot_idx),)
            elif slot_indices is not None:
                slot_indices = tuple(int(item) for item in slot_indices)
            else:
                slot_indices = ()

            # 参数有效性检查（避免 KeyError/越界）
            if cargo_id not in self.cargos or vehicle_id not in self.vehicles:
                return assigned_ids
            if not slot_indices:
                return assigned_ids
            if len(set(slot_indices)) != len(slot_indices):
                return assigned_ids

            vehicle = self.vehicles[vehicle_id]
            cargo = self.cargos[cargo_id]
            if len(slot_indices) != cargo.required_slots:
                return assigned_ids

            # 只允许分配仍在上料口等待的货物（位置单一真相由 station/vehicle slots 决定）
            if not self.is_cargo_at_loading_station(cargo):
                return assigned_ids

            # 不支持“重分配”（避免任务多源状态失配）；同一分配重复提交视为 no-op
            if cargo.assigned_vehicle is not None:
                return assigned_ids

            normalized_slot_indices = []
            for target_slot_idx in slot_indices:
                if target_slot_idx < 0 or target_slot_idx >= len(vehicle.slots):
                    return assigned_ids
                if vehicle.slots[target_slot_idx] is not None:
                    return assigned_ids
                if self._is_vehicle_slot_reserved(vehicle_id, target_slot_idx):
                    return assigned_ids
                normalized_slot_indices.append(target_slot_idx)

            assigned_ids.append(cargo_id)
            self._assign_loading_task(cargo_id, vehicle_id, tuple(normalized_slot_indices))
        
        elif action_type == 'assign_unloading':
            # 分配下料目标
            cargo_id = action.get('cargo_id')
            unloading_station_id = action.get('unloading_station_id')
            
            if cargo_id in self.cargos:
                cargo = self.cargos[cargo_id]
                # 下料口按需求假定“无限接收”：卸货完成后不占用下料口工位。
                # 因此这里只做两件事：
                # 1) 校验是否在允许集合内
                # 2) 校验该货物当前确实在车上（避免“提前分配”制造特殊情况）
                if (cargo.completion_time is None and
                    cargo.target_unloading_station is None and
                    unloading_station_id in cargo.allowed_unloading_stations and
                    self.is_cargo_on_vehicle(cargo)):
                    cargo.target_unloading_station = unloading_station_id
        
        return assigned_ids
    
    def _assign_loading_task(self, cargo_id: int, vehicle_id: int, slot_indices: Tuple[int, ...]):
        """分配上料任务（只标记任务，实际上料在车辆对齐时执行）"""
        cargo = self.cargos[cargo_id]

        # 标记货物任务分配
        cargo.assigned_vehicle = vehicle_id
        cargo.assigned_vehicle_slots = tuple(slot_indices)
        cargo.assigned_vehicle_slot = slot_indices[0] if slot_indices else None
    
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
                not self.is_cargo_at_loading_station(cargo)):
                continue
            
            vehicle = self.vehicles[cargo.assigned_vehicle]
            loading_station = self.loading_stations[cargo.loading_station]
            slot_indices = self._assigned_slot_indices(cargo)
            if len(slot_indices) != cargo.required_slots:
                continue
            
            # 检查车辆工位是否仍然空闲
            if any(vehicle.slots[slot_idx] is not None for slot_idx in slot_indices):
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

                # 仅在“两个单槽小卷纸”同时满足条件时，才启用双工位同时上料。
                if cargo.required_slots == 1 and len(vehicle.slots) == 2:
                    other_slot_idx = 1 - slot_indices[0]
                    for other_cargo in self.cargos.values():
                        if other_cargo.id == cargo.id:
                            continue
                        if other_cargo.completion_time is not None:
                            continue
                        if other_cargo.assigned_vehicle != cargo.assigned_vehicle:
                            continue
                        if other_cargo.required_slots != 1:
                            continue
                        if tuple(self._assigned_slot_indices(other_cargo)) != (other_slot_idx,):
                            continue
                        if other_cargo.loading_station != cargo.loading_station:
                            continue
                        if other_cargo.loading_start_time is not None:
                            continue
                        if not self.is_cargo_at_loading_station(other_cargo):
                            continue
                        # 找到同站同车的另一个货物，同时开始上料
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
                for slot_idx in slot_indices:
                    vehicle.slots[slot_idx] = cargo.id
                cargo.loading_start_time = None
                # 注意：不在这里解除锁定，统一在函数最后处理
                picked_up_ids.append(cargo.id)  # 记录完成取货的货物
        
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
            for cargo_id in self._vehicle_unique_cargo_ids(vehicle):
                if cargo_id not in self.cargos:
                    continue
                cargo = self.cargos[cargo_id]
                
                # 检查是否有卸货目标
                if cargo.target_unloading_station is None:
                    continue
                
                unloading_station = self.unloading_stations[cargo.target_unloading_station]
                slot_indices = self._assigned_slot_indices(cargo)
                
                # 如果还未开始下料，检查是否对齐
                if cargo.unloading_start_time is None:
                    # 检查是否对齐下料口
                    if not vehicle.is_aligned_with(unloading_station.position):
                        continue  # 未对齐，等待对齐
                    
                    # 对齐了，开始下料计时
                    cargo.unloading_start_time = self.current_time
                    vehicle.is_loading_unloading = True  # 锁定车辆移动
                    vehicle.velocity = 0.0  # 立即停止车辆
                    
                    # 仅支持两个单槽小卷纸在同一站同时下料。
                    if cargo.required_slots == 1 and len(vehicle.slots) == 2:
                        other_slot_idx = 1 - slot_indices[0]
                        other_cargo_id = vehicle.slots[other_slot_idx]
                        if other_cargo_id is not None and other_cargo_id in self.cargos:
                            other_cargo = self.cargos[other_cargo_id]
                            if (
                                other_cargo.id != cargo.id
                                and other_cargo.required_slots == 1
                                and other_cargo.target_unloading_station == cargo.target_unloading_station
                                and other_cargo.unloading_start_time is None
                            ):
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
                    for target_slot_idx, slot_cargo_id in enumerate(vehicle.slots):
                        if slot_cargo_id == cargo.id:
                            vehicle.slots[target_slot_idx] = None
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
                        'vehicle_id': vehicle_id
                    }
                    self.completed_cargo_list.append(completed_info)
                    
                    # 下料口按“无限接收”口径不建模工位占用，因此卸货完成不写入 station 状态。
                    
                    # 注意：不在这里解除锁定，统一在函数最后处理
                    # 记录已完成的货物ID
                    completed_cargo_ids.append(cargo_id)
        
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

        # 安全距离违例惩罚
        reward += len(self.safety_violations) * REWARD_SAFETY_VIOLATION
        reward += len(self.action_switch_violations) * REWARD_SPEED_CHANGE_PENALTY

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
            if occupied_slots == VEHICLE_SLOT_COUNT:
                reward += 0.5 * LOW_LEVEL_CONTROL_INTERVAL  # 每步+0.5的利用率奖励
            
            for cargo_id in self._vehicle_unique_cargo_ids(vehicle):
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
                'position': station.position / TRACK_LENGTH
            }
            unloading_obs.append(station_obs)
        
        # 待取货物信息（超时货物优先级更高，排在前面）
        waiting_cargos = []
        for cargo in self.cargos.values():
            if cargo.completion_time is None and self.is_cargo_at_loading_station(cargo):
                waiting_cargos.append({
                    'id': cargo.id,
                    'wait_time': cargo.wait_time(self.current_time),
                    'is_timeout': cargo.is_timeout(self.current_time),
                    'loading_station': cargo.loading_station,
                    'loading_slot': cargo.loading_slot,
                    'paper_type': cargo.paper_type,
                    'diameter_m': cargo.diameter_m,
                    'required_slots': cargo.required_slots,
                    'primary_unloading_station': cargo.primary_unloading_station,
                    'alternative_unloading_stations': list(cargo.alternative_unloading_stations),
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
                          if c.completion_time is None and self.is_cargo_at_loading_station(c))
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
        for cargo_id in self._vehicle_unique_cargo_ids(vehicle):
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
