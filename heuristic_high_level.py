"""
启发式高层控制器
替换神经网络，使用简单规则：将货物分配给最靠近上料口的空闲小车
"""

from typing import Dict, Optional
from config import *


class HeuristicHighLevelController:
    """启发式高层控制器
    
    核心规则：
    1. 货物分配：找到离上料口最近的空闲小车
    2. 下料分配：随机选择允许的下料口（保持原有逻辑）
    """
    
    def __init__(self, env):
        """
        初始化控制器
        
        Args:
            env: 环境实例
        """
        self.env = env
    
    def compute_action(self, observation: Dict) -> Optional[Dict]:
        """
        基于观测计算高层动作
        
        Args:
            observation: 环境观测
        
        Returns:
            高层动作字典或None
        """
        # 优先处理上料任务分配
        loading_action = self._assign_loading_task(observation)
        if loading_action:
            return loading_action
        
        # 其次处理下料目标分配
        unloading_action = self._assign_unloading_target(observation)
        if unloading_action:
            return unloading_action
        
        return None
    
    def _assign_loading_task(self, observation: Dict) -> Optional[Dict]:
        """
        分配上料任务：将货物分配给离上料口最近的空闲小车
        
        Args:
            observation: 环境观测
        
        Returns:
            上料任务动作或None
        """
        waiting_cargos = observation.get('waiting_cargos', [])
        
        # 按优先级排序（超时货物优先）
        waiting_cargos.sort(key=lambda x: (-x.get('priority', 0), -x['wait_time']))
        
        # 获取已分配的车辆工位（避免重复分配）
        assigned_vehicle_slots = set()
        for cargo in self.env.cargos.values():
            if (cargo.assigned_vehicle is not None and 
                self.env.is_cargo_at_loading_station(cargo)):
                for slot_idx in self.env._assigned_slot_indices(cargo):
                    assigned_vehicle_slots.add((cargo.assigned_vehicle, slot_idx))
        
        # 遍历等待的货物
        for cargo_info in waiting_cargos:
            cargo_id = cargo_info['id']
            cargo = self.env.cargos[cargo_id]
            
            # 跳过已分配的货物
            if cargo.assigned_vehicle is not None:
                continue
            
            # 获取货物所在的上料口位置
            loading_station = self.env.loading_stations[cargo.loading_station]
            loading_position = loading_station.position
            
            # 找到离上料口最近的空闲小车
            best_vehicle_id = None
            best_slot_indices = None
            min_distance = float('inf')
            
            for vehicle_id, vehicle in self.env.vehicles.items():
                candidate_groups = self.env._vehicle_available_slot_groups(vehicle_id, cargo)
                for slot_group in candidate_groups:
                    if any((vehicle_id, slot_idx) in assigned_vehicle_slots for slot_idx in slot_group):
                        continue

                    distance = vehicle.distance_to(loading_position)
                    if distance < min_distance:
                        min_distance = distance
                        best_vehicle_id = vehicle_id
                        best_slot_indices = slot_group
            
            # 如果找到合适的车辆，立即分配
            if best_vehicle_id is not None:
                return {
                    'type': 'assign_loading',
                    'cargo_id': cargo_id,
                    'vehicle_id': best_vehicle_id,
                    'slot_indices': list(best_slot_indices or ()),
                    'priority': cargo_info.get('priority', 0)
                }
        
        return None
    
    def _assign_unloading_target(self, observation: Dict) -> Optional[Dict]:
        """
        分配下料目标：为已装载货物选择下料口
        
        Args:
            observation: 环境观测
        
        Returns:
            下料任务动作或None
        """
        # 遍历所有车辆，找到需要分配下料目标的货物
        for vehicle_id, vehicle in self.env.vehicles.items():
            for cargo_id in self.env._vehicle_unique_cargo_ids(vehicle):
                if cargo_id is None or cargo_id not in self.env.cargos:
                    continue
                cargo = self.env.cargos[cargo_id]

                if cargo.target_unloading_station is not None:
                    continue

                target_station_id = self._select_unloading_station(vehicle_id, cargo)
                if target_station_id is None:
                    continue

                return {
                    'type': 'assign_unloading',
                    'cargo_id': cargo_id,
                    'unloading_station_id': target_station_id,
                    'slot_idx': 0
                }
        
        return None

    def _select_unloading_station(self, vehicle_id: int, cargo) -> Optional[int]:
        """优先固定主下料口，仅在备选口明显更近时切换。"""
        vehicle = self.env.vehicles[vehicle_id]
        allowed_stations = sorted(int(station_id) for station_id in cargo.allowed_unloading_stations)
        if not allowed_stations:
            return None

        primary_station = cargo.primary_unloading_station
        if primary_station not in allowed_stations:
            primary_station = allowed_stations[0]

        primary_distance = vehicle.distance_to(self.env.unloading_stations[primary_station].position)
        best_station = primary_station
        best_distance = primary_distance

        for station_id in allowed_stations:
            station = self.env.unloading_stations.get(station_id)
            if station is None:
                continue
            distance = vehicle.distance_to(station.position)
            if distance < best_distance:
                best_distance = distance
                best_station = station_id

        if (
            best_station != primary_station
            and best_distance + ALTERNATIVE_UNLOADING_SWITCH_DISTANCE < primary_distance
        ):
            return best_station
        return primary_station
