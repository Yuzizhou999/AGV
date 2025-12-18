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
                cargo.current_location.startswith("IP_")):
                assigned_vehicle_slots.add((cargo.assigned_vehicle, cargo.assigned_vehicle_slot))
        
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
            best_slot_idx = None
            min_distance = float('inf')
            
            for vehicle_id, vehicle in self.env.vehicles.items():
                # 检查车辆是否有空工位
                for slot_idx in range(2):
                    if (vehicle.slots[slot_idx] is None and 
                        (vehicle_id, slot_idx) not in assigned_vehicle_slots):
                        # 计算距离（考虑环形轨道）
                        distance = vehicle.distance_to(loading_position)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_vehicle_id = vehicle_id
                            best_slot_idx = slot_idx
                
                # 找到空工位后跳出（每个车辆只检查一个工位）
                if best_vehicle_id == vehicle_id:
                    break
            
            # 如果找到合适的车辆，立即分配
            if best_vehicle_id is not None:
                return {
                    'type': 'assign_loading',
                    'cargo_id': cargo_id,
                    'vehicle_id': best_vehicle_id,
                    'slot_idx': best_slot_idx,
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
            for slot_idx, cargo_id in enumerate(vehicle.slots):
                if cargo_id is not None:
                    cargo = self.env.cargos[cargo_id]
                    
                    # 如果货物还没有分配下料目标
                    if cargo.target_unloading_station is None:
                        # 简单策略：选择允许的第一个下料口
                        # （可以改进为选择最近的下料口）
                        allowed_stations = list(cargo.allowed_unloading_stations)
                        if allowed_stations:
                            target_station_id = allowed_stations[0]
                            
                            return {
                                'type': 'assign_unloading',
                                'cargo_id': cargo_id,
                                'unloading_station_id': target_station_id,
                                'slot_idx': 0  # 默认使用第一个工位
                            }
        
        return None
