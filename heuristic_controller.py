"""
启发式底层控制器
替换神经网络控制，使用基于规则的方法控制小车速度
"""

import numpy as np
from typing import Dict, Optional
from config import *


class HeuristicLowLevelController:
    """启发式底层控制器"""
    
    def __init__(self, environment):
        """
        初始化控制器
        
        Args:
            environment: 环境实例
        """
        self.env = environment
        # 每辆车的运动规划
        self.vehicle_plans: Dict[int, Optional[Dict]] = {i: None for i in range(MAX_VEHICLES)}
        # 运动规划包含：
        # - target_position: 目标位置
        # - accel_distance: 加速段距离
        # - decel_distance: 减速段距离
        # - cruise_distance: 匀速段距离
        # - cruise_speed: 巡航速度（可能是最大速度或更低）
        # - phase: 当前阶段 ('accel', 'cruise', 'decel', 'aligned')

    def compute_actions(self, deterministic: bool = True) -> Dict[int, int]:
        """
        统一接口：与 train.py 的控制器调用约定保持一致。

        启发式控制器本身是确定性的，不需要区分 deterministic=True/False。
        这里保留 deterministic 参数只是为了兼容 PPO 控制器的接口，避免上层出现特殊情况分支。
        """
        _ = deterministic  # 兼容参数，占位不使用
        return self.get_actions()

    def reset_episode(self):
        """每个 episode 开始前重置内部规划，避免跨 episode 残留。"""
        for vehicle_id in self.vehicle_plans.keys():
            self.vehicle_plans[vehicle_id] = None
    
    def get_actions(self) -> Dict[int, int]:
        """
        为所有车辆生成控制动作
        
        Returns:
            Dict[int, int]: {vehicle_id: action} 其中action为0(减速),1(保持),2(加速)
        """
        actions = {}
        
        for vehicle_id, vehicle in self.env.vehicles.items():
            # 更新车辆的目标位置
            self._update_vehicle_target(vehicle_id)
            
            # 根据车辆状态决定动作
            action = self._get_vehicle_action(vehicle_id)
            actions[vehicle_id] = action
        
        return actions
    
    def _update_vehicle_target(self, vehicle_id: int):
        """更新车辆的目标位置和运动规划
        
        基于车辆任务队列决定目标位置：
        1. 优先处理队列中的第一个任务（FIFO）
        2. 如果没有任务但车上有货，自动寻找下料位置
        3. 如果完全空闲，保持当前状态
        """
        vehicle = self.env.vehicles[vehicle_id]
        current_plan = self.vehicle_plans[vehicle_id]
        
        # 单一真相：目标位置从 cargo/slots 的真实状态推导
        target_position = self.env.get_vehicle_target_position(vehicle_id)
        
        # 无任务，清除规划
        if target_position is None:
            self.vehicle_plans[vehicle_id] = None
            return
        
        # 检查是否需要重新规划
        need_replan = False
        
        # 情况1: 没有规划
        if current_plan is None:
            need_replan = True
        # 情况2: 目标位置改变（优先检查，确保任务切换时立即重新规划）
        elif current_plan['target_position'] != target_position:
            need_replan = True
        # 情况3: 速度为0且不在目标位置（可能因安全距离停止）
        elif vehicle.velocity == 0.0 and not vehicle.is_aligned_with(target_position) and not vehicle.is_loading_unloading:
            need_replan = True
        # 情况4: 路径上有其他车辆正在进行上下料操作
        elif self._is_blocked_by_loading_vehicle(vehicle_id, target_position):
            need_replan = True
        # 情况5: 已经处于decel或aligned阶段，但检测到未对齐（位置或速度不满足）
        elif current_plan['phase'] in ['decel', 'aligned']:
            # 计算当前距离
            current_distance = vehicle.distance_to(target_position)
            
            # 检查位置是否接近（距离小于等于2米认为接近）
            if current_distance <= 2.0:
                # 接近目标时，检查是否真正对齐
                if not vehicle.is_aligned_with(target_position):
                    # 未对齐，需要重新规划
                    # 可能是位置不够精确，或者速度太快
                    need_replan = True
        # 情况6: 已经对齐目标位置
        elif vehicle.is_aligned_with(target_position):
            # 如果已对齐，标记为aligned阶段
            if current_plan['phase'] != 'aligned':
                current_plan['phase'] = 'aligned'
            return
        
        # 执行重新规划
        if need_replan:
            self.vehicle_plans[vehicle_id] = self._plan_motion(vehicle, target_position)
    
    def _plan_motion(self, vehicle, target_position: float, is_station: bool = True) -> Dict:
        """
        规划从当前位置到目标位置的运动轨迹
        
        运动分为三个阶段：
        1. 加速阶段：从当前速度加速到巡航速度
        2. 匀速阶段：保持巡航速度
        3. 减速阶段：从巡航速度减速到目标速度
        
        Args:
            vehicle: 车辆对象
            target_position: 目标位置
            is_station: 是否是上下料口（如果是，则需要减速到SPEED_TOLERANCE以下）
        
        Returns:
            运动规划字典
        """
        current_v = vehicle.velocity
        distance = vehicle.distance_to(target_position)
        a = MAX_ACCELERATION
        v_max = MAX_SPEED
        
        # 如果是上下料口，目标速度为SPEED_TOLERANCE/2（留有余量）
        # 否则目标速度为0
        v_target = SPEED_TOLERANCE / 2 if is_station else 0.0
        
        # 计算从当前速度减速到目标速度所需的距离
        if current_v > v_target:
            s_decel_from_current = (current_v ** 2 - v_target ** 2) / (2 * a)
        else:
            s_decel_from_current = 0
        
        # 计算从0加速到最大速度所需的距离
        s_accel_to_max = (v_max ** 2 - v_target ** 2) / (2 * a)
        
        # 计算从最大速度减速到目标速度所需的距离
        s_decel_from_max = s_accel_to_max  # 对称
        
        # 判断是否能达到最大速度
        # 如果当前速度已经较高，先计算减速到目标速度需要的距离
        if s_decel_from_current >= distance:
            # 距离太短，必须立即减速
            return {
                'target_position': target_position,
                'target_speed': v_target,
                'cruise_speed': v_target,
                'accel_distance': 0,
                'cruise_distance': 0,
                'decel_distance': distance,
                'phase': 'decel'
            }
        
        # 计算如果加速到最大速度再减速，总共需要多少距离
        # s_total = (v_max^2 - v_current^2)/(2a) + (v_max^2 - v_target^2)/(2a)
        s_accel = (v_max ** 2 - current_v ** 2) / (2 * a)
        total_accel_decel = s_accel + s_decel_from_max
        
        if total_accel_decel <= distance:
            # 可以达到最大速度
            cruise_speed = v_max
            accel_distance = s_accel
            decel_distance = s_decel_from_max
            cruise_distance = distance - total_accel_decel
        else:
            # 不能达到最大速度，计算最高能达到的速度
            # s_accel + s_decel = distance
            # (v_cruise^2 - v_current^2)/(2a) + (v_cruise^2 - v_target^2)/(2a) = distance
            # (2*v_cruise^2 - v_current^2 - v_target^2)/(2a) = distance
            # v_cruise^2 = a*distance + (v_current^2 + v_target^2)/2
            cruise_speed_squared = a * distance + (current_v ** 2 + v_target ** 2) / 2
            if cruise_speed_squared > v_target ** 2:
                cruise_speed = min(v_max, np.sqrt(cruise_speed_squared))
            else:
                cruise_speed = max(current_v, v_target)
            
            accel_distance = (cruise_speed ** 2 - current_v ** 2) / (2 * a) if cruise_speed > current_v else 0
            decel_distance = (cruise_speed ** 2 - v_target ** 2) / (2 * a)
            cruise_distance = distance - accel_distance - decel_distance
            
            # 修正可能的数值误差
            if cruise_distance < 0:
                cruise_distance = 0
                # 重新分配距离
                accel_distance = distance / 2
                decel_distance = distance / 2
        
        # 确定初始阶段
        if current_v < cruise_speed - 0.1:
            phase = 'accel'
        elif accel_distance <= 0 and cruise_distance <= 0:
            phase = 'decel'
        elif cruise_distance > 0:
            phase = 'cruise'
        else:
            phase = 'decel'
        
        return {
            'target_position': target_position,
            'target_speed': v_target,
            'cruise_speed': cruise_speed,
            'accel_distance': accel_distance,
            'cruise_distance': cruise_distance,
            'decel_distance': decel_distance,
            'phase': phase,
            'accel_start_distance': distance,  # 记录规划时的总距离
        }
    
    def _get_vehicle_action(self, vehicle_id: int) -> int:
        """
        获取车辆的控制动作
        
        Returns:
            int: 0(减速), 1(保持), 2(加速)
        """
        vehicle = self.env.vehicles[vehicle_id]
        plan = self.vehicle_plans[vehicle_id]
        
        # 如果正在上下料，强制保持静止
        if vehicle.is_loading_unloading:
            # 如果速度不为0，减速
            if vehicle.velocity > 0:
                return 0  # 减速到0
            else:
                return 1  # 保持静止，不移动
        
        # 无规划时，智能巡航：空闲车辆主动接近最近的待取货位置
        if plan is None:
            return self._cruise_action_smart(vehicle_id)
        
        # 如果已经对齐目标位置，保持静止
        if plan['phase'] == 'aligned':
            if vehicle.velocity > 0:
                return 0  # 减速到停止
            else:
                return 1  # 保持静止
        
        # 计算当前到目标的距离
        current_distance = vehicle.distance_to(plan['target_position'])
        
        # 更新阶段
        self._update_phase(vehicle, plan, current_distance)
        
        # 根据阶段决定动作
        if plan['phase'] == 'accel':
            # 加速阶段
            if vehicle.velocity < plan['cruise_speed'] - 0.1:
                # 检查安全距离
                if self._will_violate_safety(vehicle_id, 2):
                    return 1  # 保持
                return 2  # 加速
            else:
                # 达到巡航速度
                plan['phase'] = 'cruise' if plan['cruise_distance'] > 0 else 'decel'
                return 1  # 保持
        
        elif plan['phase'] == 'cruise':
            # 匀速阶段
            if vehicle.velocity < plan['cruise_speed'] - 0.1:
                if self._will_violate_safety(vehicle_id, 2):
                    return 1
                return 2  # 加速到巡航速度
            elif vehicle.velocity > plan['cruise_speed'] + 0.1:
                return 0  # 减速到巡航速度
            else:
                # 检查是否需要进入减速阶段
                if current_distance <= plan['decel_distance'] + 1.0:  # 余量
                    plan['phase'] = 'decel'
                    return 0  # 开始减速
                else:
                    # 检查安全距离
                    if self._will_violate_safety(vehicle_id, 1):
                        return 0  # 减速
                    return 1  # 保持
        
        elif plan['phase'] == 'decel':
            # 减速阶段
            v_target = plan.get('target_speed', 0.0)
            
            # 检查是否已经很接近目标
            if current_distance <= 1.0:  # tolerance
                # 检查速度是否接近目标速度
                if vehicle.velocity > v_target + 0.1:
                    return 0  # 继续减速到目标速度
                elif vehicle.is_aligned_with(plan['target_position']):
                    plan['phase'] = 'aligned'
                    return 1  # 已对齐（位置+速度），保持
                else:
                    # 位置接近但未对齐
                    # 检查是位置不够精确还是速度太快
                    position_dist = min(
                        vehicle.distance_to(plan['target_position']),
                        TRACK_LENGTH - vehicle.distance_to(plan['target_position'])
                    )
                    
                    if position_dist <= 1.0 and vehicle.velocity > SPEED_TOLERANCE:
                        # 位置对齐但速度太快，继续减速
                        return 0
                    elif position_dist > 1.0 and vehicle.velocity <= SPEED_TOLERANCE:
                        # 速度已经很低但位置不准确，触发重新规划
                        # 通过在下一次update时重新检测来触发
                        return 1  # 暂时保持，等待下一次重新规划
                    else:
                        # 位置和速度都不满足，继续减速
                        return 0 if vehicle.velocity > v_target else 1
            else:
                # 还没到目标，继续减速
                if vehicle.velocity > v_target + 0.1:
                    return 0  # 减速
                else:
                    # 速度已经降到目标速度但还没到目标位置
                    # 保持当前速度，继续靠近目标
                    return 1  # 保持
        
        return 1  # 默认保持
    
    def _update_phase(self, vehicle, plan: Dict, current_distance: float):
        """根据当前状态更新运动阶段"""
        if plan['phase'] == 'aligned':
            return
        
        # 获取目标速度
        v_target = plan.get('target_speed', 0.0)
        
        # 检查是否应该进入减速阶段
        # 计算从当前速度减速到目标速度需要的距离
        if vehicle.velocity > v_target:
            decel_needed = (vehicle.velocity ** 2 - v_target ** 2) / (2 * MAX_ACCELERATION)
        else:
            decel_needed = 0
        decel_needed += 1.0  # 安全余量
        
        if current_distance <= decel_needed:
            plan['phase'] = 'decel'
        elif vehicle.velocity >= plan['cruise_speed'] - 0.1:
            # 已达到或接近巡航速度
            if plan['cruise_distance'] > 0 and current_distance > decel_needed:
                plan['phase'] = 'cruise'
            else:
                plan['phase'] = 'decel'
        elif vehicle.velocity < plan['cruise_speed'] - 0.1:
            # 还未达到巡航速度
            if current_distance > decel_needed:
                plan['phase'] = 'accel'
            else:
                plan['phase'] = 'decel'
    
    def _cruise_action(self, vehicle_id: int) -> int:
        """
        巡航模式：保持匀速运行（考虑安全距离）
        
        Returns:
            int: 0(减速), 1(保持), 2(加速)
        """
        vehicle = self.env.vehicles[vehicle_id]
        
        # 目标速度为最大速度的70%（巡航速度）
        cruise_speed = MAX_SPEED * 0.7
        
        if vehicle.velocity < cruise_speed - 0.1:
            # 需要加速，但要检查安全距离
            if self._will_violate_safety(vehicle_id, 2):
                return 1  # 保持
            return 2  # 加速
        elif vehicle.velocity > cruise_speed + 0.1:
            return 0  # 减速
        else:
            # 检查前方是否有车太近
            if self._will_violate_safety(vehicle_id, 1):
                return 0  # 减速
            return 1  # 保持
    
    def _cruise_action_smart(self, vehicle_id: int) -> int:
        """
        智能巡航模式：空闲车辆主动接近最近的待取货位置
        
        策略：
        1. 如果车辆有空位，寻找最近的有货物等待的上料口
        2. 朝着目标方向缓慢移动，保持一定速度
        3. 如果没有待取货物，则普通巡航
        
        Returns:
            int: 0(减速), 1(保持), 2(加速)
        """
        vehicle = self.env.vehicles[vehicle_id]
        
        # 只有有空位的车辆才主动寻找货物
        if not vehicle.has_empty_slot():
            return self._cruise_action(vehicle_id)
        
        # 寻找最近的有待取货物的上料口
        nearest_station_pos = None
        min_distance = float('inf')
        
        for loading_station in self.env.loading_stations.values():
            # 检查该上料口是否有待取货物（未被分配或已分配给当前车辆）
            has_waiting_cargo = False
            for cargo in self.env.cargos.values():
                if (cargo.loading_station == loading_station.id and
                    self.env.is_cargo_at_loading_station(cargo) and
                    cargo.completion_time is None and
                    (cargo.assigned_vehicle is None or cargo.assigned_vehicle == vehicle_id)):
                    has_waiting_cargo = True
                    break
            
            if has_waiting_cargo:
                distance = vehicle.distance_to(loading_station.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_station_pos = loading_station.position
        
        # 如果没有找到待取货物，执行普通巡航
        if nearest_station_pos is None:
            return self._cruise_action(vehicle_id)
        
        # 计算朝向目标的理想速度（较低速度，避免过快到达）
        target_cruise_speed = MAX_SPEED * 0.5  # 50%最大速度
        
        # 如果距离很近（小于安全距离的2倍），停车等待
        if min_distance < SAFETY_DISTANCE * 2:
            if vehicle.velocity > 0.1:
                return 0  # 减速
            else:
                return 1  # 保持停止
        
        # 根据当前速度和目标速度调整
        if vehicle.velocity < target_cruise_speed - 0.2:
            # 需要加速，但要检查安全距离
            if self._will_violate_safety(vehicle_id, 2):
                return 1  # 保持
            return 2  # 加速
        elif vehicle.velocity > target_cruise_speed + 0.2:
            return 0  # 减速
        else:
            # 检查前方是否有车太近
            if self._will_violate_safety(vehicle_id, 1):
                return 0  # 减速
            return 1  # 保持
    
    def _is_blocked_by_loading_vehicle(self, vehicle_id: int, target_position: float) -> bool:
        """
        检查从当前位置到目标位置的路径上是否有其他车辆正在进行上下料操作
        
        Args:
            vehicle_id: 当前车辆ID
            target_position: 目标位置
        
        Returns:
            bool: 如果路径上有车辆在上下料则返回True
        """
        vehicle = self.env.vehicles[vehicle_id]
        
        # 计算到目标的距离（顺时针方向）
        distance_to_target = vehicle.distance_to(target_position)
        
        # 检查其他车辆
        for other_id, other_vehicle in self.env.vehicles.items():
            if other_id == vehicle_id:
                continue
            
            # 只关心正在进行上下料的车辆
            if not other_vehicle.is_loading_unloading:
                continue
            
            # 检查该车辆是否在当前车辆到目标的路径上
            # 计算当前车辆到其他车辆的距离（顺时针方向）
            distance_to_other = vehicle.distance_to(other_vehicle.position)
            
            # 如果其他车辆在路径上（距离小于到目标的距离），则被堵住
            if distance_to_other < distance_to_target:
                return True
        
        return False
    
    def _will_violate_safety(self, vehicle_id: int, action: int) -> bool:
        """
        检查执行某个动作是否会违反安全距离
        
        Args:
            vehicle_id: 车辆ID
            action: 0(减速), 1(保持), 2(加速)
        
        Returns:
            bool: 是否会违反安全距离
        """
        vehicle = self.env.vehicles[vehicle_id]
        
        # 计算新速度
        if action == 0:
            new_velocity = max(0, vehicle.velocity - MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
        elif action == 1:
            new_velocity = vehicle.velocity
        else:  # action == 2
            new_velocity = min(MAX_SPEED, vehicle.velocity + MAX_ACCELERATION * LOW_LEVEL_CONTROL_INTERVAL)
        
        # 复用环境的安全距离检测逻辑，避免双份实现出现不一致
        return not self.env._check_safety_distance(vehicle_id, new_velocity)
