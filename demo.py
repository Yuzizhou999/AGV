"""
演示脚本：展示系统运行情况
"""

import torch
import numpy as np
from config import *
from environment import Environment
from heuristic_high_level import HeuristicHighLevelController
from agent_low_level import LowLevelAgent, LowLevelController
import json
from datetime import datetime


class DemoRunner:
    """演示运行器"""
    
    def __init__(self, demo_duration: float = 300.0, seed: int = 42):
        """
        初始化演示
        
        Args:
            demo_duration: 演示持续时间(秒)
            seed: 随机种子
        """
        self.demo_duration = demo_duration
        
        # 初始化环境
        self.env = Environment(seed=seed)

        # 初始化低层智能体
        low_level_obs_dim = 3 + MAX_VEHICLES
        low_level_action_dim = 3
        self.low_level_agent = LowLevelAgent(low_level_obs_dim, low_level_action_dim)

        # 初始化控制器（使用启发式高层控制器）
        self.high_level_controller = HeuristicHighLevelController(self.env)
        self.low_level_controller = LowLevelController(self.low_level_agent, self.env)
        
        # 统计信息
        self.events = []
    
    def log_event(self, timestamp: float, event_type: str, details: str):
        """记录事件"""
        self.events.append({
            'timestamp': timestamp,
            'time_hms': f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{int(timestamp%60):02d}",
            'type': event_type,
            'details': details
        })
    
    def run(self):
        """运行演示"""
        print("\n" + "=" * 100)
        print("强化学习多智能体穿梭车调度系统 - 演示运行")
        print("=" * 100)
        print(f"配置信息:")
        print(f"  环形轨道长度: {TRACK_LENGTH}")
        print(f"  车辆数量: {MAX_VEHICLES}")
        print(f"  上料口数: {NUM_LOADING_STATIONS} (位置: {LOADING_POSITIONS})")
        print(f"  下料口数: {NUM_UNLOADING_STATIONS} (位置: {UNLOADING_POSITIONS})")
        print(f"  演示时长: {self.demo_duration}秒 ({self.demo_duration/60:.1f}分钟)")
        print(f"  最大速度: {MAX_SPEED}")
        print(f"  安全距离: {SAFETY_DISTANCE}")
        print("=" * 100)
        print()
        
        obs = self.env.reset()
        self.log_event(0, "INIT", "仿真开始")
        
        step_count = 0
        next_high_level_decision = 0.0
        
        print("运行仿真...")
        print()
        
        while self.env.current_time < min(self.demo_duration, EPISODE_DURATION):
            # 高层决策
            high_level_action = None
            if self.env.current_time >= next_high_level_decision:
                high_level_action = self.high_level_controller.compute_action(obs)
                next_high_level_decision = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
                
                # 记录任务分配事件
                if high_level_action:
                    action_type = high_level_action.get('type', 'unknown')
                    if action_type == 'assign_loading':
                        cargo_id = high_level_action.get('cargo_id')
                        vehicle_id = high_level_action.get('vehicle_id')
                        self.log_event(
                            self.env.current_time,
                            "ASSIGN_LOADING",
                            f"货物{cargo_id} -> 车辆{vehicle_id}"
                        )
                    elif action_type == 'assign_unloading':
                        cargo_id = high_level_action.get('cargo_id')
                        unload_id = high_level_action.get('unloading_station_id')
                        self.log_event(
                            self.env.current_time,
                            "ASSIGN_UNLOADING",
                            f"货物{cargo_id} -> 下料口{unload_id}"
                        )
            
            # 低层控制
            low_level_actions = self.low_level_controller.compute_actions()
            
            # 执行一步
            obs, reward, done = self.env.step(high_level_action, low_level_actions)
            step_count += 1
            
            # 检查新货物到达
            if len(self.env.cargos) > 0:
                for cargo_id, cargo in self.env.cargos.items():
                    if (cargo.arrival_time == self.env.current_time and 
                        not any(e['type'] == 'CARGO_ARRIVAL' and e['details'].endswith(str(cargo_id)) 
                               for e in self.events)):
                        self.log_event(
                            self.env.current_time,
                            "CARGO_ARRIVAL",
                            f"货物{cargo_id}到达上料口{cargo.loading_station}"
                        )
            
            # 定期打印进度
            if step_count % 100 == 0:
                progress = (self.env.current_time / min(self.demo_duration, EPISODE_DURATION)) * 100
                print(f"进度: {progress:6.2f}% | 时间: {self.env.current_time:7.2f}s | "
                      f"货物总数: {len(self.env.cargos):3d} | "
                      f"已完成: {self.env.completed_cargos:3d} | "
                      f"超时: {self.env.timed_out_cargos:2d}")
            
            if done:
                break
        
        print()
        print("=" * 100)
        print("仿真结果统计")
        print("=" * 100)
        print()
        
        # 打印车辆最终状态
        print("车辆最终状态:")
        for vehicle_id, vehicle in self.env.vehicles.items():
            print(f"  车辆{vehicle_id}:")
            print(f"    位置: {vehicle.position:.2f}")
            print(f"    速度: {vehicle.velocity:.2f}")
            print(f"    工位1: {'有货物' if vehicle.slots[0] is not None else '空'}")
            print(f"    工位2: {'有货物' if vehicle.slots[1] is not None else '空'}")
        print()
        
        # 打印货物统计
        print("货物统计:")
        completed_count = sum(1 for c in self.env.cargos.values() if c.completion_time is not None)
        waiting_count = sum(1 for c in self.env.cargos.values() 
                           if c.completion_time is None and c.current_location.startswith("IP_"))
        on_vehicle_count = sum(1 for c in self.env.cargos.values()
                              if c.current_location.startswith("vehicle_"))
        
        print(f"  总货物数: {len(self.env.cargos)}")
        print(f"  已完成: {completed_count}")
        print(f"  待取: {waiting_count}")
        print(f"  车上: {on_vehicle_count}")
        print(f"  完成率: {completed_count/max(1, len(self.env.cargos))*100:.2f}%")
        print()
        
        # 打印性能指标
        print("性能指标:")
        if completed_count > 0:
            avg_completion_time = np.mean([c.completion_time - c.arrival_time 
                                          for c in self.env.cargos.values() 
                                          if c.completion_time is not None])
            print(f"  平均完成时间: {avg_completion_time:.2f}秒")
        
        if len(self.env.cargos) > 0:
            avg_wait_time = np.mean([c.wait_time(self.env.current_time) 
                                     for c in self.env.cargos.values()])
            print(f"  平均等待时间: {avg_wait_time:.2f}秒")
        
        print(f"  超时件数: {self.env.timed_out_cargos}")
        print()
        
        # 保存事件日志
        self._save_event_log()
        
        print("=" * 100)
        print()
    
    def _save_event_log(self):
        """保存事件日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"demo_event_log_{timestamp}.json"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
        
        print(f"事件日志已保存到: {log_path}")
        print(f"总事件数: {len(self.events)}")


class DetailedVisualizationDemo:
    """详细可视化演示"""
    
    def __init__(self):
        self.env = Environment(seed=42)

        # 初始化低层智能体
        low_level_agent = LowLevelAgent(3 + MAX_VEHICLES, 3)

        # 初始化控制器（使用启发式高层控制器）
        self.high_level_controller = HeuristicHighLevelController(self.env)
        self.low_level_controller = LowLevelController(low_level_agent, self.env)
    
    def print_state(self, step: int):
        """打印当前系统状态"""
        print(f"\n--- 仿真步 {step}, 时间 {self.env.current_time:.2f}s ---")
        print(f"当前货物数: {len(self.env.cargos)}, 已完成: {self.env.completed_cargos}")
        
        print("\n车辆状态:")
        for vid, v in self.env.vehicles.items():
            print(f"  V{vid}: pos={v.position:.1f}, vel={v.velocity:.2f}, "
                  f"slot=[{v.slots[0] if v.slots[0] is not None else '-'}, "
                  f"{v.slots[1] if v.slots[1] is not None else '-'}]")
        
        print("\n上料口状态:")
        for sid, s in self.env.loading_stations.items():
            print(f"  IP{sid}: pos={s.position:.1f}, "
                  f"slot=[{s.slots[0] if s.slots[0] is not None else '-'}, "
                  f"{s.slots[1] if s.slots[1] is not None else '-'}]")
        
        print("\n等待货物:")
        waiting = [c for c in self.env.cargos.values() 
                  if c.completion_time is None and c.current_location.startswith("IP_")]
        for cargo in waiting[:5]:  # 最多显示5个
            print(f"  货物{cargo.id}: 等待{cargo.wait_time(self.env.current_time):.1f}s")
    
    def run_short_demo(self, num_steps: int = 50):
        """运行短演示，显示详细状态"""
        print("\n" + "=" * 80)
        print("详细状态演示（50步）")
        print("=" * 80)
        
        obs = self.env.reset()
        
        for step in range(num_steps):
            high_level_action = self.high_level_controller.compute_action(obs)
            low_level_actions = self.low_level_controller.compute_actions()
            obs, reward, done = self.env.step(high_level_action, low_level_actions)
            
            if step % 10 == 0 or step == num_steps - 1:
                self.print_state(step)
            
            if done:
                break
        
        print("\n" + "=" * 80)


def main():
    """主函数"""
    print("\n开始执行演示...\n")
    
    # 运行主演示
    demo = DemoRunner(demo_duration=600.0)  # 10分钟演示
    demo.run()
    
    # 运行详细演示
    print("\n执行详细状态演示...\n")
    detailed_demo = DetailedVisualizationDemo()
    detailed_demo.run_short_demo(num_steps=50)


if __name__ == "__main__":
    main()
