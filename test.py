"""
测试脚本：验证环境和智能体的功能
"""

import unittest
import numpy as np
import torch
from config import *
from environment import Environment, Cargo, Vehicle, LoadingStation, UnloadingStation
from agent_high_level import HighLevelAgent, HighLevelController
from agent_low_level import LowLevelAgent, LowLevelController


class TestEnvironment(unittest.TestCase):
    """环境单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.env = Environment(seed=42)
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        # 检查车辆数量
        self.assertEqual(len(self.env.vehicles), MAX_VEHICLES)
        
        # 检查上料口数量
        self.assertEqual(len(self.env.loading_stations), NUM_LOADING_STATIONS)
        
        # 检查下料口数量
        self.assertEqual(len(self.env.unloading_stations), NUM_UNLOADING_STATIONS)
        
        # 检查车辆初始状态
        for vehicle in self.env.vehicles.values():
            self.assertEqual(len(vehicle.slots), 2)
            self.assertEqual(vehicle.velocity, 0.0)
            self.assertIsNone(vehicle.slots[0])
            self.assertIsNone(vehicle.slots[1])
    
    def test_track_normalization(self):
        """测试轨道位置规范化"""
        # 测试边界情况
        self.assertEqual(self.env._normalize_position(0.0), 0.0)
        self.assertEqual(self.env._normalize_position(TRACK_LENGTH), 0.0)
        self.assertEqual(self.env._normalize_position(TRACK_LENGTH + 10), 10.0)
        self.assertEqual(self.env._normalize_position(-10), TRACK_LENGTH - 10)
    
    def test_cargo_generation(self):
        """测试货物生成"""
        # 运行一段时间应该生成货物
        initial_cargo_count = len(self.env.cargos)
        
        # 快速推进到货物应该到达的时间
        self.env.current_time = self.env.next_arrival_time + 1.0
        new_cargo_ids = self.env._check_and_generate_cargo()
        
        # 应该生成新货物
        self.assertGreater(len(self.env.cargos), initial_cargo_count)
        self.assertGreater(len(new_cargo_ids), 0)
    
    def test_vehicle_alignment(self):
        """测试车辆与工位对齐"""
        vehicle = self.env.vehicles[0]
        loading_station = self.env.loading_stations[0]
        
        # 将车放在工位位置
        vehicle.position = loading_station.position
        self.assertTrue(vehicle.is_aligned_with(loading_station.position))
        
        # 移动车离开
        vehicle.position = loading_station.position + 5.0
        self.assertFalse(vehicle.is_aligned_with(loading_station.position))
    
    def test_safety_distance_check(self):
        """测试安全距离约束"""
        # 初始状态应该满足安全距离
        vehicle0 = self.env.vehicles[0]
        vehicle1 = self.env.vehicles[1]
        
        # 将车辆0放在车辆1前面很近的位置（沿行驶方向）
        # 车辆0在位置48，车辆1在位置50，车辆0继续前进会超过车辆1
        vehicle0.position = vehicle1.position - SAFETY_DISTANCE + 0.5  # 48.5
        vehicle0.velocity = 0.0
        
        # 以MAX_SPEED前进0.5秒后，new_position = 48.5 + 5*0.5 = 51.0
        # 此时车辆0在车辆1(50)前面，需要检查从51到50的环形距离
        # 环形距离 = 100 - 51 + 50 = 99，这是安全的
        
        # 正确的测试：将车辆0放在车辆1的后面很近的位置
        # 车辆1在50，我们把车辆0放在52（即车辆1后面，距离为100-52+50=98）
        # 不对，环形轨道上，如果车辆0在位置52，车辆1在50，
        # 那么车辆0到车辆1的前向距离是 50 - 52 + 100 = 98（如果52 > 50）
        
        # 重新设计：车辆0在车辆1的前方不远处，如果车辆0减速或停止，车辆1会追上
        # 但这里我们测试的是车辆0的安全检查
        # 安全检查是：车辆0移动后，与前方车辆的距离
        
        # 设置场景：车辆1在10的位置，车辆0在8的位置
        # 车辆0向前移动，需要与前方的车辆1保持距离
        vehicle1.position = 10.0
        vehicle0.position = 10.0 - SAFETY_DISTANCE + 0.5  # 8.5
        
        # 车辆0以MAX_SPEED前进：new_position = 8.5 + 5*0.5 = 11.0
        # 此时车辆0(11.0)已经超过车辆1(10.0)，前向距离为 100 - 11 + 10 = 99
        # 这仍然是安全的
        
        # 正确场景：车辆0在车辆1后面，会撞上车辆1
        # 车辆1在10，车辆0在9.5（距离0.5 < SAFETY_DISTANCE=2）
        # 但是安全检查计算的是移动后的new_position
        vehicle1.position = 10.0
        vehicle0.position = 7.0  # 车辆0在车辆1后面3米
        
        # 如果车辆0以MAX_SPEED=5前进0.5秒，new_position = 7 + 2.5 = 9.5
        # 前向距离 = 10 - 9.5 = 0.5 < SAFETY_DISTANCE(2)，不安全！
        is_safe = self.env._check_safety_distance(0, MAX_SPEED)
        self.assertFalse(is_safe)
    
    def test_step_execution(self):
        """测试单步执行"""
        initial_time = self.env.current_time
        
        # 执行一步
        low_level_actions = {0: 1, 1: 1}  # 保持速度
        obs, reward, done = self.env.step(None, low_level_actions)
        
        # 检查时间是否增加
        self.assertGreater(self.env.current_time, initial_time)
        
        # 检查观测结构
        self.assertIn('vehicles', obs)
        self.assertIn('loading_stations', obs)
        self.assertIn('unloading_stations', obs)
        self.assertIn('global_info', obs)
    
    def test_episode_termination(self):
        """测试episode结束"""
        # 快速推进到episode结束
        self.env.current_time = EPISODE_DURATION - LOW_LEVEL_CONTROL_INTERVAL / 2
        
        low_level_actions = {0: 1, 1: 1}
        obs, reward, done = self.env.step(None, low_level_actions)
        
        # 执行一步后应该超过EPISODE_DURATION
        self.assertTrue(self.env.current_time >= EPISODE_DURATION or done)


class TestVehicle(unittest.TestCase):
    """车辆类单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.vehicle = Vehicle(
            id=0,
            position=10.0,
            velocity=1.0,
            slots=[None, None],
            slot_operation_end_time=[0.0, 0.0],
            is_loading_unloading=False
        )
    
    def test_empty_slot_detection(self):
        """测试空工位检测"""
        self.assertTrue(self.vehicle.has_empty_slot())
        
        self.vehicle.slots[0] = 0
        self.assertTrue(self.vehicle.has_empty_slot())
        
        self.vehicle.slots[1] = 1
        self.assertFalse(self.vehicle.has_empty_slot())
    
    def test_distance_calculation(self):
        """测试距离计算"""
        # 直线距离
        distance = self.vehicle.distance_to(15.0)
        self.assertEqual(distance, 5.0)
        
        # 环形距离（向后）
        self.vehicle.position = 95.0
        distance = self.vehicle.distance_to(5.0)
        self.assertEqual(distance, 10.0)


class TestLoadingStation(unittest.TestCase):
    """上料站单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.station = LoadingStation(0, 20.0)
    
    def test_cargo_placement(self):
        """测试货物放置"""
        # 放置货物
        success = self.station.place_cargo(0, 0)
        self.assertTrue(success)
        self.assertEqual(self.station.slots[0], 0)
        
        # 尝试在相同位置放置第二个货物
        success = self.station.place_cargo(1, 0)
        self.assertFalse(success)
    
    def test_cargo_removal(self):
        """测试货物移除"""
        self.station.place_cargo(0, 0)
        
        # 移除货物
        cargo_id = self.station.remove_cargo(0)
        self.assertEqual(cargo_id, 0)
        self.assertIsNone(self.station.slots[0])


class TestHighLevelAgent(unittest.TestCase):
    """高层智能体测试"""
    
    def setUp(self):
        """测试前设置"""
        self.obs_dim = 9
        self.action_dim = 10
        self.agent = HighLevelAgent(self.obs_dim, self.action_dim)
    
    def test_agent_initialization(self):
        """测试智能体初始化"""
        self.assertEqual(self.agent.obs_dim, self.obs_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
    
    def test_action_selection(self):
        """测试动作选择"""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        action = self.agent.select_action(obs)
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_memory_storage(self):
        """测试经验存储"""
        state = np.random.randn(self.obs_dim)
        action = 0
        reward = 1.0
        next_state = np.random.randn(self.obs_dim)
        done = False
        
        self.agent.store_transition(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_network_training(self):
        """测试网络训练"""
        # 填充内存
        for _ in range(100):
            state = np.random.randn(self.obs_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.obs_dim)
            done = np.random.rand() > 0.8
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        # 训练
        loss = self.agent.train(batch_size=32)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)


class TestLowLevelAgent(unittest.TestCase):
    """下层智能体测试"""
    
    def setUp(self):
        """测试前设置"""
        self.obs_dim = 5
        self.action_dim = 3
        self.agent = LowLevelAgent(self.obs_dim, self.action_dim)
    
    def test_agent_initialization(self):
        """测试智能体初始化"""
        self.assertEqual(self.agent.obs_dim, self.obs_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
    
    def test_action_space(self):
        """测试动作空间"""
        # 动作应该是 0, 1, 2
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        for _ in range(100):
            action = self.agent.select_action(obs)
            self.assertIn(action, [0, 1, 2])


class IntegrationTest(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.env = Environment(seed=42)
        
        # 初始化智能体
        self.high_level_obs_dim = MAX_VEHICLES * 4 + NUM_LOADING_STATIONS * 3 + 2
        self.high_level_action_dim = 10
        self.high_level_agent = HighLevelAgent(
            self.high_level_obs_dim, 
            self.high_level_action_dim
        )
        
        self.low_level_agent = LowLevelAgent(
            3 + MAX_VEHICLES, 3
        )
        
        # 初始化控制器
        self.high_level_controller = HighLevelController(
            self.high_level_agent, 
            self.env
        )
        self.low_level_controller = LowLevelController(
            self.low_level_agent, 
            self.env
        )
    
    def test_full_episode(self):
        """测试完整episode"""
        obs = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 100  # 限制步数以加快测试
        
        while steps < max_steps:
            # 高层决策
            high_level_action = self.high_level_controller.compute_action(obs)
            
            # 低层控制
            low_level_actions = self.low_level_controller.compute_actions()
            
            # 执行一步
            obs, reward, done = self.env.step(high_level_action, low_level_actions)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 验证执行成功
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, float)
    
    def test_cargo_flow(self):
        """测试货物流向"""
        obs = self.env.reset()
        
        # 快速推进到有货物为止
        while len(self.env.cargos) == 0 and self.env.current_time < 100:
            high_level_action = self.high_level_controller.compute_action(obs)
            low_level_actions = self.low_level_controller.compute_actions()
            obs, reward, done = self.env.step(high_level_action, low_level_actions)
        
        # 应该有货物
        self.assertGreater(len(self.env.cargos), 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestVehicle))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadingStation))
    suite.addTests(loader.loadTestsFromTestCase(TestHighLevelAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestLowLevelAgent))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTest))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
