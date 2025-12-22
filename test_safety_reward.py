"""
测试安全距离奖励机制
验证当车辆试图违反安全距离时，是否正确施加惩罚
"""

from environment import Environment
from config import REWARD_SAFETY_VIOLATION, MAX_SPEED, SAFETY_DISTANCE

def test_safety_reward():
    """测试安全距离奖励功能"""
    env = Environment()

    # 设置场景：两辆车靠得很近
    vehicle0 = env.vehicles[0]
    vehicle1 = env.vehicles[1]

    # 车辆1在前方10.0位置
    vehicle1.position = 10.0
    vehicle1.velocity = 0.0

    # 车辆0在后方8.5位置（距离1.5米，但加速后会违反）
    # 如果车辆0加速：new_velocity = 0 + 1.0*0.5 = 0.5
    # new_position = 8.5 + 0.5*0.5 = 8.75
    # 前向距离 = 10.0 - 8.75 = 1.25 < SAFETY_DISTANCE(2.0) ✓ 违反！
    vehicle0.position = 8.5
    vehicle0.velocity = 0.0

    print("=" * 60)
    print("测试场景：")
    print(f"  车辆0位置: {vehicle0.position:.2f}m，速度: {vehicle0.velocity:.2f}m/s")
    print(f"  车辆1位置: {vehicle1.position:.2f}m，速度: {vehicle1.velocity:.2f}m/s")
    print(f"  当前距离: {vehicle1.position - vehicle0.position:.2f}m")
    print(f"  安全距离: {SAFETY_DISTANCE:.2f}m")
    print("=" * 60)

    # 尝试让车辆0全速前进（会违反安全距离）
    low_level_actions = {
        0: 2,  # 车辆0加速
        1: 1   # 车辆1保持
    }

    high_level_action = None

    # 执行一步
    obs, reward, done = env.step(high_level_action, low_level_actions)

    print("\n执行结果：")
    print(f"  安全违例车辆: {env.safety_violations}")
    print(f"  违例数量: {len(env.safety_violations)}")
    print(f"  单次违例惩罚: {REWARD_SAFETY_VIOLATION:.2f}")
    print(f"  总违例惩罚: {len(env.safety_violations) * REWARD_SAFETY_VIOLATION:.2f}")
    print(f"  本次step总奖励: {reward:.2f}")
    print("=" * 60)

    # 验证
    if len(env.safety_violations) > 0:
        print("[PASS] 成功检测到安全违例")
        if 0 in env.safety_violations:
            print("[PASS] 正确识别车辆0违例")
        print(f"[PASS] 奖励中包含惩罚项（{len(env.safety_violations) * REWARD_SAFETY_VIOLATION:.2f}分）")
    else:
        print("[FAIL] 未检测到安全违例（预期应该检测到）")

    # 测试场景2：安全距离足够时不应该有惩罚
    print("\n" + "=" * 60)
    print("测试场景2：安全距离充足")
    env2 = Environment()

    # 车辆相距很远
    env2.vehicles[0].position = 0.0
    env2.vehicles[1].position = 50.0

    obs2, reward2, done2 = env2.step(None, {0: 2, 1: 1})

    print(f"  安全违例车辆: {env2.safety_violations}")
    print(f"  本次step总奖励: {reward2:.2f}")

    if len(env2.safety_violations) == 0:
        print("[PASS] 安全距离充足时无违例惩罚")
    else:
        print(f"[FAIL] 不应该有违例惩罚（实际: {env2.safety_violations}）")
    print("=" * 60)

if __name__ == "__main__":
    test_safety_reward()
