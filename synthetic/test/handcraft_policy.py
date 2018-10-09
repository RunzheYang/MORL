from __future__ import absolute_import, division, print_function
import gym
import sys
import numpy as np


def heuristic(env, s, target=0):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    offsets = [(0, 0), (-0.60, 0.1), (0.68, 0.33)]
    x_offset, y_offset = offsets[target]
    angle_targ = (s[0] + x_offset) * 0.5 + s[
        2] * 1.0  # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.9 * np.abs(s[0] + x_offset) + y_offset  # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    # if env.continuous:
    # 	a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
    # 	a = np.clip(a, -1, +1)
    # else:
    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1

    return a


if __name__ == "__main__":
    env = gym.make('MultiObj-LunarLander-v0' if len(sys.argv) < 2 else sys.argv[1])
    # env = LunarLanderContinuous()
    # target = int(input("0:yellow | 1:blue | 2:red\n"))
    for target in [0, 1, 2]:
        s = env.reset()
        total_reward = 0
        steps = 0
        while True:
            a = heuristic(env, s, target=target)
            s, r, done, info = env.step(a)
            env.render()
            total_reward += r
            if steps % 20 == 0 or done:
                # print(["{:+0.2f}".format(x) for x in s])
                print("step {} mean_reward {:+0.2f}".format(steps, total_reward.mean()))
                print("total_reward:", ["{:+0.2f}".format(r) for r in total_reward])
            steps += 1
            # if done: break
            if done:
                for _ in range(20): env.render()
                break
