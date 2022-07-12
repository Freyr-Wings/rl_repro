import numpy as np
import time

def show_env_and_policy(env, policy):
    print("Environment and corresponding policy:")
    for ri in range(env.env.nrow):
        for ci in range(env.env.ncol):
            print("[{}{}]".format(str(env.env.desc[ri][ci], 'utf-8'), int(policy[ri*env.env.ncol+ci])), end="")
        print()

def run_episode(env, policy, gamma = 1.0, render = False):
    obs = env.reset()
    total_reward = 0
    while True:
        if render:
            env.render("human")
            time.sleep(1)
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += reward
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 1000):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    scores_avg = np.mean(scores)
    print('Average scores for {} episode: {}'.format(n, scores_avg))