import numpy as np
import gym

from utils import *


def extract_policy(v, gamma = 1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    v = np.zeros(env.env.nS)
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration: {}.'.format(i+1))
            break
    return v


if __name__ == '__main__':
    # env_name  = 'FrozenLake-v0'
    env_name = "FrozenLake8x8-v0"
    env = gym.make(env_name)

    optimal_v = value_iteration(env)
    policy = extract_policy(optimal_v)

    show_env_and_policy(env, policy)
    evaluate_policy(env, policy)
