import numpy as np
import gym

from utils import *


def extract_policy(v, gamma = 1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v

def policy_iteration(env, gamma = 1.0):
    policy = np.random.choice(env.env.nA, size=(env.env.nS))
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':
    # env_name  = 'FrozenLake-v0'
    env_name = "FrozenLake8x8-v0"
    env = gym.make(env_name)

    policy = policy_iteration(env)

    show_env_and_policy(env, policy)
    evaluate_policy(env, policy)

