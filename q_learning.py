import gym
import gym_cityflow
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

EPISODES = 10000
max_state_space_size = 1000000
gamma = 0.95
alpha = 0.4
min_alpha = 0.01
random_prob = 0.1
min_random_prob = 0
decrease_step = 100
print_step = 1000


def obs_to_state(obs):
    state = 0
    base = int(math.log(max_state_space_size)/math.log(len(obs)))
    for i in range(len(obs)):
        state += obs[i] * (base ** i)
    return int(state)


def get_action(q_table, obs, env):
    state = obs_to_state(obs)
    action = np.argmax(q_table[state], -1)
    return action


def update_q_table(q_table, obs, action, next_obs, reward, env):
    state = obs_to_state(obs)
    next_state = obs_to_state(next_obs)
    #print(f"state: {state}, next_state: {next_state}")
    sample = reward + gamma * np.max(q_table[next_state])
    old = q_table[state][action]
    q_table[state][action] = (1-alpha) * old + alpha * sample
    return


def print_ep_data(ep, reward, turns):
    print(f"Episode {ep}, mean reward: {reward/turns}")


if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    action_space = env.action_space.n
    q_table = np.zeros((max_state_space_size, action_space))
    alpha_k = (alpha - min_alpha) / (EPISODES / decrease_step)
    random_prob_k = (random_prob - min_random_prob) / (EPISODES / decrease_step)
    mean_rewards = []
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        turns = 0
        while not done:
            p = np.random.random(1)[0]
            if p < random_prob:
                action = np.random.randint(0, 9)
            else:
                action = get_action(q_table, obs, env)
            #action = get_action(q_table, obs, env)
            #print(action)
            next_obs, reward, done, info = env.step(action)
            #print(f"**{turns}**")
            #print(next_obs)
            #print(reward)
            update_q_table(q_table, obs, action, next_obs, reward, env)
            total_reward += reward
            obs = deepcopy(next_obs)
            turns += 1
        #print(q_table[0])
        mean_rewards.append(total_reward/turns)

        if ep % print_step == 0:
            print_ep_data(ep, total_reward, turns)
        if ep % decrease_step == 1:
            random_prob = random_prob - random_prob_k
            alpha = alpha - alpha_k

    #print(turns)
    plt.plot(mean_rewards)
    plt.ylabel("mean reward")
    plt.xlabel("episode")
    plt.savefig("q_learning_plot.png")

