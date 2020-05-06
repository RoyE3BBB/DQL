import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from Cartpole.brain_policygrad import PolicyGradient

## set env
env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space.shape)
# print(env.observation_space.high)
# print(env.observation_space.low)

pg = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0])

for episode in np.arange(3000):
    observation = env.reset()

    while True:
        env.render()

        action = pg.choose_action(observation[np.newaxis, :])

        observation_, reward, done, _ = env.step(action)

        pg.store_memorys(observation, action, reward)

        if done:
            rewards, total_reward, total_step = pg.learn()
            print('episode: {}, total reward: {}, total step: {}'.format(episode, total_reward, total_step))
            # if episode == 0:
            #     plt.plot(rewards)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_
