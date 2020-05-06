import numpy as np
import gym
from Cartpole.brain_AC import Actor, Critic

## para
MAX_EPISODE = 3000

## set env
env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped
n_feature = env.observation_space.shape[0]
n_action = env.action_space.n

actor = Actor(n_feature, n_action)
critic = Critic(n_feature, n_action)

circle_reward = 0

for episode in np.arange(MAX_EPISODE):
    observations = env.reset()
    observations = observations[np.newaxis, :]
    total_reward = 0
    total_step = 0

    while True:
        env.render()

        action = actor.choose_action(observations)

        observations_, reward, done, _ = env.step(action)

        observations_ = observations_[np.newaxis, :]

        if done: reward = -10

        advantage = critic.learn(observations, observations_, reward)
        actor.learn(observations, [action], advantage)

        observations = observations_
        total_reward += reward
        total_step += 1

        if done:
            print('episode: {}, total reward: {}, total step: {}'.format(episode, total_reward, total_step))
            circle_reward += total_reward

            if episode % 100 == 0:
                print('average reward in 100 episode: {}'.format(circle_reward / 100))
                circle_reward = 0
            break