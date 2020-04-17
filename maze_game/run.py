from maze_game.brain import RL
from maze_game.maze_env import Maze
import tensorflow as tf
import numpy as np

def run():
    step = 0
    for episode in tf.range(300):
        observation = env.reset()

        while True:
            env.render()
            action = DQL.choose_action(observation)
            observation_, reward, done = env.step(action)
            DQL.store_transition(observation, action, reward, observation_)

            if(step > 250) and (step % 5 ==0):
                DQL.learn()

            step += 1
            observation = observation_
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    DQL = RL(env.n_actions, env.n_features)
    env.after(100, run())
    env.mainloop()