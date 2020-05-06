import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

class PolicyGradient:
    def __init__(self, n_actions, n_features, reward_decay=0.99, learning_rate=0.001):
        self.n_actions = n_actions
        self.n_features = n_features
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate

        ## build model
        self.policygrad_model = self.build_net()

        ## initializer
        self.w_initializer = tf.initializers.RandomNormal(mean=0, stddev=0.3)
        self.b_initializer = tf.initializers.constant(0.2)

        ## loss & optimizer
        self.losser = tf.keras.losses.sparse_categorical_crossentropy
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        ## memorys
        self.mem_features = []
        self.mem_actions = []
        self.mem_rewards = []

    def choose_action(self, obsvertions):
        actions = self.policygrad_model.predict(obsvertions)
        target_action = np.random.choice(np.arange(actions.shape[1]), p=actions.reshape(-1))
        return target_action

    def store_memorys(self, features, actions, rewards):
        self.mem_features.append(features)
        self.mem_actions.append(actions)
        self.mem_rewards.append(rewards)

    def reward_norm(self, mem_rewards):
        ## reward decay
        rewards = np.zeros_like(mem_rewards)
        total_decay_reward = 0
        for i in reversed(np.arange(len(mem_rewards))):
            total_decay_reward = self.reward_decay * total_decay_reward + mem_rewards[i]
            rewards[i] = total_decay_reward

        ## normalization
            rewards = rewards - np.mean(rewards) / np.std(rewards)
            # rewards = rewards - np.mean(rewards)
            # rewards = rewards / np.std(rewards)

        return tf.convert_to_tensor(rewards, dtype=tf.float32)

    def build_net(self):
        ## build net
        model = Sequential([
            Dense(32,
                  activation='relu',
                  kernel_initializer=tf.initializers.RandomNormal(mean=0, stddev=0.3),
                  bias_initializer=tf.initializers.constant(0.1),
                  input_shape=[self.n_features]),
            Dropout(0.2),
            Dense(self.n_actions,
                  activation='softmax',
                  kernel_initializer=tf.initializers.RandomNormal(mean=0, stddev=0.3),
                  bias_initializer=tf.initializers.constant(0.1))
        ])

        return model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=tf.float64),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ))
    def train(self, feature, action, reward):
        with tf.GradientTape() as tape:
            action_ = self.policygrad_model(feature)
            loss = tf.multiply(self.losser(action, action_, ), reward)

        gradients = tape.gradient(loss, self.policygrad_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policygrad_model.trainable_variables))

    def learn(self):
        rewards = self.reward_norm(self.mem_rewards)
        total_reward = np.sum(self.mem_rewards)
        total_step = len(self.mem_rewards)

        s, a, r = tf.convert_to_tensor(self.mem_features), tf.convert_to_tensor(self.mem_actions), tf.convert_to_tensor(self.mem_rewards)
        # print(s, a, r)
        self.train(s, a, r)

        # for feature, action, reward in zip(self.mem_features, self.mem_actions, rewards):
        #     self.train(feature[np.newaxis, :], action, rewards)

        ## clear memorys
        self.mem_actions.clear()
        self.mem_features.clear()
        self.mem_rewards.clear()

        return rewards, total_reward, total_step