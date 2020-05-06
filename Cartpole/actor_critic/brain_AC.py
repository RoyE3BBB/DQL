import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

## para
LEARNING_RATE = 0.001
REWARD_DECAY = 0.9

## initializer
W_INIT = tf.keras.initializers.RandomNormal(mean=.0, stddev=.1)
B_INIT = tf.keras.initializers.constant(value=.1)

class Actor:
    def __init__(self, n_feature, n_action, learning_rate=0.001, reward_decay=0.9):
        self.n_feature = n_feature
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.model = self.build_net()
        self.losser = tf.keras.losses.sparse_categorical_crossentropy
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        # self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)


    def build_net(self):
        model = Sequential([
            Dense(units=20,
                  activation='relu',
                  kernel_initializer=W_INIT,
                  bias_initializer=B_INIT,
                  input_shape=[self.n_feature]),
            Dense(units=self.n_action,
                  activation='softmax',
                  kernel_initializer=W_INIT,
                  bias_initializer=B_INIT)
        ])

        return model

    def choose_action(self, obsverations):
        prob = self.model.predict(obsverations)
        prob = prob.reshape(-1)
        action = np.random.choice(np.arange(prob.shape[0]), p=prob)

        return action

    @tf.function
    def learn(self, obsverations, action, advantage):
        with tf.GradientTape() as tape:
            pred = self.model(obsverations)
            loss = self.losser(action, pred) * advantage

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class Critic:
    def __init__(self, n_feature, n_action, learning_rate=0.001, reward_decay=0.9):
        self.n_feature = n_feature
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.model = self.build_net()
        self.losser = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def build_net(self):
        model = Sequential([
            Dense(units=20,
                  activation='relu',
                  kernel_initializer=W_INIT,
                  bias_initializer=B_INIT,
                  input_shape=[self.n_feature]),
            Dense(units=1,
                  activation=None,
                  kernel_initializer=W_INIT,
                  bias_initializer=B_INIT)
        ])

        return model

    @tf.function
    def learn(self, obsverations, obsverations_, reward):
        target = reward + self.reward_decay * self.model(obsverations_)

        with tf.GradientTape() as tape:
            pred = self.model(obsverations)
            loss = self.losser(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        td_error = target - pred

        return td_error





























