from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_probability as tfp

class ACModel(Model):
    def __init__(self, n_feature, n_action, learning_rate=0.001, gamma=0.99):
        super().__init__()
        self.n_feature = n_feature
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.gamma = gamma

        ## losser & optimizer
        self.loss = None
        self.optimizer = None

        ## initializer
        self.w_init = tf.keras.initializers.RandomNormal(0., .1)

class Actor(ACModel):
    def __init__(self, n_feature, n_action, a_bound=None):
        super().__init__(n_feature, n_action)
        self.a_bound = a_bound

        self.loss = tf.keras.losses.sparse_categorical_crossentropy
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

        ## actor net
        self.dense_ac = Dense(256, activation='relu', kernel_initializer=self.w_init)
        self.output_ac_mu = Dense(self.n_action, activation='tanh', kernel_initializer=self.w_init)
        self.output_ac_sigma = Dense(self.n_action, activation='softplus', kernel_initializer=self.w_init)

    def call(self, inputs):
        l1 = self.dense_ac(inputs)
        output_mu = self.output_ac_mu(l1)
        output_sigma = self.output_ac_sigma(l1)

        return output_mu, output_sigma

    def compute_loss(self, obsverations, actions, td_error):
        with tf.GradientTape() as tape:
            mu, sigma = self.call(obsverations)
            mu, sigma = self.para_normalization(mu, sigma)

            dist = tfp.distributions.Normal(mu, sigma)
            log_prob = dist.log_prob(actions)
            loss = tf.multiply(log_prob, td_error)

            ## encourage exploration
            entropy = dist.entropy()
            loss = 0.01 * entropy + loss

            loss = -tf.reduce_mean(loss)
            # loss = tf.reduce_mean(loss)

        grad = tape.gradient(loss, self.trainable_variables)

        return grad

    def apply_grad(self, grad):
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def choose_action(self, obsverations):
        mu, sigma = self.call(obsverations)
        mu, sigma = self.para_normalization(mu, sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        action = tf.clip_by_value(dist.sample(1), self.a_bound[0], self.a_bound[1])

        return action

    def para_normalization(self, mu, sigma):
        mu = mu * self.a_bound[1]
        sigma = sigma + 1e-4

        return mu, sigma

class Critic(ACModel):
    def __init__(self, n_feature, n_action):
        super().__init__(n_feature, n_action)

        self.loss_ct = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

        ## critic net
        self.dense_ct = Dense(128, activation='relu', kernel_initializer=self.w_init)
        self.output_ct = Dense(1, kernel_initializer=self.w_init)


    def call(self, inputs,):
        l1 = self.dense_ct(inputs)
        output = self.output_ct(l1)

        return output

    def compute_loss(self, s, r, s_):
        target = r + self.gamma * self.call(s_)
        with tf.GradientTape() as tape:
            pred = self.call(s)
            td_error = target - pred
            loss = tf.reduce_mean(self.loss_ct(target, pred))

        grad = tape.gradient(loss, self.trainable_variables)
        return td_error, grad

    def apply_grad(self, grad):
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

def update_para(train_ac, train_ct, glo_ac, glo_ct):
    train_ac.set_weights(glo_ac.get_weights())
    train_ct.set_weights(glo_ct.get_weights())