from model import Actor, Critic
import numpy as np
import tensorflow as tf
import multiprocessing as mtp
import gym

## env para
GAME = 'Pendulum-v0'
env = gym.make(GAME)
N_FEATURE = env.observation_space.shape[0]
N_ACTION = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

## episode para
MAX_EPISODE = 2000 + 1
MAX_STEP = 200 + 1
UPDATE_CIRCLE = 5

## processing coordinator
GLOBAL_REWARD = mtp.Array('f', MAX_EPISODE + 1)
GLOBAL_EPISODE = mtp.Value('i', 0)
N_WORKER = 4

## global para lock
LOCK_STEP = mtp.Lock()
LOCK_REWARD = mtp.Lock()

class Worker(object):
    def __init__(self, name):
        self.name = name
        self.env = gym.make(GAME)

        self.actor = Actor(N_FEATURE, N_ACTION, A_BOUND)
        self.critic = Critic(N_FEATURE, N_ACTION)
        self.actor.build(input_shape=(None, self.actor.n_feature))
        self.critic.build(input_shape=(None, self.critic.n_feature))

    def work(self, queue_para, queue_buffer):
        buffer_s, buffer_r, buffer_a, buffer_s_ = [], [], [], []

        ## syn_net_paras
        self.updata_para(queue_para)

        while GLOBAL_EPISODE.value < MAX_EPISODE:
            episode_reword = 0
            obsveration = self.env.reset()

            for episode_step in np.arange(1, MAX_STEP):
                if self.name == 'worker_0':
                    self.env.render()
                # self.env.render()

                obsveration = obsveration[np.newaxis, :]
                action = self.actor.choose_action(obsveration)
                obsveration_, reward, done, _ = self.env.step(action)
                done = True if episode_step == MAX_STEP - 1 else False

                obsveration = np.squeeze(obsveration)
                action = np.squeeze(action, axis=(0, 1))
                reward = np.squeeze(reward, axis=(0))
                obsveration_ = np.squeeze(obsveration_)

                episode_reword += reward
                buffer_s.append(obsveration)
                buffer_a.append(action)
                buffer_r.append((reward + 8) / 8)
                buffer_s_.append(obsveration_)

                if episode_step % UPDATE_CIRCLE == 0 or done:
                    s = tf.convert_to_tensor(buffer_s, dtype=tf.float32)
                    a = tf.convert_to_tensor(buffer_a, dtype=tf.float32)
                    r = tf.convert_to_tensor(buffer_r, dtype=tf.float32)
                    s_ = tf.convert_to_tensor(buffer_s_, dtype=tf.float32)

                    td_error, grad_ct = self.critic.compute_loss(s, r, s_)
                    grad_ac = self.actor.compute_loss(s, a, td_error)
                    queue_buffer.put([grad_ac, grad_ct])

                    self.updata_para(queue_para)

                    buffer_s.clear()
                    buffer_a.clear()
                    buffer_s_.clear()
                    buffer_r.clear()

                    if done:
                        LOCK_STEP.acquire()
                        i = GLOBAL_EPISODE.value
                        if i == 0:
                            GLOBAL_REWARD[i] = episode_reword
                        else:
                            GLOBAL_REWARD[i] = 0.9 * GLOBAL_REWARD[i - 1] + 0.1 * episode_reword
                            print('name: {}, global episode: {}, average reward: {}'.format(self.name, i,
                                                                                            GLOBAL_REWARD[i - 1]))

                        GLOBAL_EPISODE.value += 1
                        LOCK_STEP.release()
                        break

                obsveration = obsveration_

    def updata_para(self, queue_para):
        para_ac, para_ct = queue_para.get()
        self.actor.set_weights(para_ac)
        self.critic.set_weights(para_ct)

class Brain(object):
    def __init__(self):
        self.env = gym.make(GAME)

        self.actor = Actor(N_FEATURE, N_ACTION, A_BOUND)
        self.critic = Critic(N_FEATURE, N_ACTION)
        self.actor.build(input_shape=(None, self.actor.n_feature))
        self.critic.build(input_shape=(None, self.critic.n_feature))

    def work(self, queues_para, queues_buffer):
        self.syn_global_para()

        while True:
            for i in np.arange(N_WORKER):
                grad_ac, grad_ct = queues_buffer[i].get()
                self.actor.apply_grad(grad_ac)
                self.critic.apply_grad(grad_ct)

                para_ac = self.actor.get_weights()
                para_ct = self.critic.get_weights()
                queues_para[i].put([para_ac, para_ct])


            if GLOBAL_EPISODE.value == MAX_EPISODE:
                break

    def syn_global_para(self):
        para_ac = self.actor.get_weights()
        para_ct = self.critic.get_weights()
        for i in np.arange(N_WORKER):
            queues_para[i].put([para_ac, para_ct])



if __name__ == '__main__':
    ## create brain & workers
    brain = Brain()
    workers = []
    for i in np.arange(N_WORKER):
        name = 'worker_' + str(i)
        workers.append(Worker(name))

    ## create mutiprocessing queue
    queues_para = []
    queues_buffer = []
    for i in np.arange(N_WORKER):
        queues_para.append(mtp.Queue(1))
        queues_buffer.append(mtp.Queue(1))

    ## start work
    coordinator = mtp.Process(target=brain.work, args=(queues_para, queues_buffer))
    coordinator.start()

    agents = []
    for i in np.arange(N_WORKER):
        agents.append(mtp.Process(target=workers[i].work, args=(queues_para[i], queues_buffer[i])))
        agents[i].start()

    coordinator.join()
    print('finish')