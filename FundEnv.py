import numpy as np
import pandas as pd
import copy
import tensorflow as tf
import os

#### item embedding dict
df_items = pd.read_csv('item_space.csv')
item_ids = df_items['item_ids']
item_emb = df_items['item_emb'].apply(lambda x: [float(_) for _ in x.split(',')])
item_ids_emb_dict = dict(zip(item_ids, item_emb))
#### samples
df_samples = pd.read_csv('samples.csv')

item_num = 3
state_num = 6
emb_dim = 10


def get_state_ids(loc, sample_data):
    state_ids = sample_data.iloc[loc]
    state_ids = np.array(state_ids.split('|'), dtype=int)
    return state_ids


def get_item_emb(ids, dict):
    if len(np.asarray(ids).shape) == 1:
        emb = np.array([dict[v] for v in ids])
    else:
        emb = []
        for i in ids:
            emb.append(np.array([dict[v] for v in i]))
        emb = np.asarray(emb)
    return emb


def next_s(state, action, reward_detail):
    next_state = copy.copy(state)
    for i in range(0, len(reward_detail)):
        if reward_detail[i] > 0:
            next_state = np.append(next_state, action[i])
    next_state = next_state[-state_num:]
    return next_state


class FundEnv(object):
    def __init__(self):
        self.state = get_state_ids(0, df_samples['state'])
        self.user_count = 0
        self.step_count = 0
        self.max_step = 10
        self.reset()
        self._preprocess_data()

        self._build_reward_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    @staticmethod
    def _preprocess_data():
        df_samples['state_emb'] = df_samples['state'].apply(lambda x: [int(_) for _ in x.split('|')])
        df_samples['reward_detail_emb'] = df_samples['reward_detail'].apply(lambda x: [int(_) for _ in x[1:-1].split()])

    def _build_reward_net(self):
        self.data = tf.placeholder(tf.float32, [None, state_num * emb_dim], 'train_data')
        self.label = tf.placeholder(tf.int64, [None, item_num], 'train_data')

        layer_1 = tf.layers.dense(
            inputs=self.data,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        layer_2 = tf.layers.dense(
            inputs=layer_1,
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.pred = tf.layers.dense(
            inputs=layer_2,
            units=3,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='output'
        )

        self.loss = tf.losses.mean_squared_error(self.label, self.pred)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def _train_reward_net(self, data, label):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.data: data,
            self.label: label
        })
        return loss

    def load_reward_net(self):
        # check if there is saved model
        if 'checkpoint' in os.listdir():
            self.saver.restore(self.sess, 'checkpoint/reward_net.ckpt')
        # train new model
        else:
            data = np.asarray(list(df_samples['state_emb']))
            label = np.asarray(list(df_samples['reward_detail_emb']))
            data_emb = np.reshape(get_item_emb(data, item_ids_emb_dict), [-1, state_num * emb_dim])
            for i in range(10000):
                loss = self._train_reward_net(data_emb, label)
                print('epoch %d: loss=%.4f' % (i, loss))
            self.saver.save(self.sess, 'checkpoint/reward_net.ckpt')
            print('model saved')

    def predict_reward(self):
        state_emb = get_item_emb(self.state, item_ids_emb_dict)
        state_emb = np.reshape(state_emb, [1, -1])
        pred = self.sess.run(self.pred, feed_dict={
            self.data: state_emb
        })
        return pred[0]

    def test(self):
        data = np.asarray(list(df_samples['state_emb']))
        label = np.asarray(list(df_samples['reward_detail_emb']))

        data_emb = np.reshape(get_item_emb(data, item_ids_emb_dict), [-1, state_num * emb_dim])

        pred = self.sess.run(self.pred, feed_dict={
            self.data: data_emb
        })
        for i in range(len(df_samples)):
            print(pred[i], end='')
            print(label[i])

    def reset(self):
        if self.user_count > 9999:
            self.user_count = 0
        self.step_count = 0
        self.state = get_state_ids(self.user_count, df_samples['state'])
        state_emb = get_item_emb(self.state, item_ids_emb_dict)
        return state_emb

    def step(self, action):
        # reward_detail = np.random.choice([0, 0, 0, 1, 1, 5], 3)
        reward_detail = self.predict_reward()
        reward = sum(reward_detail)
        self.state = next_s(self.state, action, reward_detail)
        self.step_count += 1
        if self.step_count > self.max_step:
            done = 1
            self.user_count += 1
        else:
            done = 0
        state_emb = get_item_emb(self.state, item_ids_emb_dict)
        return state_emb, reward, done

