#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/11/26
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import pprint as pp
from replay_buffer import ReplayBuffer
from simulator import Simulator
from pre_process_data import recall_data
from util.logger import logger

from actor import Actor
from critic import Critic


class OUNoise:
    """noise for action"""
    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state


def gene_actions(item_space, weight_batch):
    """use output of actor network to calculate action list
    Args:
        item_space: recall items, dict: id: embedding
        weight_batch: actor network outputs

    Returns:
        recommendation list
    """
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    max_ids = list()
    for weight in weight_batch:
        score = np.dot(item_weights, weight)
        idx = np.argmax(score)
        max_ids.append(item_ids[idx])
    return max_ids


def gene_action(item_space, weight):
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    score = np.dot(item_weights, weight)
    idx = np.argmax(score)
    return item_ids[idx]


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("reward", episode_reward)
    episode_max_q = tf.Variable(0.)
    tf.summary.scalar("max_q_value", episode_max_q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, episode_max_q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def learn_from_batch(replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
    if replay_buffer.size() < batch_size:
        pass
    samples = replay_buffer.sample_batch(batch_size)
    state_batch = np.asarray([_[0] for _ in samples])
    action_batch = np.asarray([_[1] for _ in samples])
    reward_batch = np.asarray([_[2] for _ in samples])
    n_state_batch = np.asarray([_[3] for _ in samples])

    # calculate predicted q value
    action_weights = actor.predict_target(state_batch)
    n_action_batch = gene_actions(item_space, action_weights, action_len)
    target_q_batch = critic.predict_target(n_state_batch.reshape((-1, s_dim)), n_action_batch.reshape((-1, a_dim)))
    y_batch = []
    for i in range(batch_size):
        y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])

    # train critic
    q_value, critic_loss, _ = critic.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))

    # train actor
    action_weight_batch_for_gradients = actor.predict(state_batch)
    action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)
    a_gradient_batch = critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))
    actor.train(state_batch, a_gradient_batch[0])

    # update target networks
    actor.update_target_network()
    critic.update_target_network()

    return np.amax(q_value), critic_loss


def train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args):
    # set up summary operators
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # initialize target network weights
    actor.hard_update_target_network()
    critic.hard_update_target_network()

    # initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']))

    for i in range(int(args['max_episodes'])):
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        item_space = recall_data
        state = env.reset()
        # update average parameters every 1000 episodes
        if (i + 1) % 10 == 0:
            env.rewards, env.group_sizes, env.avg_states, env.avg_actions = env.avg_group()
        for j in range(args['max_episodes_len']):
            weight = actor.predict(np.reshape(state, [1, s_dim])) + exploration_noise.noise().reshape(
                (1, int(args['action_item_num']), int(a_dim / int(args['action_item_num'])))
            )
            action = gene_actions(item_space, weight, int(args['action_item_num']))
            reward, n_state = env.step(action[0])
            replay_buffer.add(list(state.reshape((s_dim,))),
                              list(action.reshape((a_dim,))),
                              [reward],
                              list(n_state.reshape((s_dim,))))
            ep_reward += reward
            ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic, item_space,
                                                        args['action_item_num'], s_dim, a_dim)
            ep_q_value += ep_q_value_
            loss += critic_loss
            state = n_state
            if (j + 1) % 50 == 0:
                logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, ep_reward))
            summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                           summary_vars[1]: ep_q_value,
                                                           summary_vars[2]: loss})
            writer.add_summary(summary_str, i)

    writer.close()
    saver = tf.train.Saver()
    ckpt_path = os.path.join(os.path.dirname(__file__), "models")
    saver.save(sess, ckpt_path, write_meta_graph=False)


def main(args):
    # init memory data
    # data = load_data()
    with tf.Session() as sess:
        # simulated environment
        env = Simulator()
        s_dim = int(args['embedding']) * int(args['state_item_num'])
        a_dim = int(args['embedding']) * int(args['action_item_num'])

        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)

        train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--embedding", help="dimension of item embedding", default=30)
    parser.add_argument("--state_item_num", help="click history list length for user", default=12)
    parser.add_argument("--action_item_num", help="length of the recommendation item list", default=4)
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=100)
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_)
