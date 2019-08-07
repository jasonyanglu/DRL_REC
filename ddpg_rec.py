import numpy as np
import tensorflow as tf
import os

from actor import Actor
from critic import Critic
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from FundEnv import item_ids_emb_dict, get_item_emb


class DDPG_REC:

    def __init__(self, state_item_num, action_item_num, emb_dim, batch_size, tau, actor_lr, critic_lr,
                 gamma, buffer_size, item_space, summary_dir):

        self.state_item_num = state_item_num
        self.action_item_num = action_item_num
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.item_space = item_space
        self.summary_dir = summary_dir

        self.sess = tf.Session()

        self.s_dim = emb_dim * state_item_num
        self.a_dim = emb_dim * action_item_num
        self.actor = Actor(self.sess, state_item_num, action_item_num, emb_dim, batch_size, tau, actor_lr)
        self.critic = Critic(self.sess, state_item_num, action_item_num, emb_dim,
                             self.actor.get_num_trainable_vars(), gamma, tau, critic_lr)
        self.exploration_noise = OUNoise(self.a_dim)

        # set up summary operators
        self.summary_ops, self.summary_vars = self.build_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # initialize target network weights
        self.actor.hard_update_target_network()
        self.critic.hard_update_target_network()

        # initialize replay memory
        self.replay_buffer = ReplayBuffer(buffer_size)

    def gene_actions(self, weight_batch):
        """use output of actor network to calculate action list
        Args:
            weight_batch: actor network outputs

        Returns:
            recommendation list
        """
        item_ids = list(self.item_space.keys())
        item_weights = list(self.item_space.values())
        max_ids = list()
        for weight in weight_batch:
            score = np.dot(item_weights, np.transpose(weight))
            idx = np.argmax(score, 0)
            max_ids.append([item_ids[_] for _ in idx])
        return max_ids

    # def gene_action(self, weight):
    #     """use output of actor network to calculate action list
    #     Args:
    #         weight: actor network outputs
    #
    #     Returns:
    #         recommendation list
    #     """
    #     item_ids = list(self.item_space.keys())
    #     item_weights = list(self.item_space.values())
    #     score = np.dot(item_weights, np.transpose(weight))
    #     idx = np.argmax(score)
    #     return item_ids[idx]

    @staticmethod
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

    def _train(self):
        samples = self.replay_buffer.sample_batch(self.batch_size)
        state_batch = np.asarray([_[0] for _ in samples])
        action_batch = np.asarray([_[1] for _ in samples])
        reward_batch = np.asarray([_[2] for _ in samples])
        n_state_batch = np.asarray([_[3] for _ in samples])

        seq_len_batch = np.asarray([self.state_item_num] * self.batch_size)

        # calculate predicted q value
        action_weights = self.actor.predict_target(state_batch, seq_len_batch)  # [batch_size,
        n_action_batch = self.gene_actions(action_weights.reshape((-1, self.action_item_num, self.emb_dim)))
        n_action_emb_batch = get_item_emb(n_action_batch, item_ids_emb_dict)
        target_q_batch = self.critic.predict_target(n_state_batch.reshape((-1, self.s_dim)),
                                                    n_action_emb_batch.reshape((-1, self.a_dim)), seq_len_batch)
        y_batch = []
        for i in range(self.batch_size):
            y_batch.append(reward_batch[i] + self.critic.gamma * target_q_batch[i])

        # train critic
        q_value, critic_loss, _ = self.critic.train(state_batch, action_batch,
                                                    np.reshape(y_batch, (self.batch_size, 1)), seq_len_batch)

        # train actor
        action_weight_batch_for_gradients = self.actor.predict(state_batch, seq_len_batch)
        action_batch_for_gradients = self.gene_actions(action_weight_batch_for_gradients)
        action_emb_batch_for_gradients = get_item_emb(action_batch_for_gradients, item_ids_emb_dict)
        a_gradient_batch = self.critic.action_gradients(state_batch,
                                                        action_emb_batch_for_gradients.reshape((-1, self.a_dim)),
                                                        seq_len_batch)
        self.actor.train(state_batch, a_gradient_batch[0], seq_len_batch)

        # update target networks
        self.actor.update_target_network()
        self.critic.update_target_network()

        return np.amax(q_value), critic_loss

    def action(self, state):
        weight = self.actor.predict(np.reshape(state, [1, self.s_dim]), np.array([self.state_item_num])) + \
                 self.exploration_noise.noise().reshape(
                     (1, self.action_item_num, int(self.a_dim / self.action_item_num)))
        action = self.gene_actions(weight)
        return np.array(action[0])

    def perceive_and_train(self, state, action, reward, n_state, done):
        action_emb = get_item_emb(action, item_ids_emb_dict)
        self.replay_buffer.add(list(state.reshape((self.s_dim,))),
                               list(action_emb.reshape((self.a_dim,))),
                               [reward],
                               list(n_state.reshape((self.s_dim,))))

        # Store transitions to replay start size then start training
        ep_q_value_, critic_loss = 0, 0
        if self.replay_buffer.size() > self.batch_size:
            ep_q_value_, critic_loss = self._train()

        # if self.time_step % 10000 == 0:
        # self.actor_network.save_network(self.time_step)
        # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

        return ep_q_value_, critic_loss

    def write_summary(self, ep_reward, ep_q_value, loss, i):
        summary_str = self.sess.run(self.summary_ops, feed_dict={self.summary_vars[0]: ep_reward,
                                                                 self.summary_vars[1]: ep_q_value,
                                                                 self.summary_vars[2]: loss})
        self.writer.add_summary(summary_str, i)

    def save(self):
        self.writer.close()
        saver = tf.train.Saver()
        ckpt_path = os.path.join(os.path.dirname(__file__), "models")
        saver.save(self.sess, ckpt_path, write_meta_graph=False)
