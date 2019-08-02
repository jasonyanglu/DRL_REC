import numpy as np
import tensorflow as tf

from actor import Actor
from critic import Critic
from ou_noise import OUNoise


class DDPG:

    def __init__(self, env, state_item_num, action_item_num, emb_dim, batch_size, tau, actor_lr, critic_lr, gamma):

        sess = tf.Session()

        s_dim = emb_dim * state_item_num
        a_dim = emb_dim * action_item_num
        self.actor = Actor(sess, s_dim, a_dim, batch_size, emb_dim, action_item_num, tau, actor_lr)
        self.critic = Critic(sess, s_dim, a_dim, self.actor.get_num_trainable_vars(), gamma, tau, critic_lr)
        self.exploration_noise = OUNoise(a_dim)

    @staticmethod
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

    def train(self, replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
        if replay_buffer.size() < batch_size:
            pass
        samples = replay_buffer.sample_batch(batch_size)
        state_batch = np.asarray([_[0] for _ in samples])
        action_batch = np.asarray([_[1] for _ in samples])
        reward_batch = np.asarray([_[2] for _ in samples])
        n_state_batch = np.asarray([_[3] for _ in samples])

        # calculate predicted q value
        action_weights = actor.predict_target(state_batch)
        n_action_batch = self.gene_actions(item_space, action_weights, action_len)
        target_q_batch = critic.predict_target(n_state_batch.reshape((-1, s_dim)),
                                               n_action_batch.reshape((-1, a_dim)))
        y_batch = []
        for i in range(batch_size):
            y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])

        # train critic
        q_value, critic_loss, _ = critic.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))

        # train actor
        action_weight_batch_for_gradients = actor.predict(state_batch)
        action_batch_for_gradients = self.gene_actions(item_space, action_weight_batch_for_gradients, action_len)
        a_gradient_batch = critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))
        actor.train(state_batch, a_gradient_batch[0])

        # update target networks
        actor.update_target_network()
        critic.update_target_network()

        return np.amax(q_value), critic_loss
