import tensorflow as tf


class Critic(object):
    """value function approximator"""

    def __init__(self, sess, state_item_num, action_item_num, emb_dim, num_actor_vars, gamma, tau, learning_rate,
                 scope="critic"):
        self.sess = sess
        self.state_item_num = state_item_num
        self.action_item_num = action_item_num
        self.num_actor_vars = num_actor_vars
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        self.s_dim = emb_dim * state_item_num
        self.a_dim = emb_dim * action_item_num

        with tf.variable_scope(self.scope):
            # estimator critic network
            self.state, self.action, self.q_value, self.len_seq = self._build_net("estimator_critic")
            # self.network_params = tf.trainable_variables()[self.num_actor_vars:]
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator_critic")

            # target critic network
            self.target_state, self.target_action, self.target_q_value, self.target_len_seq = self._build_net(
                "target_critic")
            # self.target_network_params = tf.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assgin(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.a_gradient = tf.gradients(self.q_value, self.action)

    @staticmethod
    def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
            len_seq = tf.placeholder(tf.int64, [None, ], name="critic_len_seq")

            state_ = tf.reshape(state, [-1, self.state_item_num, int(self.s_dim / self.state_item_num)])
            cell = tf.nn.rnn_cell.GRUCell(self.state_item_num,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer()
                                          )
            out_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            out_state = self._gather_last_output(out_state, len_seq)

            inputs = tf.concat([out_state, action], axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            q_value = tf.layers.Dense(1)(layer2)
            return state, action, q_value, len_seq

    def train(self, state, action, predicted_q_value, len_seq):
        return self.sess.run([self.q_value, self.loss, self.optimizer], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.len_seq: len_seq
        })

    # def predict(self, state, action, len_seq):
    #     return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def predict_target(self, state, action, len_seq):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state,
                                                             self.target_action: action,
                                                             self.target_len_seq: len_seq})

    def action_gradients(self, state, action, len_seq):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)
