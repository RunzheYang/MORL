###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

""" 
Implementation of DQN -  Deep Q Network

The algorithm is developed with tflearn + Tensorflow

Author: Pei-Hao Su
"""
import tensorflow as tf

# ===========================
#   Deep Q Network
# ===========================
class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, minibatch_size=64,
                 architecture='duel', h1_size=130, h2_size=50, dropout_rate=0.):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.minibatch_size = minibatch_size

        # Create the deep Q network
        self.inputs, self.action, self.Qout = \
                        self.create_ddq_network(self.architecture, self.h1_size, self.h2_size, dropout_rate=dropout_rate)
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_Qout = \
                        self.create_ddq_network(self.architecture, self.h1_size, self.h2_size, dropout_rate=dropout_rate)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])
        #self.temperature = tf.placeholder(shape=None,dtype=tf.float32)

        # for Boltzman exploration
        #self.softmax_Q = tf.nn.softmax(self.self.Qout/self.temperature)

        # Predicted Q given state and chosed action
        #actions_one_hot = tf.one_hot(self.action, self.a_dim, 1.0, 0.0, name='action_one_hot')
        actions_one_hot = self.action

        if architecture!= 'dip':
            self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1, name='q_acted'),
                                 [self.minibatch_size, 1])
        else:
            self.pred_q = self.Qout #DIP case, not sure if will work

        #self.pred_q = tf.reduce_sum(self.Qout * actions_one_hot, reduction_indices=1, name='q_acted_target')

        #self.a_maxQ = tf.argmax(self.Qout, 1)
        #action_maxQ_one_hot = tf.one_hot(self.a_maxQ, self.a_dim, 1.0, 0.0, name='action_maxQ_one_hot')
        #self.action_maxQ_target = tf.reduce_sum(self.target_Qout * action_maxQ_one_hot, reduction_indices=1, name='a_maxQ_target')

        # Define loss and optimization Op
        self.diff = self.sampled_q - self.pred_q
        self.loss = tf.reduce_mean(self.clipped_error(self.diff), name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        # gs = tf.gradients(self.loss, self.network_params)
        # capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in zip(gs, self.network_params)]
        #
        # self.optimize = self.optimizer.apply_gradients(capped_gvs)

    def create_ddq_network(self, architecture = 'duel', h1_size = 130, h2_size = 50, dropout_rate=0.):
        keep_prob = 1 - dropout_rate
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        if architecture == 'duel':
            W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([h1_size]))
            h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

            # value function
            W_value = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_value = tf.Variable(tf.zeros([h2_size]))
            h_value = tf.nn.relu(tf.matmul(h_fc1, W_value) + b_value)

            W_value = tf.Variable(tf.truncated_normal([h2_size, 1], stddev=0.01))
            b_value = tf.Variable(tf.zeros([1]))
            value_out  = tf.matmul(h_value, W_value) + b_value

            # advantage function
            W_advantage = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([h2_size]))
            h_advantage = tf.nn.relu(tf.matmul(h_fc1, W_advantage) + b_advantage)

            W_advantage = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([self.a_dim]))
            Advantage_out  = tf.matmul(h_advantage, W_advantage) + b_advantage

            Qout = value_out + (Advantage_out - tf.reduce_mean(Advantage_out, axis=1, keep_dims=True))

        elif architecture == 'dip':

            # state network
            W_fc1_s = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
            b_fc1_s = tf.Variable(tf.zeros([h1_size]))
            h_fc1_s = tf.nn.relu(tf.matmul(inputs, W_fc1_s) + b_fc1_s)

            # action network
            W_fc1_a = tf.Variable(tf.truncated_normal([self.a_dim, h1_size], stddev=0.01))
            b_fc1_a = tf.Variable(tf.zeros([h1_size]))
            h_fc1_a = tf.nn.relu(tf.matmul(action, W_fc1_a) + b_fc1_a)

            W_fc2_s = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_fc2_s = tf.Variable(tf.zeros([h2_size]))
            h_fc2_s = tf.nn.relu(tf.matmul(h_fc1_s, W_fc2_s) + b_fc2_s)

            W_fc2_a = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_fc2_a = tf.Variable(tf.zeros([h2_size]))
            h_fc2_a = tf.nn.relu(tf.matmul(h_fc1_a, W_fc2_a) + b_fc2_a)

            Qout = tf.reduce_sum(tf.multiply(h_fc2_s, h_fc2_a), 1)

        else:
            W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([h1_size]))
            h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
            if keep_prob < 1:
                h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_fc2 = tf.Variable(tf.zeros([h2_size]))
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
            if keep_prob < 1:
                h_fc2 = tf.nn.dropout(h_fc2, keep_prob)

            W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_out = tf.Variable(tf.zeros([self.a_dim]))
            Qout = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout

    def train(self, inputs, action, sampled_q):
        return self.sess.run([self.pred_q, self.optimize, self.loss], feed_dict={ #yes, needs to be changed too
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def predict(self, inputs):
        return self.sess.run(self.Qout, feed_dict={
            self.inputs: inputs
        })

    def predict_dip(self, inputs, action):
        return self.sess.run(self.Qout, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    """
    def predict_Boltzman(self, inputs, temperature):
        return self.sess.run(self.softmax_Q, feed_dict={
            self.inputs: inputs
            self.temperature = temperature
        })
    """

    def predict_action(self, inputs):
        return self.sess.run(self.pred_q, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_Qout, feed_dict={
            self.target_inputs: inputs
        })

    def predict_target_dip(self, inputs, action):
        return self.sess.run(self.target_Qout, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def predict_target_with_action_maxQ(self, inputs):
        return self.sess.run(self.action_maxQ_target, feed_dict={
            self.target_inputs: inputs,
            self.inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params) #yes, but no need to change

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, load_filename)
                print "Successfully loaded:", load_filename
            except:
                print "Could not find old network weights"
        else:
            print 'nothing loaded in first iteration'

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, save_filename) # yes but no need to change

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false


class NNFDeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, si_state_dim, sd_state_dim, action_dim, learning_rate, tau, num_actor_vars, minibatch_size=64,
                 architecture='duel', h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0.):
        #super(NNFDeepQNetwork, self).__init__(sess, si_state_dim + sd_state_dim, action_dim, learning_rate, tau, num_actor_vars,
        #                                      minibatch_size=64, architecture='duel', h1_size=130, h2_size=50)
        self.sess = sess
        self.si_dim = si_state_dim
        self.sd_dim = sd_state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.minibatch_size = minibatch_size
        self.sd_enc_size = sd_enc_size
        self.si_enc_size = si_enc_size
        self.dropout_rate = dropout_rate

        # Create the deep Q network
        self.inputs, self.action, self.Qout = \
                        self.create_nnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate)
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_Qout = \
                        self.create_nnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])

        # Predicted Q given state and chosed action
        actions_one_hot = self.action

        if architecture!= 'dip':
            self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1, name='q_acted'),
                                 [self.minibatch_size, 1])
        else:
            self.pred_q = self.Qout

        # Define loss and optimization Op
        self.diff = self.sampled_q - self.pred_q
        self.loss = tf.reduce_mean(self.clipped_error(self.diff), name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

    def create_nnfdq_network(self, h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0.):
        inputs = tf.placeholder(tf.float32, [None, self.sd_dim + self.si_dim])
        keep_prob = 1 - dropout_rate
        sd_inputs, si_inputs = tf.split(inputs, [self.sd_dim, self.si_dim], 1)
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        W_sdfe = tf.Variable(tf.truncated_normal([self.sd_dim, sd_enc_size], stddev=0.01))
        b_sdfe = tf.Variable(tf.zeros([sd_enc_size]))
        h_sdfe = tf.nn.relu(tf.matmul(sd_inputs, W_sdfe) + b_sdfe)
        if keep_prob < 1:
            h_sdfe = tf.nn.dropout(h_sdfe, keep_prob)

        W_sife = tf.Variable(tf.truncated_normal([self.si_dim, si_enc_size], stddev=0.01))
        b_sife = tf.Variable(tf.zeros([si_enc_size]))
        h_sife = tf.nn.relu(tf.matmul(si_inputs, W_sife) + b_sife)
        if keep_prob < 1:
            h_sife = tf.nn.dropout(h_sife, keep_prob)

        W_fc1 = tf.Variable(tf.truncated_normal([sd_enc_size+si_enc_size, h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(tf.concat((h_sdfe, h_sife), 1), W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([h2_size]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.a_dim]))
        Qout = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout

    def predict(self, inputs):
        return self.sess.run(self.Qout, feed_dict={ #inputs where a single flat_bstate
            self.inputs: inputs
        })

    def predict_dip(self, inputs, action):
        return self.sess.run(self.Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs where a single flat_bstate
            self.target_inputs: inputs
        })

    def predict_target_dip(self, inputs, action):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.target_inputs: inputs,
            self.target_action: action
        })

    def train(self, inputs, action, sampled_q):
        return self.sess.run([self.pred_q, self.optimize, self.loss], feed_dict={ #all the inputs are arrays of 64
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, save_filename)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, load_filename)
                print "Successfully loaded:", load_filename
            except:
                print "Could not find old network weights"
        else:
            print 'nothing loaded in first iteration'


class RNNFDeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, si_state_dim, sd_state_dim, action_dim, learning_rate, tau, num_actor_vars, minibatch_size=64,
                 architecture='duel', h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0., slot='si'):
        #super(NNFDeepQNetwork, self).__init__(sess, si_state_dim + sd_state_dim, action_dim, learning_rate, tau, num_actor_vars,
        #                                      minibatch_size=64, architecture='duel', h1_size=130, h2_size=50)
        self.sess = sess
        self.si_dim = si_state_dim
        self.sd_dim = sd_state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.minibatch_size = minibatch_size
        self.sd_enc_size = sd_enc_size
        self.si_enc_size = si_enc_size
        self.dropout_rate = dropout_rate

        # Create the deep Q network
        self.inputs, self.action, self.Qout = \
                        self.create_rnnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate, slot=slot)
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_Qout = \
                        self.create_rnnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate, tn='target', slot=slot)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])

        # Predicted Q given state and chosed action
        actions_one_hot = self.action

        if architecture!= 'dip':
            self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1, name='q_acted'),
                                 [self.minibatch_size, 1])
        else:
            self.pred_q = self.Qout

        # Define loss and optimization Op
        self.diff = self.sampled_q - self.pred_q
        self.loss = tf.reduce_mean(self.clipped_error(self.diff), name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

    #def create_slot_encoder(self):


    def create_rnnfdq_network(self, h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0.,
                              tn='normal', slot='si'):
        inputs = tf.placeholder(tf.float32, [None, self.sd_dim + self.si_dim])
        keep_prob = 1 - dropout_rate
        sd_inputs, si_inputs = tf.split(inputs, [self.sd_dim, self.si_dim], 1)
        action = tf.placeholder(tf.float32, [None, self.a_dim])
        if slot == 'sd':
            sd_inputs = tf.reshape(sd_inputs, (tf.shape(sd_inputs)[0], 1, self.sd_dim))

            #slots encoder
            with tf.variable_scope(tn):
                #try:
                    lstm_cell = tf.nn.rnn_cell.GRUCell(self.sd_enc_size)
                    hidden_state = lstm_cell.zero_state(tf.shape(sd_inputs)[0], tf.float32)
                    _, h_sdfe = tf.nn.dynamic_rnn(lstm_cell, sd_inputs, initial_state=hidden_state)
                #except:
                #    lstm_cell = tf.contrib.rnn.GRUCell(self.sd_enc_size)
                #    hidden_state = lstm_cell.zero_state(tf.shape(sd_inputs)[0], tf.float32)
                #    _, h_sdfe = tf.contrib.rnn.dynamic_rnn(lstm_cell, sd_inputs, initial_state=hidden_state)
        else:
            W_sdfe = tf.Variable(tf.truncated_normal([self.sd_dim, sd_enc_size], stddev=0.01))
            b_sdfe = tf.Variable(tf.zeros([sd_enc_size]))
            h_sdfe = tf.nn.relu(tf.matmul(sd_inputs, W_sdfe) + b_sdfe)
            if keep_prob < 1:
                h_sdfe = tf.nn.dropout(h_sdfe, keep_prob)

        W_sife = tf.Variable(tf.truncated_normal([self.si_dim, si_enc_size], stddev=0.01))
        b_sife = tf.Variable(tf.zeros([si_enc_size]))
        h_sife = tf.nn.relu(tf.matmul(si_inputs, W_sife) + b_sife)
        if keep_prob < 1:
            h_sife = tf.nn.dropout(h_sife, keep_prob)

        W_fc1 = tf.Variable(tf.truncated_normal([sd_enc_size+si_enc_size, h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(tf.concat((h_sdfe, h_sife), 1), W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([h2_size]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.a_dim]))
        Qout = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout

    def predict(self, inputs):
        return self.sess.run(self.Qout, feed_dict={ #inputs where a single flat_bstate
            self.inputs: inputs
        })

    def predict_dip(self, inputs, action):
        return self.sess.run(self.Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs where a single flat_bstate
            self.target_inputs: inputs
        })

    def predict_target_dip(self, inputs, action):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.target_inputs: inputs,
            self.target_action: action
        })

    def train(self, inputs, action, sampled_q):
        return self.sess.run([self.pred_q, self.optimize, self.loss], feed_dict={ #all the inputs are arrays of 64
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, save_filename)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, load_filename)
                print "Successfully loaded:", load_filename
            except:
                print "Could not find old network weights"
        else:
            print 'nothing loaded in first iteration'
