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
Implementation of eNAC -  Episodic Natural Actor Critic

The algorithm is developed with Tensorflow

Author: Pei-Hao Su
"""
import tensorflow as tf
import numpy as np

import numpy as np
import tensorflow as tf

from random import choice
from time import sleep
from time import time

# =================================
#   Episodic Natural Actor Critic
# =================================

class ENACNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, architecture = 'vanilla', h1_size = 130, h2_size = 50, is_training = True):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.is_training = is_training

        #Input and hidden layers
        self.inputs  = tf.placeholder(tf.float32, [None, self.s_dim])
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])

        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01))
        b_fc1 = tf.Variable(0.0 * tf.ones([self.h1_size]))
        h_fc1 = tf.nn.relu6(tf.matmul(self.inputs, W_fc1) + b_fc1)

        # policy function
        W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
        b_policy = tf.Variable(0.0 * tf.ones([self.h2_size]))
        h_policy = tf.nn.relu6(tf.matmul(h_fc1, W_policy) + b_policy)

        W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

        # prevent problem when calling log(self.policy)
        if self.is_training:
            self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001
        else:
            self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001

        # Reinforcement Learning
        # Only the worker network need ops for loss functions and gradient updating.
        self.actions_onehot = self.actions
        self.advantages = tf.placeholder(tf.float32, [None])

        self.params = tf.trainable_variables()
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        #Loss functions
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs))
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.loss = self.policy_loss - self.entropy #* 0.01

        #self.policy_grads = tf.gradients(self.policy_loss, self.params)
        self.policy_grads = tf.gradients(self.loss, self.params)

        # natural gradient vairables
        self.W_fc1_ng = tf.Variable(tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01))
        self.b_fc1_ng = tf.Variable(0.0 * tf.ones([self.h1_size]))

        self.W_fc2_ng = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
        self.b_fc2_ng = tf.Variable(0.0 * tf.ones([self.h2_size]))

        self.W_policy_ng = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        self.b_policy_ng = tf.Variable(0.0 * tf.ones([self.a_dim]))

        self.natural_grads = [ \
                self.W_fc1_ng, self.b_fc1_ng, \
                self.W_fc2_ng, self.b_fc2_ng, \
                self.W_policy_ng, self.b_policy_ng\
                ]

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.apply_gradients(zip(self.natural_grads, self.params))
        #self.optimize = self.optimizer.minimize(self.loss)

        ############################
        # Supervised Learning
        ############################
        self.policy_y = tf.placeholder(tf.int64, [None])
        self.policy_y_one_hot = tf.one_hot(self.policy_y, self.a_dim, 1.0, 0.0, name='policy_y_one_hot')

        self.loss_sl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.policy, labels=self.policy_y_one_hot))
        self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.params]) * 0.001

        self.loss_sl_combined = ( self.loss_sl + self.lossL2 ) * 10

        self.optimizer_sl = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize_sl = self.optimizer_sl.minimize(self.loss_sl_combined)

        self.policy_picked = tf.argmax(self.policy,1)

        correct_prediction = tf.equal(tf.argmax(self.policy,1), self.policy_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #self.params = tf.trainable_variables()
        #self.params = [v.name for v in tf.trainable_variables()]
        ############################
        ############################

    def train(self, inputs, actions, W_fc1_ng, b_fc1_ng, W_fc2_ng, b_fc2_ng, W_policy_ng, b_policy_ng):
        return self.sess.run([self.policy_loss, self.entropy, self.loss, self.optimize], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.W_fc1_ng:   W_fc1_ng,
            self.b_fc1_ng:   b_fc1_ng,
            self.W_fc2_ng:   W_fc2_ng,
            self.b_fc2_ng:   b_fc2_ng,
            self.W_policy_ng:W_policy_ng,
            self.b_policy_ng:b_policy_ng
        })

    def train2(self, inputs, actions, W_fc1_ng, b_fc1_ng, W_fc2_ng, b_fc2_ng, W_policy_ng, b_policy_ng):
        return self.sess.run([self.policy_loss, self.optimize], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.W_fc1_ng:   W_fc1_ng,
            self.b_fc1_ng:   b_fc1_ng,
            self.W_fc2_ng:   W_fc2_ng,
            self.b_fc2_ng:   b_fc2_ng,
            self.W_policy_ng:W_policy_ng,
            self.b_policy_ng:b_policy_ng
        })

    def train_SL(self, inputs, labels):
        return self.sess.run([self.loss_sl_combined, self.loss_sl, self.lossL2, self.optimize_sl], feed_dict={
            self.inputs:     inputs,
            self.policy_y:   labels
        })

    def get_policy_gradient(self, inputs, actions):
        return self.sess.run(self.policy_grads, feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
        })

    def getPolicy(self, inputs):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs,
        })

    def get_params(self):
        return self.sess.run(self.params)

    def error(self, inputs, labels):
        return self.sess.run([self.loss_sl], feed_dict={
            self.inputs:     inputs,
            self.policy_y:   labels
        })

    def predict_action(self, inputs):
        return self.sess.run(self.policy_picked, feed_dict={
            self.inputs: inputs
        })

    def predict_policy(self, inputs):
        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs
        })

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
        print 'Saving enac-network...'
        #self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_filename)

    def clipped_error(self, x): 
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

class NNFENACNetwork(object):
    def __init__(self, sess, si_state_dim, sd_state_dim, action_dim, learning_rate, tau, num_actor_vars, architecture='vanilla',
                 h1_size=130, h2_size=50, is_training=True, sd_enc_size=40, si_enc_size=80, dropout_rate=0.):
        #super(NNFENACNetwork, self).__init__(sess, si_state_dim + sd_state_dim, action_dim, learning_rate, tau, num_actor_vars,
        #                                     architecture=architecture, h1_size=h1_size, h2_size=h2_size, is_training=is_training)
        self.sess = sess
        self.si_dim = si_state_dim
        self.sd_dim = sd_state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.is_training = is_training

        # Input and hidden layers
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])
        self.inputs = tf.placeholder(tf.float32, [None, self.sd_dim + self.si_dim])
        sd_inputs, si_inputs = tf.split(self.inputs, [self.sd_dim, self.si_dim], 1)

        keep_prob = 1 - dropout_rate

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

        W_fc1 = tf.Variable(tf.truncated_normal([sd_enc_size+si_enc_size, self.h1_size], stddev=0.01))
        b_fc1 = tf.Variable(0.0 * tf.ones([self.h1_size]))
        h_fc1 = tf.nn.relu6(tf.matmul(tf.concat((h_sdfe, h_sife), 1), W_fc1) + b_fc1)

        # policy function
        W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
        b_policy = tf.Variable(0.0 * tf.ones([self.h2_size]))
        h_policy = tf.nn.relu6(tf.matmul(h_fc1, W_policy) + b_policy)

        W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

        # prevent problem when calling log(self.policy)
        if self.is_training:
            self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001
        else:
            self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001

        # Reinforcement Learning
        # Only the worker network need ops for loss functions and gradient updating.
        self.actions_onehot = self.actions
        self.advantages = tf.placeholder(tf.float32, [None])

        self.params = tf.trainable_variables()
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        # Loss functions
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs))
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.loss = self.policy_loss - self.entropy  # * 0.01

        # self.policy_grads = tf.gradients(self.policy_loss, self.params)
        self.policy_grads = tf.gradients(self.loss, self.params)

        # natural gradient vairables
        self.W_fc1_ng = tf.Variable(tf.truncated_normal([sd_enc_size+si_enc_size, self.h1_size], stddev=0.01))
        self.b_fc1_ng = tf.Variable(0.0 * tf.ones([self.h1_size]))

        self.W_fc2_ng = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
        self.b_fc2_ng = tf.Variable(0.0 * tf.ones([self.h2_size]))

        self.W_policy_ng = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        self.b_policy_ng = tf.Variable(0.0 * tf.ones([self.a_dim]))

        self.natural_grads = [ self.W_fc1_ng, self.b_fc1_ng, self.W_fc2_ng, self.b_fc2_ng, self.W_policy_ng,
                               self.b_policy_ng]

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.apply_gradients(zip(self.natural_grads, self.params))
        # self.optimize = self.optimizer.minimize(self.loss)

        ############################
        # Supervised Learning
        ############################
        self.policy_y = tf.placeholder(tf.int64, [None])
        self.policy_y_one_hot = tf.one_hot(self.policy_y, self.a_dim, 1.0, 0.0, name='policy_y_one_hot')

        self.loss_sl = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.policy, labels=self.policy_y_one_hot))
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.params]) * 0.001

        self.loss_sl_combined = (self.loss_sl + self.lossL2) * 10

        self.optimizer_sl = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize_sl = self.optimizer_sl.minimize(self.loss_sl_combined)

        self.policy_picked = tf.argmax(self.policy, 1)

        correct_prediction = tf.equal(tf.argmax(self.policy, 1), self.policy_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, inputs, actions, W_fc1_ng, b_fc1_ng, W_fc2_ng, b_fc2_ng, W_policy_ng, b_policy_ng):
        return self.sess.run([self.policy_loss, self.entropy, self.loss, self.optimize], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.W_fc1_ng:   W_fc1_ng,
            self.b_fc1_ng:   b_fc1_ng,
            self.W_fc2_ng:   W_fc2_ng,
            self.b_fc2_ng:   b_fc2_ng,
            self.W_policy_ng:W_policy_ng,
            self.b_policy_ng:b_policy_ng
        })

    def train2(self, inputs, actions, W_fc1_ng, b_fc1_ng, W_fc2_ng, b_fc2_ng, W_policy_ng, b_policy_ng):
        return self.sess.run([self.policy_loss, self.optimize], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.W_fc1_ng:   W_fc1_ng,
            self.b_fc1_ng:   b_fc1_ng,
            self.W_fc2_ng:   W_fc2_ng,
            self.b_fc2_ng:   b_fc2_ng,
            self.W_policy_ng:W_policy_ng,
            self.b_policy_ng:b_policy_ng
        })

    def train_SL(self, inputs, labels):
        return self.sess.run([self.loss_sl_combined, self.loss_sl, self.lossL2, self.optimize_sl], feed_dict={
            self.inputs:     inputs,
            self.policy_y:   labels
        })

    def get_policy_gradient(self, inputs, actions):
        return self.sess.run(self.policy_grads, feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
        })

    def getPolicy(self, inputs):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs,
        })

    def get_params(self):
        return self.sess.run(self.params)

    def error(self, inputs, labels):
        return self.sess.run([self.loss_sl], feed_dict={
            self.inputs:     inputs,
            self.policy_y:   labels
        })

    def predict_action(self, inputs):
        return self.sess.run(self.policy_picked, feed_dict={
            self.inputs: inputs
        })

    def predict_policy(self, inputs):
        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs
        })

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
        print 'Saving enac-network...'
        #self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_filename)

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false