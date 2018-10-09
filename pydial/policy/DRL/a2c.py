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
Implementation of A2C -  Advantage Actor Critic

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

# ===========================
#   Advantage Actor Critic
# ===========================

class A2CNetwork(object):
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

        if self.h2_size > 0:
            # value function
            W_value = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
            b_value = tf.Variable(0.0 * tf.ones([self.h2_size]))
            h_value = tf.nn.relu6(tf.matmul(h_fc1, W_value) + b_value)

            W_value = tf.Variable(tf.truncated_normal([self.h2_size, 1], stddev=0.01))
            b_value = tf.Variable(0.0 * tf.ones([1]))
            self.value = tf.matmul(h_value, W_value) + b_value

            # policy function
            W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
            b_policy = tf.Variable(0.0 * tf.ones([self.h2_size]))
            h_policy = tf.nn.relu6(tf.matmul(h_fc1, W_policy) + b_policy)

            W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
            b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

            # prevent problem when calling log(self.policy)
            if self.is_training:
                self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001

                # scale 0.2 doesn't help too much, TODO: delete next line
                #self.policy = tf.nn.softmax(0.2 * (tf.matmul(h_policy, W_policy) + b_policy)) + 0.00001
            else:
                self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001

        else:  # 1 hidden layer
            # value function
            W_value = tf.Variable(tf.truncated_normal([self.h1_size, 1], stddev=0.01))
            b_value = tf.Variable(0.0 * tf.ones([1]))
            self.value = tf.matmul(h_fc1, W_value) + b_value

            # policy function
            W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.a_dim], stddev=0.01))
            b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

            # prevent problem when calling log(self.policy)
            if self.is_training:
                self.policy = tf.nn.softmax(tf.matmul(h_fc1, W_policy) + b_policy) + 0.00001

                # scale 0.2 doesn't help too much, TODO: delete next line
                # self.policy = tf.nn.softmax(0.2 * (tf.matmul(h_policy, W_policy) + b_policy)) + 0.00001
            else:
                self.policy = tf.nn.softmax(tf.matmul(h_fc1, W_policy) + b_policy) + 0.00001

        # all parameters
        self.vars = tf.trainable_variables()

        # Reinforcement Learning
        #Only the worker network need ops for loss functions and gradient updating.
        self.actions_onehot = self.actions
        self.target_v = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.weights = tf.placeholder(tf.float32, [None])

        # eq. 3.3 from Jason's paper
        self.rho_forward = tf.placeholder(tf.float32, [None])
        #self.rho_whole = tf.placeholder(tf.float32, [None])

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        #Loss functions
        self.value_diff = self.rho_forward * tf.square(self.target_v - tf.reshape(self.value, [-1]))
        #self.value_diff = self.clipped_error(self.value_diff)
        self.value_diff = tf.clip_by_value(self.value_diff, -2, 2)
        self.value_loss = 0.5 * tf.reduce_sum(self.value_diff)

        self.policy_diff = tf.log(self.responsible_outputs) * self.advantages * self.weights
        #self.policy_diff = self.clipped_error(self.policy_diff)
        self.policy_diff = tf.clip_by_value(self.policy_diff, -20, 20)
        self.policy_loss = -tf.reduce_sum(self.policy_diff)

        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        #self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1


        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # add clipping too!
        # clipping
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
        self.optimize = self.optimizer.apply_gradients(capped_gvs)

        #self.optimize = self.optimizer.minimize(self.loss)

        ############################
        # Supervised Learning
        ############################
        self.policy_y = tf.placeholder(tf.int64, [None])
        self.policy_y_one_hot = tf.one_hot(self.policy_y, self.a_dim, 1.0, 0.0, name='policy_y_one_hot')

        self.loss_sl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.policy, labels=self.policy_y_one_hot))
        
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars
                                if 'bias' not in v.name]) * 0.001

        self.loss_combined = self.loss_sl + self.lossL2

        self.optimizer_sl = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize_sl = self.optimizer_sl.minimize(self.loss_combined)

        self.policy_picked = tf.argmax(self.policy,1)

        correct_prediction = tf.equal(tf.argmax(self.policy,1), self.policy_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #self.params = tf.trainable_variables()
        self.params = [v.name for v in tf.trainable_variables()]
        ############################
        ############################


        ############################
        # Supervised Learning + Reinforcement Learning
        ############################
        self.loss_all = self.loss + self.loss_sl

        self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate)

        # clipping
        gvs = self.optimizer_all.compute_gradients(self.loss_all)
        capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
        self.optimize_all = self.optimizer.apply_gradients(capped_gvs)

        ############################
        ############################

    def getPolicy(self, inputs):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs,
        })

    def getLoss(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward
        })


    def train(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss, self.optimize], feed_dict={
            self.inputs:     inputs,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward
        })


    def train_SL(self, inputs, labels):
        #return self.sess.run([self.loss_sl, self.optimize_sl], feed_dict={
        return self.sess.run([self.loss_combined, self.optimize_sl], feed_dict={
            self.inputs:     inputs,
            self.policy_y:   labels
        })

    """
    def train_SL_RL(self, inputs_RL, actions, discounted_rewards, advantages, weights, rho_forward, inputs_SL, labels_SL, self.optimizer_all):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss, self.optimize], feed_dict={
            self.inputs:     inputs_RL,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward
            #self.rho_whole: rho_whole
        })
    """

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

    def predict_value(self, inputs):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs
        })

    def predict_action_value(self, inputs):
        return self.sess.run([self.policy, self.value], feed_dict={
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
        print 'Saving a2c-network... ', save_filename
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_filename)

    def clipped_error(self, x): 
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

