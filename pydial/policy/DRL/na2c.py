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

import numpy as np
import tensorflow as tf

import trpo_utils.distribution.utils as utils

# ===========================
#   Advantage Actor Critic
# ===========================

class A2CNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, architecture = 'vanilla', h1_size = 130, h2_size = 50, KL_delta = 0.01, is_training = True):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.is_training = is_training

        self.KL_delta = KL_delta

        #Input and hidden layers
        self.inputs_sl  = tf.placeholder(tf.float32, [None, self.s_dim])
        self.inputs_rl  = tf.placeholder(tf.float32, [None, self.s_dim])
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])

        def build_model(inputs):
            W_fc1 = tf.get_variable(
                        name='w1',
                        initializer=tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01)
                    )
            b_fc1 = tf.get_variable(
                        name='b1',
                        initializer=0.0 * tf.ones([self.h1_size])
                    )
            h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1, name='h1')

            if h2_size > 0:
                # value function
                W_fc2_v = tf.get_variable(
                            name='w2_v',
                            initializer=tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01)
                          )
                b_fc2_v = tf.get_variable(
                            name='b2_v',
                            initializer=0.0 * tf.ones([self.h2_size])
                          )
                h_fc2_v = tf.nn.relu(tf.matmul(h_fc1, W_fc2_v) + b_fc2_v, name='h2')

                W_value = tf.get_variable(
                            name='w3_v',
                            initializer=tf.truncated_normal([self.h2_size, 1], stddev=0.01)
                          )
                b_value = tf.get_variable(
                            name='b3_v',
                            initializer=0.0 * tf.ones([1])
                          )
                value = tf.matmul(h_fc2_v, W_value) + b_value

                # policy function
                W_fc2_p = tf.get_variable(
                            name='w2_p',
                            initializer=tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01)
                          )
                b_fc2_p = tf.get_variable(
                            name='b2_p',
                            initializer=0.0 * tf.ones([self.h2_size])
                          )
                h_fc2_p = tf.nn.relu(tf.matmul(h_fc1, W_fc2_p) + b_fc2_p, name='h3')

                W_policy = tf.get_variable(
                            name='w3_p',
                            initializer=tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01)
                          )
                b_policy = tf.get_variable(
                            name='b3_p',
                            initializer=0.0 * tf.ones([self.a_dim])
                           )

                policy = tf.nn.softmax(tf.matmul(h_fc2_p, W_policy) + b_policy)

            else:  # 1 hidden layer
                # value function
                W_value = tf.get_variable(
                            name='w2_v',
                            initializer=tf.truncated_normal([self.h1_size, 1], stddev=0.01)
                          )
                b_value = tf.get_variable(
                            name='b2_v',
                            initializer=0.0 * tf.ones([1])
                          )
                value = tf.matmul(h_fc1, W_value) + b_value

                # policy function
                W_policy = tf.get_variable(
                            name='w2_p',
                            initializer=tf.truncated_normal([self.h1_size, self.a_dim], stddev=0.01)
                          )
                b_policy = tf.get_variable(
                            name='b2_p',
                            initializer=0.0 * tf.ones([self.a_dim])
                           )
                policy = tf.nn.softmax(tf.matmul(h_fc1, W_policy) + b_policy)
            
            return value, policy

        with tf.variable_scope('a2c') as scope:        
            self.value_sl, self.policy_sl = build_model(self.inputs_sl)
            scope.reuse_variables()
            self.value_rl, self.policy_rl = build_model(self.inputs_rl)

        # all parameters
        self.vars = tf.trainable_variables()

        # only policy parameters
        self.policy_vars = []
        for i in self.vars:
            if '_p' in i.name or 'w1' in i.name or 'b1' in i.name:
                self.policy_vars.append(i)

        with tf.variable_scope('avg_a2c') as scope:        
            self.value_avg_sl, self.policy_avg_sl = build_model(self.inputs_sl)
            scope.reuse_variables()
            self.value_avg_rl, self.policy_avg_rl = build_model(self.inputs_rl)

        # avg network all parameters
        self.avg_vars = tf.trainable_variables()[len(self.vars):]

        # only avg policy parameters
        self.avg_policy_vars = []
        for i in self.avg_vars:
            if '_p' in i.name or 'w1' in i.name or 'b1' in i.name:
                self.avg_policy_vars.append(i)

        # Op for periodically updating policy network
        self.update_avg_policy_vars = \
            [self.avg_policy_vars[i].assign(\
                tf.multiply(self.policy_vars[i], self.tau) + tf.multiply(self.avg_policy_vars[i], 1. - self.tau))\
                    for i in range(len(self.avg_policy_vars))]

        # Reinforcement Learning
        #Only the worker network need ops for loss functions and gradient updating.
        self.actions_onehot = self.actions
        self.target_v = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.weights = tf.placeholder(tf.float32, [None])

        # eq. 3.3 from Jason's paper
        self.rho_forward = tf.placeholder(tf.float32, [None])

        self.responsible_outputs = tf.reduce_sum(self.policy_rl * self.actions_onehot, [1])

        #Loss functions
        self.value_diff = self.rho_forward * tf.square(self.target_v - tf.reshape(self.value_rl, [-1]))
        self.value_loss = 0.5 * tf.reduce_sum(self.value_diff)

        ### Policy loss with KL constraint ###
        self.policy_diff = tf.log(self.responsible_outputs + 0.00001) * self.advantages * self.weights
        self.policy_loss = -tf.reduce_sum(self.policy_diff)

        ### KL divergence of 'policy params' and 'average policy params' ###
        self.KL = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=self.policy_rl, labels=np.array(self.policy_rl)/np.array(self.policy_avg_rl)+0.00001))

        policy_g = tf.gradients(self.policy_loss, self.policy_vars)
        KL_k = tf.gradients(-self.KL, self.policy_vars)

        shapes = map(utils.var_shape, self.vars)
        print shapes
        
        kg_dot = tf.reduce_sum([tf.reduce_sum(gp * kp) for (gp, kp) in zip(policy_g, KL_k)] )
        kk_dot = tf.reduce_sum([tf.reduce_sum(kp * kp) for kp in KL_k])

        """ No bool usage in tensorflow...
        if kk_dot > 0:
            k_factor = tf.maximum(0, (tf.subtract(kg_dot, self.KL_delta) / kk_dot))
        else:
            k_factor = 0
        """
        def returnNull():
            return tf.constant(0.0)
        def returnValidNum():
            return tf.maximum(tf.constant(0.0), (tf.subtract(kg_dot, self.KL_delta) / kk_dot))
    
        k_factor = tf.cond(kk_dot > 0, returnValidNum, returnNull)

        z = [gp - tf.multiply(k_factor, kp) for kp, gp in zip(KL_k, policy_g)]

        self.policy_KL_loss = 0
        for v, zp in zip(self.policy_vars, z):
            self.policy_KL_loss += tf.reduce_sum(v * zp)  # following from the chains rule (Wang 2016)

        # entropy loss
        self.entropy = - tf.reduce_sum(self.policy_rl * tf.log(self.policy_rl+0.00001))

        # total loss
        #self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1
        self.loss = 0.5 * self.value_loss + self.policy_KL_loss - self.entropy * 0.1

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # clipping
        #gvs = self.optimizer.compute_gradients(self.loss)
        #capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
        gs = tf.gradients(self.loss, self.vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in zip(gs,self.vars)]

        self.optimize = self.optimizer.apply_gradients(capped_gvs)


    def getPolicy(self, inputs):
        return self.sess.run(self.policy_rl, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl:  inputs,
        })

    def getLoss(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):#, rho_whole):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss], feed_dict={
            self.inputs_rl:     inputs,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward,
            self.inputs_sl:  inputs,
            #self.rho_whole: rho_whole
        })

    def train(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):#, rho_whole):
        return self.sess.run([self.value_loss, self.policy_loss, self.KL, self.policy_KL_loss, self.entropy, self.loss, self.optimize], feed_dict={
            self.inputs_rl:     inputs,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward,
            self.inputs_sl:  inputs,
            #self.rho_whole: rho_whole
        })

    def train_SL(self, inputs, labels):
        #return self.sess.run([self.loss_sl, self.optimize_sl], feed_dict={
        return self.sess.run([self.loss_sl_combined, self.optimize_sl], feed_dict={
            self.inputs_rl:  inputs,
            self.inputs_sl:  inputs,
            self.policy_y:   labels
        })

    def train_SL_RL(self, inputs_RL, actions, discounted_rewards, advantages, weights, rho_forward, inputs_SL, labels_SL):
        return self.sess.run([self.value_loss, self.policy_loss, self.KL, self.policy_KL_loss, self.entropy, self.loss, self.loss_sl_combined, self.loss_all, self.optimize_all], feed_dict={
            self.inputs_rl:  inputs_RL,
            self.actions:    actions,
            self.target_v:   discounted_rewards,
            self.advantages: advantages,
            self.weights:    weights,
            self.rho_forward: rho_forward,
            self.inputs_sl:  inputs_SL,
            self.policy_y:   labels_SL
        })


    def error(self, inputs, labels):
        return self.sess.run(self.loss_sl, feed_dict={
            self.inputs_rl:     inputs,
            self.inputs_sl:     inputs,
            self.policy_y:   labels
        })

    def predict_action(self, inputs):
        return self.sess.run(self.policy_picked, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl: inputs
        })


    def predict_policy(self, inputs):
        return self.sess.run(self.policy_rl, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl: inputs
        })

    def predict_value(self, inputs):
        return self.sess.run(self.value_rl, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl: inputs
        })

    def predict_action_value(self, inputs):
        return self.sess.run([self.policy_rl, self.value_rl], feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl: inputs
        })


    def get_curr_network_var(self, inputs):
        return self.sess.run(self.vars, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl:  inputs,
        })

    def get_avg_network_var(self, inputs):
        return self.sess.run(self.avg_vars, feed_dict={
            self.inputs_rl: inputs,
            self.inputs_sl:  inputs,
        })

    def update_avg_network(self):
        self.sess.run(self.update_avg_policy_vars)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully loaded:", load_filename
        except:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving a2c-network...'
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_filename)

    def clipped_error(self, x): 
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

