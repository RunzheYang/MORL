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
Implementation of DQN -  Deep Q Network with BBQ

The algorithm is developed with  Tensorflow

Author: Chris Tegho - Original Script by Pei-Hao Su
"""
import tensorflow as tf
import numpy as np

# ===========================
#   Deep Q Network -  with Bayes By Backprop
# ===========================
class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, \
                    num_actor_vars, architecture = 'duel', h1_size = 130, h2_size = 50, \
            n_samples = 1, batch_size = 32, sigma_prior = 0.5, n_batches = 1000.0, \
            stddev_var_mu = 0.01,  stddev_var_logsigma = 0.1, mean_log_sigma = 0.1, importance_sampling = False, \
            alpha_divergence = False, alpha = 1.0, sigma_eps = 1.0):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.sigma_prior = sigma_prior
        self.n_batches = n_batches
        self.stddev_var_mu = stddev_var_mu
        self.stddev_var_logsigma = stddev_var_logsigma
        self.mean_log_sigma = mean_log_sigma
        self.importance_sampling = importance_sampling
        self.alpha_divergence = alpha_divergence
        self.alpha = alpha
        self.sigma_eps = sigma_eps

            # Create the deep Q network
        self.inputs, self.action, self.log_qw, self.log_pw, self.h_samples, self.Qout, self.Variance, self.Mean, self.log_pwi, self.log_qwi = \
            self.create_bbq_network(self.architecture, self.h1_size, self.h2_size, self.n_samples, self.sigma_prior, \
            self.stddev_var_mu,  self.stddev_var_logsigma, self.mean_log_sigma, self.sigma_eps)
        self.network_params = tf.trainable_variables() #remove self.variance ct506

        # Target Network
        self.target_inputs, self.target_action, self.tlog_qw, self.tlog_pw, self.th_samples, self.target_Qout, self.tVariance, self.tMean, self.tlog_pwi, self.tlog_qwi = self.create_bbq_network(self.architecture, self.h1_size, self.h2_size, self.n_samples, self.sigma_prior, self.stddev_var_mu,  self.stddev_var_logsigma, self.mean_log_sigma, self.sigma_eps)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]   #remove self.variance ct506

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])
        self.episodecount = tf.placeholder(tf.float32, shape=None)
        self.log_likelihood = 0.0
        self.pi_i = (1.0 / self.n_batches) #(2**(np.float(self.n_batches)-1-self.episodecount)) / (2**(np.float(self.n_batches))-1)

        actions_one_hot = self.action
        self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1), [self.batch_size, 1])
        self.loss = 0.

        if (self.importance_sampling):
            self.loss_sum = 0.
            for i in range(self.n_samples):
                pred_q_i_ =  tf.reshape(tf.reduce_sum(self.h_samples[i] * actions_one_hot, axis=1), [self.batch_size,1])
                pred_q_i = (self.sampled_q - pred_q_i_)**2
            	loss_i_ = tf.reduce_sum((self.pi_i) * (self.log_qwi[i] - self.log_pwi[i])) + tf.reduce_sum(pred_q_i) / float(self.batch_size)
                if (i == 0):
                    loss_i = loss_i_
                    loss_i = tf.reshape(loss_i, [1, 1])
                else:
                    loss_i = tf.concat(axis=0, values=[loss_i, tf.reshape(loss_i_, [1,1])])
	    #self.ct506 = loss_i
	    self.log_likelihood +=  tf.reduce_sum(pred_q_i) / float(self.batch_size)
            loss_i_minus_max = loss_i - tf.reduce_max(loss_i, axis=0, keep_dims=False)
            #self.ct5066 = loss_i_minus_max
            for i in range(self.n_samples):
                explossi = tf.exp(loss_i_minus_max[i])
                self.loss_sum += explossi
                self.loss += explossi*loss_i[i]
	    #self.ct507 = explossi
	    #self.ct508 = self.loss_sum
	    #self.ct510 = self.loss
            self.loss /= self.loss_sum

	    #self.ct509 = loss_i
	    #self.ct511 = self.loss

        elif (self.alpha_divergence):
            #loss_i = tf.Variable(tf.zeros([self.n_samples, self.batch_size,1]))
            for i in range(self.n_samples):
                pred_q_i_ =  tf.reshape(tf.reduce_sum(self.h_samples[i] * actions_one_hot, axis=1), [self.batch_size,1])
                # the likelihood using target weights
                pred_q_i = tf.multiply(((self.sampled_q - pred_q_i_)**2),-self.alpha)
                #self.ct507 = (self.sampled_q - pred_q_i_)**2
                if (i == 0):
                    loss_i = pred_q_i
                    loss_i = tf.reshape(loss_i, [1, self.batch_size,1])
                else:
                    loss_i = tf.concat(axis=0, values=[loss_i, tf.reshape(pred_q_i, [1, self.batch_size,1])])

            #self.ct506 = loss_i
            #self.ct5066 = (self.sampled_q - pred_q_i_)**2
            self.log_likelihood = tf.reduce_logsumexp(loss_i, axis=0, keep_dims=True)
            #self.ct508 = self.log_likelihood
            #self.ct509 = tf.reduce_logsumexp(loss_i)
            self.log_likelihood /= self.n_samples
            self.log_likelihood = (-1./self.alpha)*tf.reduce_sum(self.log_likelihood)

            #self.ct511 = tf.reduce_sum(self.ct507)

            #self.log_likelihood /= self.n_samples
            self.loss = tf.reduce_sum(self.pi_i * (self.log_qw - self.log_pw)) + self.log_likelihood / float(self.batch_size)

        elif (False):
            self.pi_i = 1./(2**(self.episodecount)-2**(self.episodecount-np.float(self.n_batches)))
            for i in range(self.n_samples):
                pred_q_i =  tf.reshape(tf.reduce_sum(self.h_samples[i] * actions_one_hot, axis=1), [64,1])
                # the likelihood using target weights
                #self.log_likelihood += tf.reduce_sum(self.log_gaussian(self.sampled_q, pred_q_i, self.sigma_prior))
                self.log_likelihood += (self.sampled_q - pred_q_i)**2

            self.log_likelihood /= self.n_samples
            self.log_likelihood = tf.reduce_sum(self.log_likelihood)
            self.loss = tf.reduce_sum((self.pi_i) * (self.log_qw - self.log_pw)) + self.log_likelihood / float(self.batch_size)
        else:
            for i in range(self.n_samples):
                pred_q_i =  tf.reshape(tf.reduce_sum(self.h_samples[i] * actions_one_hot, axis=1), [self.batch_size,1])
                # the likelihood using target weights
                #self.log_likelihood += tf.reduce_sum(self.log_gaussian(self.sampled_q, pred_q_i, self.sigma_prior))
                self.log_likelihood += (self.sampled_q - pred_q_i)**2
	    #self.ct508 = self.h_samples
            self.log_likelihood /= self.n_samples
	    #self.ct509 = self.log_likelihood
            self.log_likelihood = tf.reduce_sum(self.log_likelihood)
	    #self.ct510 = tf.reduce_sum(self.ct508)
            self.loss = tf.reduce_sum(self.pi_i * (self.log_qw - self.log_pw)) + self.log_likelihood / float(self.batch_size)
	    #self.ct511 = self.pi_i * (self.log_qw - self.log_pw)
        self.diff = self.sampled_q - self.pred_q
        #self.ct510 = self.log_likelihood
        self.KL = (self.pi_i * (self.log_qw - self.log_pw))

        # Define loss and optimization Op
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

    def create_bbq_network(self, architecture = 'duel', h1_size = 130, h2_size = 50, n_samples = 1, sigma_prior = 0.05, \
	stddev_var_mu = 0.01,  stddev_var_logsigma = 0.01, mean_log_sigma = 1.0, sigma_eps = 1.0):

        inputs = tf.placeholder(tf.float32, [None, self.s_dim], name = 'inputs')
        action = tf.placeholder(tf.float32, [None, self.a_dim], name = 'action')

        W_fc1_mu = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=stddev_var_mu))
        b_fc1_mu = tf.Variable(tf.zeros([h1_size]))
        W_fc1_logsigma = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], mean=mean_log_sigma, stddev=stddev_var_logsigma))
        b_fc1_logsigma = tf.Variable(tf.zeros([h1_size]))

        W_fc2_mu = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=stddev_var_mu))
        b_fc2_mu = tf.Variable(tf.zeros([h2_size]))
        W_fc2_logsigma = tf.Variable(tf.truncated_normal([h1_size, h2_size], mean=mean_log_sigma, stddev=stddev_var_logsigma))
        b_fc2_logsigma = tf.Variable(tf.zeros([h2_size]))

        W_out_mu = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=stddev_var_mu))
        b_out_mu = tf.Variable(tf.zeros([self.a_dim]))
        W_out_logsigma = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], mean=mean_log_sigma, stddev=stddev_var_logsigma))
        b_out_logsigma = tf.Variable(tf.zeros([self.a_dim]))

        # building the objective
        # remember, we're evaluating by samples
        log_pw, log_qw = 0., 0.
        h_samples = []
        Qout = tf.Variable(tf.zeros([self.a_dim]))
        log_pwi, log_qwi = [], []


        for _ in xrange(n_samples):
            epsilon_w1 = tf.truncated_normal((self.s_dim, h1_size), mean=0., stddev=sigma_eps) #ct506
            epsilon_b1 = tf.truncated_normal((h1_size,), mean=0., stddev=sigma_eps)
            W1 = W_fc1_mu + tf.multiply(tf.log(1. + tf.exp(W_fc1_logsigma)), epsilon_w1)
            b1 = b_fc1_mu + tf.multiply(tf.log(1. + tf.exp(b_fc1_logsigma)), epsilon_b1)

            epsilon_w2 = tf.truncated_normal((h1_size, h2_size), mean=0., stddev=sigma_eps)
            epsilon_b2 = tf.truncated_normal((h2_size,), mean=0., stddev=sigma_eps)
            W2 = W_fc2_mu + tf.multiply(tf.log(1. + tf.exp(W_fc2_logsigma)), epsilon_w2)
            b2 = b_fc2_mu + tf.multiply(tf.log(1. + tf.exp(b_fc2_logsigma)), epsilon_b2)

            epsilon_wout = tf.truncated_normal((h2_size, self.a_dim), mean=0., stddev=sigma_eps)
            epsilon_bout = tf.truncated_normal((self.a_dim,), mean=0., stddev=sigma_eps)
            Wout = W_out_mu + tf.multiply(tf.log(1. + tf.exp(W_out_logsigma)), epsilon_wout) #t506
            bout = b_out_mu + tf.multiply(tf.log(1. + tf.exp(b_out_logsigma)), epsilon_bout)

            a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
            a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
            h = tf.matmul(a2, Wout) + bout

            sample_log_pw, sample_log_qw = 0., 0.

            for W, b, W_mu, W_logsigma, b_mu, b_logsigma in [(W1, b1, W_fc1_mu, W_fc1_logsigma, b_fc1_mu, b_fc1_logsigma),
                                                            (W2, b2, W_fc2_mu, W_fc2_logsigma, b_fc2_mu, b_fc2_logsigma),
                                (Wout, bout, W_out_mu, W_out_logsigma, b_out_mu, b_out_logsigma)]:


                # first weight prior
                sample_log_pw += tf.reduce_sum(self.log_gaussian(W, 0., sigma_prior))
                sample_log_pw += tf.reduce_sum(self.log_gaussian(b, 0., sigma_prior))

                # then approximation
                sample_log_qw += tf.reduce_sum(self.log_gaussian(W, W_mu, tf.log(1. + tf.exp(W_logsigma))))
                sample_log_qw += tf.reduce_sum(self.log_gaussian(b, b_mu, tf.log(1. + tf.exp(b_logsigma)))) #ct506

            h_samples += [h]
            Qout += h
            log_pw += sample_log_pw
            log_qw += sample_log_qw
            log_pwi += [sample_log_pw]
            log_qwi += [sample_log_qw]

	#self.ct506 = Qout
	#self.ct507 = log_qw- log_pw
        log_qw /= n_samples
        log_pw /= n_samples
        Qout /= n_samples
	#self.ct5066 = Qout

        return inputs, action, log_qw, log_pw, h_samples, Qout, W_fc2_logsigma, W_fc2_mu, log_pwi, log_qwi

    def train(self, inputs, action, sampled_q, episodecount):
        return self.sess.run([self.pred_q, self.optimize, self.loss, self.log_likelihood, self.Variance, self.Mean, self.diff, self.KL], feed_dict={
        self.inputs: inputs,
        self.target_inputs: inputs,
        self.action: action,
        self.target_action: action,
        self.sampled_q: sampled_q,
        self.episodecount : episodecount
        })
    #


    def predict(self, inputs):
        #return self.sess.run(self.pred_q, feed_dict={
        return self.sess.run(self.Qout, feed_dict={
            self.inputs: inputs
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
        #return self.sess.run(self.pred_q, feed_dict={
        return self.sess.run(self.target_Qout, feed_dict={
            self.target_inputs: inputs
        })

    def predict_target_with_action_maxQ(self, inputs):
        #return self.sess.run(self.pred_q_target, feed_dict={
        return self.sess.run(self.action_maxQ_target, feed_dict={
            self.target_inputs: inputs,
            self.inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully loaded:", load_filename
        except:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.saver.save(self.sess, save_filename)

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

    def nonlinearity(self, x):
	    return tf.nn.relu(x)

    def log_gaussian(self, x, mu, sigma):
	    return -0.5 * np.log(2 * np.pi) - tf.log(tf.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

    def log_gaussian_logsigma(self, x, mu, logsigma):
	    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * tf.exp(logsigma))

    def get_random(self, shape, mean, stddev):
	    return tf.random_normal(shape, mean=mean, stddev=stddev)
