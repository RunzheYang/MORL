from utils import *
#from DRL.trpo_utils.utils import *
import tensorflow as tf
import numpy as np


class DiagonalCategory(object):
    def __init__(self, dim=0):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        return tf.reduce_mean(old_dist_info_vars * tf.log((old_dist_info_vars + 1e-8) / (new_dist_info_vars + 1e-8)))

    def likelihood_ratio_sym(self, x_var, new_dist_info_vars, old_dist_info_vars):
        """
        \frac{\pi_\theta}{\pi_{old}}
        :param x_var: actions
        :param new_dist_info_vars: means + logstds
        :param old_dist_info_vars: old_means + old_logstds
        :return:
        """
        N = tf.shape(x_var)[0]
        p_n = slice_2d(new_dist_info_vars, tf.range(0, N), x_var)
        oldp_n = slice_2d(old_dist_info_vars, tf.range(0, N), x_var)
        return p_n / oldp_n

    def entropy(self, dist_infos):
        return tf.reduce_mean(-dist_infos * tf.log(dist_infos + 1e-8))
