import tensorflow as tf
import numpy as np


class DiagonalGaussian(object):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["log_std"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, -1)

    def likelihood_ratio_sym(self, x_var, new_dist_info_vars, old_dist_info_vars):
        """
        \frac{\pi_\theta}{\pi_{old}}
        :param x_var: actions
        :param new_dist_info_vars: means + logstds
        :param old_dist_info_vars: old_means + old_logstds
        :return:
        """
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        \frac{1}{(2\pi)^{\frac{n}{2}}\sigma_\theta}exp(-(\frac{a-\mu_{\pi_\theta}}{2\sigma_\theta})^2)
        :param x_var:
        :param dist_info_vars:
        :return:
        """
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]
        zs = (x_var - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, -1) - \
               0.5 * tf.reduce_sum(tf.square(zs), -1) - \
               0.5 *means.get_shape()[-1].value * np.log(2 * np.pi)

    def kl_sym_firstfixed(self, old_dist_info_vars):
        mu = old_dist_info_vars["mean"]
        logstd = old_dist_info_vars["log_std"]
        mu1 , logstd1 = map(tf.stop_gradient , [mu , logstd])
        mu2 , logstd2 = mu , logstd

        return self.kl_sym(dict(mean=mu1, log_std=logstd1), dict(mean=mu2, log_std=logstd2))

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["log_std"]
        return tf.reduce_sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)))

    @property
    def dist_info_keys(self):
        return ["mean", "log_std"]
