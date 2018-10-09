import numpy as np
import tensorflow as tf
import random
import scipy.signal

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32

def discount(x, gamma):
    """
    scipy.signal.lfilter(b, a, x, axis=-1, zi=None)[source]
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                      - a[1]*y[n-1] - ... - a[N]*y[n-N]
    :param x:
    :param gamma:
    :return:
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(axis=0, values=[tf.reshape(grad, [np.prod(var_shape(v))])
                         for (grad, v) in zip(grads, var_list)])

def flatparam(var_list):
    return tf.concat(axis=0, values=[tf.reshape(v, [np.prod(var_shape(v))])
                         for v in var_list])

# set theta
class SetFromFlat(object):
    def __init__(self, var_list, session):
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        self.sess = session
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.sess.run(self.op, feed_dict={self.theta: theta})

# get theta
class GetFlat(object):
    def __init__(self, var_list, session):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = session

    def __call__(self):
        return self.op.eval(session=self.sess)


def slice_2d(x, inds0, inds1):
    # assume that a path have 1000 vector, then ncols=action dims, inds0=1000,inds1=
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


# def linesearch(f, x, fullstep, expected_improve_rate):
#     accept_ratio = .1
#     max_backtracks = 10
#     fval, old_kl, entropy = f(x)
#     for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
#         xnew = x + stepfrac * fullstep
#         newfval, new_kl, new_ent= f(xnew)
#         # actual_improve = newfval - fval # minimize target object
#         # expected_improve = expected_improve_rate * stepfrac
#         # ratio = actual_improve / expected_improve
#         # if ratio > accept_ratio and actual_improve > 0:
#         #     return xnew
#         if newfval<fval and new_kl<=pms.max_kl:
#             return xnew
#     return x

def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval, old_kl, entropy = f(x)
    max_kl = 0.01
    for (_n_backtracks, stepfrac) in enumerate(.3**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, new_kl, new_ent= f(xnew)
        # actual_improve = newfval - fval # minimize target object
        # expected_improve = expected_improve_rate * stepfrac
        # ratio = actual_improve / expected_improve
        # if ratio > accept_ratio and actual_improve > 0:
        #     pms.max_kl *= 1.002
        #     return xnew
        if newfval<fval and new_kl<=0.01:
            max_kl *=1.002
            return xnew
    return x

def linesearch_parallel(f, x, fullstep, expected_improve_rate):
    fval, old_kl, entropy = f(x)
    xnew = x - fullstep
    newfval, new_kl, new_ent = f(xnew)
    max_kl = 0.01
    if newfval < fval and new_kl <= max_kl:
        max_kl *= 1.002
        return xnew
    else:
        f(x)
        return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def countMatrixMultiply(matrix):
    result_end = []
    for j in matrix:
        result = 1.0
        for i in j:
            result *= i
        result_end.append(result)
    return np.array(result_end)

def kl_sym(old_dist_means, old_dist_logstds, new_dist_means, new_dist_logstds):
    old_std = tf.exp(old_dist_logstds)
    new_std = tf.exp(new_dist_logstds)
    # means: (N*A)
    # std: (N*A)
    # formula:
    # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
    # ln(\sigma_2/\sigma_1)
    numerator = tf.square(old_dist_means - new_dist_means) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2 * tf.square(new_std) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + new_dist_logstds - old_dist_logstds)

def kl_sym_gradient(old_dist_means, old_dist_logstds, new_dist_means, new_dist_logstds):
    old_std = tf.exp(old_dist_logstds)
    new_std = tf.exp(new_dist_logstds)
    numerator = tf.square(tf.stop_gradient(new_dist_means) - new_dist_means) + \
                tf.square(tf.stop_gradient(new_std)) - tf.square(new_std)


    denominator = 2 * tf.square(new_std) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + new_dist_logstds - tf.stop_gradient(new_dist_logstds))

