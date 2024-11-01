import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

EPS = 1e-8
def clip_but_pass_gradient2(x, l=EPS):
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((l - x)*clip_low)
def new_relu(x, alpha_actv):
    r = tf.math.reciprocal(clip_but_pass_gradient2(x+alpha_actv,l=EPS))
    return r #+ part_3*0
def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
def mlp_policy(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
#        x = tf.layers.dropout(x,0.1,training=trainp)
    return x
def CNN(x, y,activation=tf.nn.relu, output_activation=None):
    x0 = tf.layers.conv1d(x, filters=256, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    w = tf.layers.dense(y, units=256,activation=tf.nn.sigmoid)
    w = tf.reshape(w,[-1,1,256])
    xw  = tf.multiply(x0, w)
    #    d = tf.math.sqrt(tf.constant([1/64.0]))
#    x0 = tf.layers.conv1d(x, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.tanh)
#    w = tf.layers.dense(y, units=64,activation=tf.nn.sigmoid)
#    w = tf.reshape(w,[-1,4,64])
#    xw  = tf.matmul(x0, w, transpose_b=True)
#    sum_in_rows = d*tf.reduce_sum(xw, 2,keep_dims=True)
    x2 = tf.layers.conv1d(xw, filters=20, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x2, pool_size=720, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return x3_flatten
def CNN_dense(x,activation=tf.nn.leaky_relu, output_activation=None):
    alpha_actv2= tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
    x_input = x[:,0:2160]
    x_input = tf.reshape(x_input,[-1,720,3])
    w_input = x[:,2160:2160+4]
    y = tf.reshape(w_input,[-1,4])
    x00 = new_relu(x_input[:,:,2], alpha_actv2)
    x_input = tf.concat([x_input[:,:,0:2],tf.reshape(x00,[-1,720,1])], axis=-1)
    x0 = tf.layers.conv1d(x_input, filters=256, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    w = tf.layers.dense(y, units=256,activation=tf.nn.sigmoid)
    w = tf.reshape(w,[-1,1,256])
    xw  = tf.multiply(x0, w)
    #    d = tf.math.sqrt(tf.constant([1/64.0]))
#    x0 = tf.layers.conv1d(x, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.tanh)
#    w = tf.layers.dense(y, units=64,activation=tf.nn.sigmoid)
#    w = tf.reshape(w,[-1,4,64])
#    xw  = tf.matmul(x0, w, transpose_b=True)
#    sum_in_rows = d*tf.reduce_sum(xw, 2,keep_dims=True)
    x2 = tf.layers.conv1d(xw, filters=20, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x2, pool_size=720, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return tf.concat([x3_flatten,x[:,2160:]], axis=-1)
def CNN2(x, activation=tf.nn.relu, output_activation=None,):
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=20, strides=10, padding='same',activation=tf.nn.relu)
    x2 = tf.layers.conv1d(x1, filters=16, kernel_size=10, strides=3, padding='same',activation=tf.nn.relu)
    x2_flatten = tf.layers.flatten(x2)
    x3 = tf.layers.dense(x2_flatten, units=128, activation=tf.nn.relu)
    return x3
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a,hidden_sizes, activation, output_activation,alpha_actv1):
    act_dim = a.shape.as_list()[-1]
    x_input = x[:,0:2160]
    x_input = tf.reshape(x_input,[-1,720,3])
    x0 = new_relu(x_input[:,:,2], alpha_actv1)
    x_input = tf.concat([x_input[:,:,0:2],tf.reshape(x0,[-1,720,1])], axis=-1)
    w_input = x[:,2160:2160+4]
    w_input = tf.reshape(w_input,[-1,4])
    cnn_net = CNN(x_input,w_input)
    y = tf.concat([cnn_net,x[:,2160:]], axis=-1)
    net = mlp_policy(y,list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
#    pi_run = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
#    pi_run = pi
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a,hidden_sizes=(100,100,100), activation=tf.nn.leaky_relu, 
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        alpha_actv1 = tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
        mu, pi, logp_pi = policy(x, a,[100,100], activation, output_activation,alpha_actv1)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
#    action_scale = action_space.high[0]
#    mu *= action_scale
#    pi *= action_scale

    # vfs
    # the dim of q function and v function is 1, and hence it use +[1]
#    vf_cnn = lambda x : CNN(x)
#    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
#    with tf.variable_scope('q1'):
#        q1 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q1', reuse=True):
#        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('q2'):
#        q2 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q2', reuse=True):
#        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('v'):
#        v = vf_mlp(x)        
    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('values'):   
        with tf.variable_scope('CNN'):
            y = CNN_dense(x,activation, None)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([y,pi], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([y,pi], axis=-1))
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
