# -*- coding: utf-8 -*-
# import os
from os.path import dirname, abspath, join
import argparse
import sys
import numpy as np
import torch
import keras
import rospy
import time
from geometry_msgs.msg import PointStamped
from waypoint import GlobalPlanSampler
import core_attention20_points_full_share as core
from core_attention20_points_full_share import get_vars
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
sys.path.append(dirname(dirname(abspath(__file__))))
from os.path import dirname, abspath, join
import gym
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# add the gpu root
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
from envs.wrappers import ShapingRewardWrapper, StackFrame
from td3.train import initialize_policy
# 与训练代码类似的模型定义和初始化部分
obs_dim = 2160 + 4
act_dim = 2
x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

with tf.variable_scope('main'):
    mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = core.mlp_actor_critic(x_ph, a_ph)
    
with tf.variable_scope('target'):
    _, _, _, _,_, q1_pi_targ, q2_pi_targ  = core.mlp_actor_critic(x2_ph, a_ph)
    
target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(target_init)
trainables = tf.trainable_variables()
trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
sess.run(tf.global_variables_initializer())
trainable_saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/paper3_train/networks/sac1399lambda0-50")
# saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/Good_with_backward/network/configback540GMMdense11lambda0-100")
sys.path.append(dirname(dirname(abspath(__file__))))

def get_action(o, deterministic= True):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

def get_world_name(config, id):
    assert 0 <= id < 300, "BARN dataset world index ranges from 0-299"
    world_name = "BARN/world_%d.world" %(id)
    return world_name
def target_points_callback(msgs):
    return msgs

def load_policy(policy, policy_path):
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print(policy_path)
    policy.load(policy_path, "last_policy")
    policy.exploration_noise = 0
    return policy

def _debug_print_robot_status(env, count, rew, actions):
    Y = env.move_base.robot_config.Y
    X = env.move_base.robot_config.X
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, p.x, X, p.y, Y , rew))

def main(args):
    data = None
    with open(join(args.policy_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)
    env_config["kwargs"]["world_name"] = world_name
    if args.gui:
        env_config["kwargs"]["gui"] = True
    env_config["kwargs"]["init_sim"] = False
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    flage = True
    max_acc1 = [1.0,1.0] # 线速度和角速度的最大值
    max_acc2 = [1.0,1.0] # 线速度和角速度的最大值
    while ep < args.repeats:
        obs = env.reset()
        ep += 1
        print(">>>>>>>>>>>>>> Running on the step number %s <<<<<<<<<<<<<<<<" % ep)
        step = 0
        done = False
        d =  0
        while True:           
            if not args.default_dwa:
                actions = get_action(obs)
            else:
                actions = get_action(obs)
            if flage == True:
                actions[0] = max_acc1[0]*actions[0]
                actions[1] = max_acc1[1] * actions[1]
            else:
                actions[0] = max_acc2[0]*actions[0]
                actions[1] = max_acc2[1]*actions[1]
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            flage = done
            # print("the now state of flage is :", flage)
            obs = obs_new
            step += 1
            if args.verbose:
                _debug_print_robot_status(env, step, rew, actions)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--world_id', dest='id', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default="end_to_end/data_sac")
    parser.add_argument('--default_dwa', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    main(args)

