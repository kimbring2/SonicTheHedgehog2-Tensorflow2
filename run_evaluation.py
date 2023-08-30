import collections
import gym
import retro
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import cv2
import tensorflow_probability as tfp
import argparse
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from absl import flags

import network

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Sonic Evaluation')

parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--use_action_history', type=bool, default=False, help='Whether to use action history or not')
parser.add_argument('--model_name', type=str, help='name of saved model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--level_name', type=str, help='name of level')

arguments = parser.parse_args()

workspace_path = arguments.workspace_path
use_action_history = arguments.use_action_history
level_name = arguments.level_name
gpu_use = arguments.gpu_use

if gpu_use:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


possible_action_list = [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
                       ]


stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']
stage_len = len(stage_name_list) 

num_actions = len(possible_action_list)
num_hidden_units = 1024

model = network.ActorCritic(num_actions, num_hidden_units, use_action_history)

if use_action_history:
    model.load_weights(workspace_path + '/model_history/' + level_name + '/' + arguments.model_name)
else:
    model.load_weights(workspace_path + '/model/' + level_name + '/' + arguments.model_name)


seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

state_size = (84,84,3)
act_history_size = (23,num_actions,1)  

reward_sum = 0
stage_index = 0

test_stage = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2']


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a])


print("level_name.split('-')[-1]: ", level_name.split('-')[-1])

for i_episode in range(0, 10000):
    # Create the environment
    stage_name = random.choice(test_stage)
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state=level_name.split('-')[-1])

    obs = env.reset()
    #print("obs.shape: ", obs.shape)
    obs_resized = cv2.resize(obs, dsize=(84,84), interpolation=cv2.INTER_AREA) / 255.0

    state = obs_resized
    pre_state = state

    memory_state_obs = tf.zeros([1,256], dtype=np.float32)
    carry_state_obs = tf.zeros([1,256], dtype=np.float32)
    memory_state_his = tf.zeros([1,256], dtype=np.float32)
    carry_state_his = tf.zeros([1,256], dtype=np.float32)
    step = 0
    done = False

    action_index = 0

    act_history = np.zeros((23, num_actions))
    while True:
        step += 1

        env.render()

        state_reshaped = np.reshape(state, (1, *state_size))
        pre_state_reshaped = np.reshape(pre_state, (1, *state_size))

        pre_action_onehot = one_hot(action_index, len(possible_action_list))
        act_history = np.roll(act_history, 1, axis=0)
        act_history[0,:] = pre_action_onehot
        act_history_reshaped = np.reshape(act_history, (1, *act_history_size)) 

        action_probs, _, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his, _ = model(state_reshaped, 
                                                                                                         act_history_reshaped,
                                                                                                         memory_state_obs, 
                                                                                                         carry_state_obs,
                                                                                                         memory_state_his, 
                                                                                                         carry_state_his,
                                                                                                         training=False)
        action_dist = tfd.Categorical(logits=action_probs)

        mean, logvar = model.CVAE.encode(tf.expand_dims(state, 0))
        z = model.CVAE.reparameterize(mean, logvar)
        prediction = model.CVAE.sample(z)
        prediction = np.array(prediction)
        prediction = cv2.resize(prediction[0], dsize=(320,224), interpolation=cv2.INTER_AREA)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
        cv2.imshow("prediction", prediction)
        cv2.waitKey(1)

        action_index = int(action_dist.sample()[0])
        action = possible_action_list[action_index]
        
        next_obs, reward, _, info = env.step(action)
        next_obs_resized = cv2.resize(next_obs, dsize=(84,84), interpolation=cv2.INTER_AREA) / 255.0

        next_state = next_obs_resized
        screen_x = info['screen_x']

        #if screen_x >= 6795:
        #    done = True
        #    print("reward: ", reward)

        if screen_x >= 10920:
            done = True
            print("reward: ", reward)

        #if screen_x < 9996:
        #    reward = -1.0
        #    done = True

        reward_sum += reward

        pre_state = state
        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

env.close()