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
parser.add_argument('--model_name', type=str, help='name of saved model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

arguments = parser.parse_args()

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

workspace_path = arguments.workspace_path

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
num_hidden_units = 512

model = network.ActorCritic(num_actions, num_hidden_units)

model.load_weights("model/" + arguments.model_name)

seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

reward_sum = 0
stage_index = 0

# Create the environment
env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')

for i_episode in range(0, 10000):
    observation = env.reset()
    observation_resized = cv2.resize(observation, dsize=(64,64), interpolation=cv2.INTER_AREA)
    state = observation_resized / 255.0
    
    stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
    stage_layer[:, :, stage_index] = 1.0
    #state = np.concatenate((state, stage_layer), axis=2)
    state = state

    memory_state = tf.zeros([1,128], dtype=np.float32)
    carry_state = tf.zeros([1,128], dtype=np.float32)
    step = 0
    while True:
        step += 1

        env.render()

        action_probs, _, memory_state, carry_state = model(tf.expand_dims(state, 0), memory_state, carry_state)
        action_dist = tfd.Categorical(logits=action_probs)
        action_index = int(action_dist.sample()[0])

        #print("action_index: ", action_index)
        action = possible_action_list[action_index]

        observation1, reward, done, info = env.step(action)
        observation1 = cv2.cvtColor(observation1, cv2.COLOR_BGR2RGB)
        #cv2.imshow("observation1", observation1)
        #cv2.waitKey(1)

        observation1_resized = cv2.resize(observation1, dsize=(64,64), interpolation=cv2.INTER_AREA)
        #observation1_resized = cv2.cvtColor(observation1_resized, cv2.COLOR_BGR2RGB)
        next_state = observation1_resized / 255.0
        stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
        stage_layer[:, :, stage_index] = 1.0
        #next_state = np.concatenate((next_state, stage_layer), axis=2)
        next_state = next_state

        reward_sum += reward

        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

env.close()