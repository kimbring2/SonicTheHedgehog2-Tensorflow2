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
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
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
num_hidden_units = 1024

model = network.ActorCritic(num_actions, num_hidden_units)

model.load_weights("model/" + arguments.model_name)

seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

reward_sum = 0
stage_index = 0

test_stage = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2']

e = 0.05

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])

for i_episode in range(0, 10000):
    # Create the environment
    stage_name = random.choice(test_stage)
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act2.Boss')

    obs = env.reset()

    obs_resized = cv2.resize(obs, dsize=(80,80), interpolation=cv2.INTER_AREA) / 255.0
    #obs_resized = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2RGB) / 255.0

    #obs = 0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]
    #obs[obs < 100] = 0
    #obs[obs >= 100] = 255
    #obs = cv2.resize(obs, dsize=(84,84), interpolation=cv2.INTER_AREA) / 255.0

    #obs_t = np.stack((obs, obs, obs, obs), axis=2)
    
    #stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
    #stage_layer[:, :, stage_index] = 1.0
    #state = np.concatenate((state, stage_layer), axis=2)
    state = obs_resized

    memory_state = tf.zeros([1,256], dtype=np.float32)
    carry_state = tf.zeros([1,256], dtype=np.float32)
    step = 0
    done = False

    action_index = 0
    pre_action_index = action_index
    while True:
        step += 1

        env.render()

        #print("action_index: ", action_index)
        #print("pre_action_index: ", pre_action_index)
        #print("")

        pre_action_onehot = one_hot(pre_action_index, len(possible_action_list))



        action_probs, _, memory_state, carry_state, cvae_loss = model(tf.expand_dims(state, 0), 
                                                                      np.array(pre_action_onehot, dtype=np.float32),
                                                                      memory_state, carry_state)
        action_dist = tfd.Categorical(logits=action_probs)
        
        mean, logvar = model.CVAE.encode(tf.expand_dims(state, 0))
        z = model.CVAE.reparameterize(mean, logvar)
        prediction = model.CVAE.sample(z)
        prediction = np.array(prediction)
        #print("prediction: ", prediction)

        cv2.imshow("prediction", prediction[0])
        cv2.waitKey(1)
        #print("info: ", info)
        #print("prediction.shape: ", prediction.shape)

        #if np.random.rand(1) > e:
        #    action_index = int(action_dist.sample()[0])
        #else:
        #    action_index = random.randint(0, len(possible_action_list) - 1)

        pre_action_index = action_index
        action_index = int(action_dist.sample()[0])
        #print("action_index: ", action_index)
        action = possible_action_list[action_index]

        next_obs, reward, _, info = env.step(action)
        #next_obs = 0.299*next_obs[:,:,0] + 0.587*next_obs[:,:,1] + 0.114*next_obs[:,:,2]
        #next_obs[next_obs < 100] = 0
        #next_obs[next_obs >= 100] = 255
        next_obs_resized = cv2.resize(next_obs, dsize=(80,80), interpolation=cv2.INTER_AREA) / 255.0
        #x_t1 = np.reshape(next_obs, (84,84,1))
        #next_obs_t = np.append(x_t1, obs_t[:, :, :3], axis=2)
        #next_obs_resized = cv2.cvtColor(next_obs_resized, cv2.COLOR_BGR2RGB) / 255.0
        #cv2.imshow("observation1", observation1)
        #cv2.waitKey(1)
        #print("info: ", info)

        next_state = next_obs_resized
        screen_x = info['screen_x']
        if screen_x >= 10920:
            done = True
            #print("reward: ", reward)

        #stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
        #stage_layer[:, :, stage_index] = 1.0
        #next_state = np.concatenate((next_state, stage_layer), axis=2)

        reward_sum += reward

        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

env.close()