import collections
import zmq
import gym
import retro
import numpy as np
import statistics
import tqdm
import glob
import random
import tensorflow as tf
import cv2
import argparse
import os
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple
from absl import flags

parser = argparse.ArgumentParser(description='Sonic IMPALA Actor')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

writer = tf.summary.create_file_writer("tensorboard")

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))

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
state_size = (64,64,3)
stage_index = 0

state_list = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2'] 
if arguments.env_id % 2 ==0:
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')
else:
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act2')

scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    observation = env.reset()
    observation_resized = cv2.resize(observation, dsize=(64,64), interpolation=cv2.INTER_AREA)
    state = observation_resized / 255.0
    
    stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
    stage_layer[:, :, stage_index] = 1.0
    #state = np.concatenate((state, stage_layer), axis=2)
    state = state

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    last_screen_x = 0
    while True:
        try:
            step += 1

            state_reshaped = np.reshape(state, (1,*state_size)) 

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "observation": state_reshaped}
            socket.send_pyobj(env_output)
            action_index = int(socket.recv_pyobj()['action'])

            if arguments.env_id == 0 or arguments.env_id == 1:
                env.render()

            #print("action_index: ", action_index)
            action = possible_action_list[action_index]

            observation1, _, done, info = env.step(action)
            #print("info: ", info)

            screen_x_end = info['screen_x_end']
            screen_x = info['screen_x']

            #if screen_x > last_screen_x * 0.5:
            #    last_screen_x = screen_x
            #    reward = 1.0
            #    print("reward: ", reward)
            if done == True:
                reward = 1.0
                print("reward: ", reward)
            else:
                reward = 0.0

            observation1 = cv2.cvtColor(observation1, cv2.COLOR_BGR2RGB)
            observation1_resized = cv2.resize(observation1, dsize=(64,64), interpolation=cv2.INTER_AREA)
            #observation1_resized = cv2.cvtColor(observation1_resized, cv2.COLOR_BGR2RGB)
            next_state = observation1_resized / 255.0
            stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
            stage_layer[:, :, stage_index] = 1.0
            #next_state = np.concatenate((next_state, stage_layer), axis=2)
            next_state = next_state

            reward_sum += reward
            state = next_state
            if done or step == 10000:
                if arguments.env_id == 0:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    print("average_reward: " + str(average[-1]))
                else:
                    print("reward_sum: " + str(reward_sum))

                break

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')

env.close()