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
socket.connect("tcp://localhost:" + str(6555 + arguments.env_id))

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

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


def render(obs, id):
    cv2.imshow('obs_' + str(id) , obs)
    cv2.waitKey(1)

stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']
stage_len = len(stage_name_list) 

num_actions = len(possible_action_list)
state_size = (84,84,3)
act_history_size = (23,23,1)  
stage_index = 0

state_list = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2'] 
#if arguments.env_id % 2 ==0:
#    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')
#else:
#    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act2')
env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act2.Boss.Short')

scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    observation = env.reset()
    observation_resized = cv2.resize(observation, dsize=(84,84), interpolation=cv2.INTER_AREA)
    #observation_resized = cv2.cvtColor(observation_resized, cv2.COLOR_BGR2RGB)
    state = observation_resized / 255.0
    
    #stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
    #stage_layer[:, :, stage_index] = 1.0
    #state = np.concatenate((state, stage_layer), axis=2)
    state = state

    reach_end = False
    reverse_end = False
    game_over = False
    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    last_screen_x = 0
    time_out = False
    die = False

    action_index = 0
    pre_action_index = action_index
    act_history = np.zeros((23, 23))

    memory_state_obs = np.zeros([1,256], dtype=np.float32)
    carry_state_obs = np.zeros([1,256], dtype=np.float32)
    memory_state_his = np.zeros([1,256], dtype=np.float32)
    carry_state_his = np.zeros([1,256], dtype=np.float32)
    while True:
        try:
            step += 1

            state_reshaped = np.reshape(state, (1, *state_size)) 

            pre_action_onehot = one_hot(pre_action_index, len(possible_action_list))
            act_history = np.roll(act_history, 1, axis=0)
            act_history[0,:] = pre_action_onehot

            #print("state_reshaped.shape: ", state_reshaped.shape)

            act_history_reshaped = np.reshape(act_history, (1, *act_history_size)) 
            print("act_history_reshaped.shape: ", act_history_reshaped.shape)

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "observation": state_reshaped,
                          "act_history": act_history_reshaped,
                          "memory_state_obs": memory_state_obs,
                          "carry_state_obs": carry_state_obs, 
                          "memory_state_his": memory_state_his,
                          "carry_state_his": carry_state_his,
                          }
            socket.send_pyobj(env_output)
            recv_pyobj = socket.recv_pyobj()
            #print("recv_pyobj: ", recv_pyobj)

            action_index = int(recv_pyobj['action'])
            memory_state_obs = recv_pyobj['memory_state_obs']
            carry_state_obs = recv_pyobj['carry_state_obs']
            memory_state_his = recv_pyobj['memory_state_his']
            carry_state_his = recv_pyobj['carry_state_his']

            if time_out == True or die == True or reach_end == True or reverse_end == True:
                game_over = True

            if arguments.env_id == 0:
                #print("step: ", step)
                env.render()

            #print("action_index: ", action_index)
            pre_action_index = action_index
            action = possible_action_list[action_index]

            observation1, _, done, info = env.step(action)
            #reward = reward / 1000.0
            #print("info: ", info)

            screen_x_end = info['screen_x_end']
            screen_x = info['screen_x']
            lives = info['lives']

            reward = 0
            '''
            if arguments.env_id % 2 == 0:
                if step == 8000:
                    time_out = True

                if screen_x >= 10656:
                    reward = 1.0
                    reach_end = True
            else:
                if step == 10000:
                    time_out = True

                if screen_x >= 10920:
                    reward = 1.0
                    reach_end = True
            '''
            if step == 4500:
                #reward = -1.0
                time_out = True

            if screen_x_end > 10560:
                reward = 1.0
                reach_end = True

            if screen_x < 9996 and screen_x >= 9530:
                #reward = -1.0
                reverse_end = True

            if lives == 3:
                reward = -1.0
                die = True 

            observation1_resized = cv2.resize(observation1, dsize=(84,84), interpolation=cv2.INTER_AREA)
            #observation1_resized = cv2.cvtColor(observation1_resized, cv2.COLOR_BGR2RGB)
            #observation1_resized = cv2.cvtColor(observation1_resized, cv2.COLOR_BGR2RGB)
            next_state = observation1_resized / 255.0
            #stage_layer = np.zeros([64,64,stage_len], dtype=np.float32)
            #stage_layer[:, :, stage_index] = 1.0
            #next_state = np.concatenate((next_state, stage_layer), axis=2)

            #render(next_state, arguments.env_id)

            if game_over == True:
                if arguments.env_id == 0:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        #tf.summary.scalar("average_reward act1", average[-1], step=episode_step)
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    #print("average_reward act1: " + str(average[-1]))
                    print("average_reward: " + str(average[-1]))
                else:
                    print("reward_sum: " + str(reward_sum))
                '''
                elif arguments.env_id == 1:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        tf.summary.scalar("average_reward act2", average[-1], step=episode_step)
                        writer.flush()

                    print("average_reward act2: " + str(average[-1]))
                '''

                break

            reward_sum += reward
            state = next_state

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')

env.close()