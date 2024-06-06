from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from collections import deque, defaultdict
from absl import flags

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import glob
import random
import retro
import gym
import os
import argparse
import cv2
import copy

import network

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Sonic Supervised Learning')
parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
parser.add_argument('--replay_path', type=str, help='root directory of dataset')
parser.add_argument('--gpu_use', action='store_false', help='use gpu')

arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

'''
if arguments.gpu_use:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])
'''

workspace_path = arguments.workspace_path
#replay_path = arguments.replay_path
workspace_path = '/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2'
replay_path = '/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/retro-movies/human/SonicTheHedgehog2-Genesis/contest/SonicTheHedgehog2-Genesis-ChemicalPlantZone.Act1'

action_conversion_table = {
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']        
                '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['DOWN']
                '[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['A', 'DOWN']

                '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['A']
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['A', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['']
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']

                '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT']
                '[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['LEFT'] 

                '[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['A', 'RIGHT', 'DOWN']
                '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['RIGHT']
                '[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['A', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']

                '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT', 'DOWN']
                '[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['B', 'LEFT', 'RIGHT']
                '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # = ['UP']

                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT', 'DOWN']
                '[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # = ['DOWN', 'LEFT', 'RIGHT']
                '[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['A', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']

                '[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT’]

                '[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']

                '[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT', 'DOWN']
                '[1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # = ['B', 'LEFT', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]' : [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['A', 'LEFT']
                '[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['LEFT']
                '[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']

                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']

                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # = ['B', 'LEFT']
                '[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # = ['B', 'DOWN']
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['RIGHT']

                '[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['B']
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # = ['B', 'RIGHT']
                '[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]' : [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # = ['DOWN', RIGHT']

                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # = ['']
              }

# ['']:                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['LEFT']:                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['RIGHT']:                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['B']:                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['DOWN']:                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['A']:                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['UP']:                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ['B', 'LEFT']:             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['B', 'RIGHT']:            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['B', 'DOWN']:             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['A', 'LEFT']:             [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['A', 'RIGHT']:            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['A', 'DOWN']:             [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['DOWN', 'RIGHT']:         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['DOWN', 'LEFT']:          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['LEFT', 'RIGHT']:         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['B', 'RIGHT', 'DOWN']:    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['B', 'LEFT', 'DOWN'] :    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['A', 'LEFT', 'DOWN'] :    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['A', 'LEFT', 'RIGHT']:    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['A', 'RIGHT', 'DOWN']:    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['B', 'LEFT', 'RIGHT']:    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
# ['DOWN', 'LEFT', 'RIGHT']: [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


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

#print("len(possible_action_list): ", len(possible_action_list))

stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']
stage_len = len(stage_name_list)

env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)


num_actions = len(possible_action_list)
num_hidden_units = 1024

model = network.InverseActionPolicy(num_actions, num_hidden_units)
model_name = 'inverse_dynamic_model_90'
model.load_weights("model/" + model_name)


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a])


time_step = 64

while True:
    replay_file_path_list = glob.glob(replay_path + '/*.bk2')
    replay_name = random.choice(replay_file_path_list)
    replay_name = replay_name.split('/')[-1]

    replay = retro.Movie(os.path.join(replay_path, replay_name))
    replay.step()

    env.initial_state = replay.get_state()
    obs = env.reset()

    obs = cv2.resize(obs, dsize=(84, 84), interpolation=cv2.INTER_AREA)
    obs = obs / 255.0

    action_index = 0
    pre_action_index = action_index

    obs_history_list, action_history_list, action_list = [], [], []
    obs_history_list_list, action_history_list_list, action_list_list = [], [], []

    action_history = np.zeros((12, len(possible_action_list)))
    step_num = 0

    memory_state_obs = np.zeros([1,512], dtype=np.float32)
    carry_state_obs =  np.zeros([1,512], dtype=np.float32)
    memory_state_his = np.zeros([1,128], dtype=np.float32)
    carry_state_his =  np.zeros([1,128], dtype=np.float32)

    print('stepping replay')
    while replay.step():
        #env.render()
        #print("step_num: ", step_num)

        keys = []
        for i in range(len(env.buttons)):
            key = int(replay.get_key(i, 0))
            keys.append(key)

        converted_action = action_conversion_table[str(keys)]

        pre_action_index = action_index
        action_index = possible_action_list.index(converted_action)

        next_obs, rew, done, info = env.step(converted_action)
        next_obs = cv2.resize(next_obs, dsize=(84, 84), interpolation=cv2.INTER_AREA)
        next_obs = next_obs / 255.0

        ########################################################################################
        obs_history_list.append(obs)

        action_onehot = one_hot(action_index, num_actions)
        action_list.append(action_index)

        pre_action_onehot = one_hot(pre_action_index, num_actions)
        action_history = np.roll(action_history, 1, axis=0)
        action_history[0,:] = pre_action_onehot
        action_history_list.append(action_history)
        ########################################################################################

        obs = next_obs
        saved_state = env.em.get_state()
    
        if len(obs_history_list) == time_step:
            obs_history_array = np.array(obs_history_list)
            obs_history_array = np.expand_dims(obs_history_array, 0)

            action_history_array = np.array(action_history_list)
            action_history_array = np.expand_dims(action_history_array, 0)

            prediction = model(tf.constant(obs_history_array), 
                               tf.constant(action_history_array),
                               tf.constant(memory_state_obs), tf.constant(carry_state_obs),
                               tf.constant(memory_state_his), tf.constant(carry_state_his))


            act_pi = prediction[0][0]

            memory_state_obs = prediction[1]
            carry_state_obs = prediction[2]

            memory_state_his = prediction[3]
            carry_state_his = prediction[4]

            print("act_pi.shape: ", act_pi.shape)

            action_dist = tfd.Categorical(logits=act_pi)
            action_indexs = action_dist.sample()

            print("action_list: ", action_list)
            print("action_indexs: ", action_indexs)

            obs_history_list, action_history_list, action_list = [], [], []
            print("")

        step_num += 1