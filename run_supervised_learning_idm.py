from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from collections import deque, defaultdict
from absl import flags

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import retro
import gym
import os
import argparse
import cv2
import time
import os
import network

parser = argparse.ArgumentParser(description='Sonic Supervised Learning')
parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
parser.add_argument('--replay_path', type=str, help='root directory of dataset')
parser.add_argument('--level_name', type=str, help='name of level')
parser.add_argument('--use_action_history', action='store_true', default=False)
parser.add_argument('--gpu_use', action='store_true', default=False)

arguments = parser.parse_args()

workspace_path = arguments.workspace_path
use_action_history = arguments.use_action_history
level_name = arguments.level_name
gpu_use = arguments.gpu_use

if gpu_use == True:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


replay_path = os.path.join(arguments.replay_path, level_name)

if use_action_history == True:
    writer = tf.summary.create_file_writer(workspace_path + "/tensorboard_history/" + level_name)
else:
    writer = tf.summary.create_file_writer(workspace_path + "/tensorboard/" + level_name)


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

                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # = ['']
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


num_actions = len(possible_action_list)
#print("num_actions: ", num_actions)

stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']
stage_len = len(stage_name_list)

env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a])


class InverseTrajetoryDataset(tf.data.Dataset):
  def _generator(num_trajectorys):
    #env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)

    time_step = 32

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

    obs_history_list, action_history_list, real_action_list = [], [], []
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
        real_action_list.append(action_index)

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

            #print("obs_history_array.shape: ", obs_history_array.shape)
            #print("act_pi.shape: ", act_pi.shape)

            action_dist = tfd.Categorical(logits=act_pi)
            idm_actions = action_dist.sample().numpy()

            real_actions = np.array(real_action_list)

            #print("real_actions: ", real_actions)
            #print("idm_actions: ", idm_actions)

            obs_history_list, action_history_list, real_action_list = [], [], []
            obs_history_list_list, action_history_list_list, action_list_list = [], [], []
            #print("")

        step_num += 1

  def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.float32),
          args=(num_trajectorys,)
      )

dataset = tf.data.Dataset.range(1).interleave(TrajetoryDataset, 
  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)


num_hidden_units = 1024

#model = tf.keras.models.load_model('MineRL_SL_Model')
policy_model = network.ActorCritic(num_actions, num_hidden_units, use_action_history)

num_actions = len(possible_action_list)
idm_model = network.InverseActionPolicy(num_actions, num_hidden_units)
idm_model_name = 'inverse_dynamic_model_90'
idm_model.load_weights("model/" + model_name)


if arguments.pretrained_model != None:
    print("Load Pretrained Model")
    if use_action_history:
        model.load_weights(workspace_path + '/model_history/' + level_name + '/' + arguments.pretrained_model)
    else:
        model.load_weights(workspace_path + '/model/' + level_name + '/' + arguments.pretrained_model)

    #model.load_weights("model/" + arguments.pretrained_model)


cce_loss = tf.keras.losses.CategoricalCrossentropy()
cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


time_step = 64
optimizer = tf.keras.optimizers.Adam(0.0001)


#@tf.function
def supervised_replay(replay_obs_list, replay_act_list, replay_act_history_list, memory_state_obs, carry_state_obs,
                      memory_state_his, carry_state_his):
    replay_obs_array = tf.concat(replay_obs_list, 0)
    replay_act_array = tf.concat(replay_act_list, 0)
    replay_act_history_array = tf.concat(replay_act_history_list, 0)

    replay_memory_state_obs_array = tf.concat(memory_state_obs, 0)
    replay_carry_state_obs_array = tf.concat(carry_state_obs, 0)

    replay_memory_state_his_array = tf.concat(memory_state_his, 0)
    replay_carry_state_his_array = tf.concat(carry_state_his, 0)

    memory_state_obs = replay_memory_state_obs_array
    carry_state_obs = replay_carry_state_obs_array
    memory_state_his = replay_memory_state_his_array
    carry_state_his = replay_carry_state_his_array

    batch_size = replay_obs_array.shape[0]
    #tf.print("batch_size: ", batch_size)
    
    with tf.GradientTape() as tape:
        act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        cvae_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            prediction = model(tf.expand_dims(replay_obs_array[i,:,:,:], 0), 
                               tf.expand_dims(replay_act_history_array[i,:,:,:], 0),
                               memory_state_obs, carry_state_obs, memory_state_his, carry_state_his, 
                               training=True)
            act_pi = prediction[0]
            memory_state_obs = prediction[2]
            carry_state_obs = prediction[3]
            memory_state_his = prediction[4]
            carry_state_his = prediction[5]
            cvae_loss = prediction[6]
            
            act_probs = act_probs.write(i, act_pi[0])
            cvae_losses = cvae_losses.write(i, cvae_loss)

        act_probs = act_probs.stack()
        cvae_losses = cvae_losses.stack()

        replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
        replay_act_array_onehot = tf.reshape(replay_act_array_onehot, (batch_size, num_actions))
        act_loss = cce_loss_logits(replay_act_array_onehot, act_probs)

        cvae_loss = -tf.reduce_mean(cvae_losses)

        regularization_loss = tf.reduce_sum(model.losses)

        total_loss = act_loss + 1e-5 * regularization_loss + cvae_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return act_loss, cvae_loss, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his



def generate_and_save_images(model, epoch):
    batch_size = 16
    num_examples_to_generate = 16

    # Pick a sample of the test set for generating output images
    assert batch_size >= num_examples_to_generate
    for test_batch in dataset.take(1):
        episode_size = test_batch[0].shape[1]
        # episode_size:  2007
    
        replay_obs_list = test_batch[0][0]
        # replay_obs_list.shape:  (2007, 80, 80, 3)

        start_index = random.randint(0, episode_size - 33)

        test_sample = replay_obs_list[start_index:start_index + batch_size,:,:,:]

    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()


def supervised_train(dataset, training_episode):
    for batch in dataset:
        episode_size = batch[0].shape[1]
        print("episode_size: ", episode_size)
    
        replay_obs_list = batch[0][0]
        replay_act_list = batch[1][0]
        replay_act_history_list = batch[2][0]

        memory_state_obs = np.zeros([1,256], dtype=np.float32)
        carry_state_obs =  np.zeros([1,256], dtype=np.float32)
        memory_state_his = np.zeros([1,256], dtype=np.float32)
        carry_state_his =  np.zeros([1,256], dtype=np.float32)

        step_length = 128
        total_loss = 0
        for episode_index in range(0, episode_size, step_length):
            obs = replay_obs_list[episode_index:episode_index+step_length,:,:,:]
            act = replay_act_list[episode_index:episode_index+step_length,:]
            act_history = replay_act_history_list[episode_index:episode_index+step_length,:,:,:]
            
            #print("episode_index: ", episode_index)
            if len(obs) != step_length:
                break
            
            act_loss, cvae_loss, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his = supervised_replay(obs, act, 
                                                                                                                          act_history,
                                                                                                                          memory_state_obs, 
                                                                                                                          carry_state_obs,
                                                                                                                          memory_state_his, 
                                                                                                                          carry_state_his)
    
            #print("total_loss: ", total_loss)
            #print("")
            
        with writer.as_default():
            #print("training_episode: ", training_episode)
            tf.summary.scalar("act_loss", act_loss, step=training_episode)
            tf.summary.scalar("cvae_loss", cvae_loss, step=training_episode)
            writer.flush()

        if training_episode % 100 == 0:
            if use_action_history:
                model.save_weights(workspace_path + '/model_history/' + level_name + '/supervised_model_' + str(training_episode))
            else:
                model.save_weights(workspace_path + '/model/' + level_name + '/supervised_model_' + str(training_episode))

            generate_and_save_images(model.CVAE, training_episode)
            
        
for training_episode in range(0, 2000000):
    #print("training_episode: ", training_episode)
    supervised_train(dataset, training_episode)