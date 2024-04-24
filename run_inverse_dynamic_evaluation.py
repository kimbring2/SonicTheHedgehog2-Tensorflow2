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


workspace_path = arguments.workspace_path
#replay_path = arguments.replay_path
workspace_path = '/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2'
replay_path = '/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/retro-movies/human/SonicTheHedgehog2-Genesis/contest/SonicTheHedgehog2-Genesis-ChemicalPlantZone.Act1'

writer = tf.summary.create_file_writer(workspace_path + "/tensorboard")

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
num_hidden_units = 2048

model = network.InverseActionPolicy(num_actions, num_hidden_units)

model_name = 'inverse_dynamic_model_60'
model.load_weights("model/" + model_name)
 

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a])


time_step = 128

while True:
    replay_file_path_list = glob.glob(replay_path + '/*.bk2')
    replay_name = random.choice(replay_file_path_list)
    replay_name = replay_name.split('/')[-1]

    replay = retro.Movie(os.path.join(replay_path, replay_name))
    replay.step()

    env.initial_state = replay.get_state()
    obs = env.reset()

    obs = cv2.resize(obs, dsize=(64,64), interpolation=cv2.INTER_AREA) / 255.0

    action_index = 0

    obs_history = np.zeros((time_step, 64, 64, 3))
    action_history = np.zeros((time_step))

    step_num = 0

    print('stepping replay')

    memory_state = np.zeros([1,1024], dtype=np.float32)
    carry_state =  np.zeros([1,1024], dtype=np.float32)
    while replay.step():
        #env.render()
        #print("step_num: ", step_num)

        obs_history = np.roll(obs_history, 1, axis=0)
        obs_history[0,:,:,:] = obs
        
        action_history = np.roll(action_history, 1, axis=0)
        action_history[0] = action_index
        
        keys = []
        for i in range(len(env.buttons)):
            key = int(replay.get_key(i, 0))
            keys.append(key)

        converted_action = action_conversion_table[str(keys)]
        action_index = possible_action_list.index(converted_action)

        next_obs, rew, done, info = env.step(converted_action)
        next_obs = cv2.resize(next_obs, dsize=(64, 64), interpolation=cv2.INTER_AREA) / 255.0

        obs = next_obs
        saved_state = env.em.get_state()
        
        if step_num % 128 == 0:
            obs_history_temp = np.expand_dims(obs_history, 0)
            #print("obs_history_temp.shape: ", obs_history_temp.shape)
            prediction = model(tf.constant(obs_history_temp), tf.constant(memory_state), tf.constant(carry_state))
            act_pi = prediction[0]
            memory_state = prediction[1]
            carry_state = prediction[2]

            print("act_pi.shape: ", act_pi.shape)
            #print("memory_state.shape: ", memory_state.shape)
            #print("carry_state.shape: ", carry_state.shape)

            action_dist = tfd.Categorical(logits=act_pi)
            action_indexs = action_dist.sample()

            print("action_history: ", action_history)
            print("action_indexs: ", action_indexs)

            print("")
        
        step_num += 1