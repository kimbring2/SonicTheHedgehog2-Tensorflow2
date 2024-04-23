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

import network

parser = argparse.ArgumentParser(description='Sonic Supervised Learning')
parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
parser.add_argument('--replay_path', type=str, help='root directory of dataset')
parser.add_argument('--gpu_use', action='store_false', help='use gpu')

arguments = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a])


class TrajetoryDataset(tf.data.Dataset):
  def _generator(num_trajectorys):
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

        action_history_list, obs_history_list = [], []
        obs_history = np.zeros((8, 64, 64, 3))
        action_history = np.zeros((8, len(possible_action_list)))

        step_num = 0

        print('stepping replay')
        while replay.step():
            #print("step_num: ", step_num)

            obs_history_list.append(obs_history)
            action_history_list.append(action_history)

            obs_history = np.roll(obs_history, 1, axis=0)
            obs_history[0,:,:,:] = obs

            action_onehot = one_hot(action_index, len(possible_action_list))
            action_history = np.roll(action_history, 1, axis=0)
            action_history[0,:] = action_onehot
            
            keys = []
            for i in range(len(env.buttons)):
                key = int(replay.get_key(i, 0))
                keys.append(key)

            converted_action = action_conversion_table[str(keys)]

            pre_action_index = action_index
            action_index = possible_action_list.index(converted_action)

            next_obs, rew, done, info = env.step(converted_action)
            next_obs = cv2.resize(next_obs, dsize=(64, 64), interpolation=cv2.INTER_AREA) / 255.0

            obs = next_obs

            saved_state = env.em.get_state()

            #if step_num == 20:
            #    break

            step_num += 1

        yield (obs_history_list, action_history_list)

  def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.float32, tf.dtypes.int32),
          args=(num_trajectorys,)
      )

dataset = tf.data.Dataset.range(1).interleave(TrajetoryDataset, 
  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)


    
num_actions = len(possible_action_list)
num_hidden_units = 1024

#model = tf.keras.models.load_model('MineRL_SL_Model')
model = network.InverseActionPolicy(num_actions, num_hidden_units)

if arguments.pretrained_model != None:
    print("Load Pretrained Model")
    model.load_weights("model/" + arguments.pretrained_model)

    
cce_loss = tf.keras.losses.CategoricalCrossentropy()
cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001)


#@tf.function
def supervised_replay(replay_obs_list, replay_act_list):
    replay_obs_array = tf.concat(replay_obs_list, 0)
    replay_act_array = tf.concat(replay_act_list, 0)

    #print("replay_obs_array.shape: ", replay_obs_array.shape)
    #print("replay_act_array.shape: ", replay_act_array.shape)

    batch_size = replay_obs_array.shape[0]
    #tf.print("batch_size: ", batch_size)
    
    with tf.GradientTape() as tape:
        act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            #print("tf.expand_dims(replay_obs_array[i,:,:,:,:], 0).shape: ", tf.expand_dims(replay_obs_array[i,:,:,:,:], 0).shape)
            prediction = model(tf.expand_dims(replay_obs_array[i,:,:,:,:], 0), training=True)
            #print("prediction.shape: ", prediction.shape)
            act_pi = prediction[0]
        
            act_probs = act_probs.write(i, act_pi)

        act_probs = act_probs.stack()

        #print("replay_act_array.shape: ", replay_act_array.shape)
        #print("act_probs.shape: ", act_probs.shape)
        act_loss = cce_loss_logits(replay_act_array, act_probs)
        #print("act_loss: ", act_loss)

        regularization_loss = tf.reduce_sum(model.losses)

        actor_loss = act_loss + 1e-5 * regularization_loss
        total_loss = actor_loss
    
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return actor_loss


def supervised_train(dataset, training_episode):
    print("training_episode: ", training_episode)
    for batch in dataset:
        episode_size = batch[0].shape[1]
        #print("episode_size: ", episode_size)

        replay_obs_array = batch[0][0]
        replay_act_array = batch[1][0]

        print("replay_obs_array.shape: ", replay_obs_array.shape)
        print("replay_act_array.shape: ", replay_act_array.shape)

        step_length = 1
        total_loss = 0
        for episode_index in range(0, episode_size, step_length):
            #print("replay_obs_array[episode_index:episode_index+step_length,:,:,:].shape: ", 
            #    replay_obs_array[episode_index:episode_index+step_length,:,:,:].shape)

            obs = replay_obs_array[episode_index:episode_index+step_length,:,:,:]
            act = replay_act_array[episode_index:episode_index+step_length,:]
            
            #print("episode_index: ", episode_index)
            if len(obs) != step_length:
                break
            
            #print("obs.shape: ", obs.shape)
            #print("act.shape: ", act.shape)
            total_loss = supervised_replay(obs, act)
        
            print("total_loss: ", total_loss)
            print("")
            
        with writer.as_default():
            #print("training_episode: ", training_episode)
            tf.summary.scalar("actor_loss", actor_loss, step=training_episode)
            writer.flush()

        if training_episode % 100 == 0:
            model.save_weights(workspace_path + '/model/supervised_model_' + str(training_episode))
            
        
for training_episode in range(0, 2000000):
    #print("training_episode: ", training_episode)
    supervised_train(dataset, training_episode)