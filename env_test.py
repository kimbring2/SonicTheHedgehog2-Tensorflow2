import retro
import gym
import numpy as np
import os
import time
import glob
import random
import argparse
import cv2
from absl import flags
from absl import logging

#replay_path = '/home/kimbring2/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2/retro-movies/human/SonicTheHedgehog2-Genesis/contest'

parser = argparse.ArgumentParser(description='Sonic IMPALA Learner')
parser.add_argument('--replay_path', type=str, help='replay file root path')
arguments = parser.parse_args()


action_dict = {
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 0,
                '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 1,
                '[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 2,
                '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 3,
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 4,
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 5,
                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]': 6,
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 7,
                '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 8,
                '[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 9,
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 10,
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 11,
                '[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]': 12,
                '[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 13,
                '[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 14,
                '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': 15,
                '[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': 16,
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]': 17,
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 18,
                '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 19,
                '[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 20,
                '[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]': 21,
                '[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]': 22,
                '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': 23,
                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]': 24,
                '[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]': 25,
                '[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]': 26,
                '[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]': 27,
                '[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]': 28,
                '[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]': 29,
                '[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 30,
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]': 31,
                '[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]': 32,
                '[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 33,
                '[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]': 34,
                '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': 35,
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]': 36,
                '[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]': 37,
                '[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]': 38,
                '[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]': 39,
                '[1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]': 40,
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]': 41,
                '[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]': 42,
                '[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]': 43,
                '[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]': 44,
                '[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]': 45,
                '[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 46,
                '[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]': 47,
                '[0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]': 48,
                '[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]': 49,
                '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': 50,
                '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]': 51,
                '[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]': 52,
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]': 53,
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]': 54,
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]': 55,
                '[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]': 56,
                '[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]': 57,
                '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]': 58,
                '[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]': 59,
                '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]': 60,
                '[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 61,
                '[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]': 62,
                '[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]': 63,
                '[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]': 64,
                '[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]': 65,
                '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]': 66,
                '[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 67,
                '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]': 68

              }
action_list = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              ]

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


#["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

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


stage_name_list = ['EmeraldHillZone', 'ChemicalPlantZone', 'AquaticRuinZone', 'CasinoNightZone', 'HillTopZone',
                   'MysticCaveZone', 'OilOceanZone', 'MetropolisZone', 'WingFortressZone']

test = True
if test:
    print("test")
    env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
    replay_path = arguments.replay_path

    replay_file_path_list = glob.glob(replay_path + '/*.bk2')

    #replay_file_path_list = []

    break_flag = True
    for replay_name in replay_file_path_list:
        replay_name = random.choice(replay_file_path_list)
        replay_name = replay_name.split('/')[-1]
        
        #print("replay_name: ", replay_name)
        stage_name = replay_name.split('-')
        stage_name = stage_name[2].split('.')[0]
        #print("stage_name: ", stage_name)
        stage_index = stage_name_list.index(stage_name)
        #print("stage_index: ", stage_index)
        #print("")
        '''
        if replay_name == 'SonicTheHedgehog2-Genesis-WingFortressZone-0004.bk2':
            break_flag = False

        if break_flag == True:
            continue
        '''
        #replay_name = 'SonicTheHedgehog2-Genesis-MetropolisZone.Act2-0002.bk2'

        replay = retro.Movie(os.path.join(replay_path, replay_name))
        replay.step()

        env.initial_state = replay.get_state()

        obs = env.reset()
        #print("obs.shape: ", obs.shape)

        print('stepping replay')
        while replay.step():
            keys = []
            for i in range(len(env.buttons)):
                key = int(replay.get_key(i, 0))
                keys.append(key)

            #action_index = action_dict[str(keys)]

            converted_action = action_conversion_table[str(keys)]
            #print("converted_action: ", converted_action)

            action_index = possible_action_list.index(converted_action)
            #print("action_index: ", action_index)

            obs, rew, done, info = env.step(converted_action)
            #print("obs.shape: ", obs.shape)
            #print("done: ", done)
            #frame_rgb = 0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]

            # convert everything to black and white (agent will train faster)
            #frame_rgb[frame_rgb < 100] = 0
            #frame_rgb[frame_rgb >= 100] = 255

            #obs.shape:  (224, 320, 3)
            #frame_rgb1 = cv2.resize(obs, dsize=(84, 84), interpolation=cv2.INTER_AREA)
            #frame_rgb1 = obs
            frame_rgb1 = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

            #frame_rgb2 = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
            #frame_rgb2 = cv2.resize(frame_rgb2, dsize=(512, 512), interpolation=cv2.INTER_AREA)
            #print("frame_rgb.shape: ", frame_rgb.shape)
            #cv2.imshow("frame_rgb1", frame_rgb1)
            #cv2.imshow("frame_rgb2", frame_rgb2)
            #cv2.waitKey(1)

            #env.render()

            saved_state = env.em.get_state()


test_action = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
test_action_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
action_sequence_list = [#test_action,
                        #test_action,
                        #test_action,
                        #test_action,
                        #test_action,
                        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        test_action_1,
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                       ]



class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[''], ['LEFT'], ['RIGHT'], ['B'], ['DOWN'], ['B', 'LEFT'], ['B', 'RIGHT'],
                   ['B', 'DOWN'], ['A', 'LEFT'], ['DOWN', 'RIGHT'], ['B', 'RIGHT', 'DOWN'],
                   ['B', 'LEFT', 'DOWN']]

        self._actions = []
        for action in actions:
            arr = np.array([0] * 12)
            for button in action:
                arr[buttons.index(button)] = 1

            self._actions.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._actions))

        #print("self._actions: ", self._actions)

    def action(self, a):
        #print("a: ", a)
        #print("self._actions[a]: ", self._actions[a])
        ##print("")

        return self._actions[a].copy()

'''
def main():
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')
    env = SonicDiscretizer(env)

    action_index = 0

    obs = env.reset()
    while True:
        #action = env.action_space.sample()
        #action = random.choice(action_list)
        action = action_sequence_list[action_index]

        time.sleep(0.05)
        print("action: ", action)

        if action_index != len(action_sequence_list) - 1:
            action_index += 1
        else:
            action_index = 0

        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
'''