import retro
import gym
import numpy as np
import os
import time
import glob
import random
import argparse
from absl import flags
from absl import logging

parser = argparse.ArgumentParser(description='Sonic IMPALA Learner')
parser.add_argument('--replay_path', type=str, help='replay file root path')
arguments = parser.parse_args()


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]

        self._actions = []
        for action in actions:
            arr = np.array([0] * 12)
            for button in action:
                arr[buttons.index(button)] = 1

            self._actions.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._actions))

        print("self._actions: ", self._actions)

    def action(self, a):
        #print("a: ", a)
        print("self._actions[a]: ", self._actions[a])
        ##print("")

        return self._actions[a].copy()


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
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
              ]

'''
env = retro.make(game='SonicTheHedgehog2-Genesis', state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
#replay_path = '/home/kimbring2/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2/retro-movies/human/SonicTheHedgehog2-Genesis/contest'
replay_path = arguments.replay_path

replay_file_path_list = glob.glob(replay_path + '/*.bk2')
for replay_name in replay_file_path_list:
    #print("replay_name: ", replay_name)
    replay_name = random.choice(replay_file_path_list)
    #replay_name = 'SonicTheHedgehog2-Genesis-EmeraldHillZone.Act1-0000.bk2'
    replay = retro.Movie(os.path.join(replay_path, replay_name))
    replay.step()

    env.initial_state = replay.get_state()
    env.reset()

    print('stepping replay')
    while replay.step():
        keys = []
        for i in range(len(env.buttons)):
            key = int(replay.get_key(i, 0))
            keys.append(key)

        #print("keys: ", keys)
        action_index = action_dict[str(keys)]
        #print("action_index: ", action_index)

        _obs, _rew, _done, _info = env.step(keys)

        #env.render()
        #time.sleep(0.01)

        saved_state = env.em.get_state()
'''

#print("len(action_dict.keys()): ", len(action_dict.keys()))
#print("action_dict.keys(): ", action_dict.keys())

#buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
#actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
#           ['DOWN', 'B'], ['B']]

# ['LEFT']:            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# ['RIGHT']:           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# ['LEFT', 'DOWN']:    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# ['RIGHT', 'DOWN']:   [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# ['DOWN']:            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['DOWN', 'B']:       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ['B']:               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

'''
action_list = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = ['']
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = ['LEFT']
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = ['LEFT']
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = ['A']
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = ['A', 'RIGHT']
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = ['RIGHT']
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] = ['LEFT', 'RIGHT']
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = ['LEFT']
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = ['A', 'LEFT']
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = ['B', 'A']
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = ['B']
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = ['B', 'RIGHT']
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] = ['DOWN', 'LEFT']
                [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] = ['A', 'DOWN', 'RIGHT']
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] = ['DOWN', 'RIGHT']
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] = ['C']
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = ['A', 'UP']
                [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] = ['A', 'LEFT', 'RIGHT'] = ['']
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = ['B', 'LEFT'] = ['LEFT']
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = ['B', 'DOWN'] = ['DOWN']
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] = ['B', 'DOWN', 'RIGHT']
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0] = ['UP', 'LEFT']
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] = ['B', 'LEFT', 'RIGHT']
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = ['UP']
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0] = ['B', 'UP', 'LEFT']
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] = ['B', 'DOWN', 'LEFT']
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0] = ['DOWN', 'LEFT', 'RIGHT']
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] = ['B', 'A', 'LEFT', 'RIGHT']
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] = ['B', 'LEFT', 'RIGHT', 'C']
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0] = ['RIGHT', 'C']
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0] = ['LEFT', 'Y']
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] = ['RIGHT', 'Z']
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0] = ['B', 'DOWN', 'LEFT', 'RIGHT']
                [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = ['B', 'A', 'RIGHT']
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] = ['LEFT', 'C']
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = ['B', 'UP']
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] = ['B', 'UP', 'RIGHT']
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0] = ['B', 'RIGHT', 'C']
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] = ['UP', 'RIGHT']
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] = ['A', 'DOWN', 'LEFT']
                [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0] = ['B', 'UP', 'LEFT', 'RIGHT']
    
                #buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0] = ['B', 'UP', 'LEFT', 'RIGHT']
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] = ['UP', 'DOWN']
                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] = ['B', 'A', 'LEFT', 'RIGHT', 'C']
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] = ['RIGHT', 'Y']
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] = ['B', 'LEFT', 'C']
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = ['B', 'A', 'LEFT']
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] = ['B', 'UP', 'DOWN']
                [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0] = ['UP', 'LEFT', 'RIGHT']
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] = ['A', 'LEFT', 'RIGHT', 'C']
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] = ['B', 'C']
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] = ['B', 'LEFT', 'Z']
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] = ['LEFT', 'RIGHT', 'C']
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] = ['B', 'RIGHT', 'Z']
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] = ['B', 'RIGHT', 'Y']
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] = ['Y']
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1] = ['B', 'RIGHT', 'Y', 'Z']
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0] = ['B', 'UP', 'RIGHT', 'Y']
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] = ['B', 'X']
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0] = ['LEFT', 'RIGHT', 'X']
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1] = ['B', 'UP', 'LEFT', 'Z']
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] = ['B', 'DOWN', 'C']
                [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0] = ['A', 'LEFT', 'RIGHT', 'X']
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0] = ['A', 'RIGHT', 'C']
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] = ['UP', 'C']
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] = ['B', 'UP', 'C']
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0] = ['A', 'RIGHT', 'X']
              ]
'''

#["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
action_sequence_list = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       ]

def main():
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')
    #env = SonicDiscretizer(env)

    action_index = 0

    obs = env.reset()
    while True:
        #action = env.action_space.sample()
        action = random.choice(action_list)
        action = action_sequence_list[action_index]
        #print("action: ", action)

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
