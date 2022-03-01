import retro
import gym
import numpy as np
import os
import time
import glob
import random


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

              }


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

        env.render()
        #time.sleep(0.01)

        saved_state = env.em.get_state()

'''
def main():
    env = retro.make(game='SonicTheHedgehog2-Genesis', scenario='contest', state='EmeraldHillZone.Act1')
    env = SonicDiscretizer(env)

    obs = env.reset()
    while True:
        action = env.action_space.sample()
        print("action: ", action)

        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
'''