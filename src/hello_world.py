import gym
import minerl
from collections import OrderedDict
from collections import namedtuple, deque
import os
import logging
import time
import datetime
import torch
from torch import nn
import torch.nn.functional as F
import random

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import subprocess
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

minerl_environments = {"tree": "MineRLTreechop-v0",
                       "nav_dense": "MineRLNavigateDense-v0",
                       "nav": "MineRLNavigate-v0",
                       "nav_extreme_dense": "MineRLNavigateExtremeDense-v0",
                       "nav_extreme": "MineRLNavigateExtreme-v0",
                       "pickaxe": "MineRLObtainIronPickaxe-v0",
                       "pickaxe_dense": "MineRLObtainIronPickaxeDense-v0",
                       "diamond": "MineRLObtainDiamond-v0",
                       "diamond_dense": "MineRLObtainDiamondDense-v0"}


def save_image_to_disk(obs, name):
    folder_images = f"{SCRIPT_DIR}/images/"
    if not os.path.exists(folder_images):
        os.makedirs(folder_images)
    im = Image.fromarray(obs)
    im.save(f"{folder_images}/your_file_{name}.jpeg")


def get_default_action():
    action = OrderedDict()
    action["ESC"] = np.array(0, dtype=np.int64)
    action["attack"] = np.array(0, dtype=np.int64)
    action["back"] = np.array(0, dtype=np.int64)
    action["camera"] = np.array([0.0, 0.0], dtype=np.float32)
    action["drop"] = np.array(0, dtype=np.int64)
    action["forward"] = np.array(0, dtype=np.int64)
    for i in range(1, 10):
        action[f"hotbar.{i}"] = np.array(0, dtype=np.int64)
    action["inventory"] = np.array(0, dtype=np.int64)
    action["jump"] = np.array(0, dtype=np.int64)
    action["left"] = np.array(0, dtype=np.int64)
    action["pickItem"] = np.array(0, dtype=np.int64)
    action["right"] = np.array(0, dtype=np.int64)
    action["sneak"] = np.array(0, dtype=np.int64)
    action["sprint"] = np.array(0, dtype=np.int64)
    action["swapHands"] = np.array(0, dtype=np.int64)
    action["use"] = np.array(0, dtype=np.int64)
    return action


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_obs_as_batch(obs):
    obs0 = obs["pov"]
    obs0 = obs0.transpose((2, 0, 1))
    screen = np.ascontiguousarray(obs0, dtype=np.float32) / 255
    obs0 = torch.from_numpy(screen)
    return torch.unsqueeze(obs0, 0)


class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.LazyLinear(num_actions)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(batch_size, -1))


def tst_hello_world(render=True):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.basicConfig(level=logging.DEBUG)

    env = gym.make('MineRLBasaltFindCave-v0')
    print(f"gym make {time.time() - start_time}")
    # Note that this command will launch the MineRL environment, which takes time.
    # Be patient!
    obs = env.reset()
    image = get_obs_as_batch(obs)
    print(f"env.reset {time.time() - start_time}")

    done = False
    agent = DQN(num_actions=24)
    with torch.no_grad():
        out = agent(image)
    action = get_default_action()

    i = 0
    start_loop = time.time()
    while not done:
        if i % 100 == 0:
            print(f"{i} => {time.time() - start_time} average={(time.time() - start_loop) / (i + 1)} ")

        # Take a random action
        action = env.action_space.sample()
        # In BASALT environments, sending ESC action will end the episode
        # Lets not do that
        action["ESC"] = 0
        obs, reward, done, _ = env.step(action)
        if render:
            save_image_to_disk(obs["pov"], f"{i}")
        i = i + 1


def operate_xvfb_if_needed():
    print(f"platform={platform.system()}")
    if platform.system() == "Darwin":
        return
    xvfb = subprocess.Popen(['Xvfb', ':99'])
    os.environ["DISPLAY"] = ":99"


if __name__ == "__main__":
    folder = "./logs_minerl"
    log_file = f'{folder}/debug_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
    if not os.path.exists(folder):
        os.makedirs(folder)
    logging.basicConfig(level=getattr(logging, "DEBUG"),
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    operate_xvfb_if_needed()
    tst_hello_world()
