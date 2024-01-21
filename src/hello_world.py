import gym
import minerl
import os
import logging
import time
import datetime
from matplotlib import pyplot as plt
from PIL import Image
import subprocess

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


def tst_hello_world(render=True):
    start_time = time.time()
    # logging.basicConfig(level=logging.DEBUG)

    env = gym.make('MineRLBasaltFindCave-v0')
    print(f"gym make {time.time() - start_time}")
    # Note that this command will launch the MineRL environment, which takes time.
    # Be patient!
    obs = env.reset()
    print(f"env.reset {time.time() - start_time}")

    done = False

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


def operate_xvfb():
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
    operate_xvfb()
    tst_hello_world()
