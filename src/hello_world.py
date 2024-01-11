import gym
import minerl
import os
import logging
import time
import datetime


def tst_hello_world(render=False):
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
            env.render()
        i = i + 1


if __name__ == "__main__":
    folder = "./logs_minerl"
    log_file = f'{folder}/debug_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
    if not os.path.exists(folder):
        os.makedirs(folder)
    logging.basicConfig(level=getattr(logging, "DEBUG"),
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    tst_hello_world()
