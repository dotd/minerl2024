import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.mdps.pick_and_go import PickAndGo


def tst_PickAndGo(skip_frames=3):
    env = PickAndGo()
    for episode in range(5):
        print(f"episode={episode}")
        obs = env.reset()
        plt.imshow(np.transpose(obs, (1, 2, 0)))
        for i in tqdm(range(1000)):
            action = env.get_random_action()
            state_next, reward, done, info = env.step(action)
            if i % skip_frames == 0:
                plt.imshow(np.transpose(state_next, (1, 2, 0)))
                plt.show(block=False)
                plt.pause(0.001)


if __name__ == "__main__":
    tst_PickAndGo()
