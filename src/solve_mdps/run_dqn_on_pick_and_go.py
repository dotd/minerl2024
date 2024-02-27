import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging

from src.networks.networks import NetworkConvDiscreteContinuous
from src.agents.DQNAgent import DQNDiscreteContinuous
from src.mdps.pick_and_go import PickAndGo


def run_dqn_on_pick_and_go():
    network = NetworkConvDiscreteContinuous(
        num_conv_layers=3,
        input_channels=3,
        input_width=100,
        input_height=100,
        intermediate_channels=16,
        joint_linear_layers_sizes=100,
        discrete_head_sizes=50,
        continuous_head_sizes=50
    )
    """
    network = DQNDiscreteContinuous(
        seed=1945,
        num_actions,
        gamma,
        eps_greedy,
        policy_net_lr,
        replay_buffer_size,
        batch_size,
        policy_network,
        update_target_period
    )
    """
    env = PickAndGo()
    for episode in range(5):
        print(f"episode={episode}")
        obs = env.reset()
        plt.imshow(np.transpose(obs, (1, 2, 0)))
        for i in tqdm(range(1000)):
            action = env.get_random_action()
            state_next, reward, done, info = env.step(action)
            if i % 3 == 0:
                plt.imshow(np.transpose(state_next, (1, 2, 0)))
                plt.show(block=False)
                plt.pause(0.001)


if __name__ == "__main__":
    """
    network_parser = argparse.ArgumentParser()
    # Create a group for input options
    network_group = network_parser.add_argument_group('Network Options')
    network_group.add_argument('--num_conv_layers', default=3, type=int, help='')
    network_group.add_argument('--input_channels', default=3, type=int, help='')
    network_group.add_argument('--input_width', default=100, type=int, help='')
    network_group.add_argument('--input_height', default=100, type=int, help='')
    network_group.add_argument('--intermediate_channels', default=16, type=int, help='')
    network_group.add_argument('--joint_linear_layers_sizes', nargs='+', type=int, default=[100, 100, 3],help='')
    network_group.add_argument('--discrete_head_sizes', nargs='+', type=int, default=[100, 100, 3],help='')
    network_group.add_argument('--continuous_head_sizes', nargs='+', type=int, default=[100, 100, 3],help='')
    network_group.add_argument('--kernel_size', default=3, type=int, help='')
    network_group.add_argument('--padding', default=1, type=int, help='')
    seed,
    num_actions,
    gamma,
    eps_greedy,
    policy_net_lr,
    replay_buffer_size,
    batch_size,
    policy_network,
    update_target_period
    arguments = parser.parse_args()
    """
    run_dqn_on_pick_and_go()
