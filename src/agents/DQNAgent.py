import numpy as np
import pickle
import time as tim

import torch
import torch.nn.functional as F

from src.agents.ReplayBuffers import ReplayBuffer


class DQNDiscreteContinuous:
    """
    Some notes about the agent:
    1) Only receives tensors in the image form (i.e., channel, y, x)
    2) Only get tensors. No lists nor numpy arrays. The runner should do all the tranlations.
    """

    def __init__(self,
                 seed,
                 num_actions_discrete,
                 gamma,
                 eps_greedy,
                 policy_net_lr,
                 replay_buffer_size,
                 batch_size,
                 policy_network,
                 update_target_period
                 ):

        self.random = np.random.RandomState(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={self.device}")

        # network
        self.policy_net = policy_network.to(self.device)
        self.target_net = pickle.loads(pickle.dumps(self.policy_net)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.eps_greedy = eps_greedy

        self.num_actions = num_actions
        self.gamma = gamma  # gamma of the Belmman Equation

        self.policy_net_lr = policy_net_lr

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_net_lr)
        self.last_time_saved_model = tim.time()
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)
        self.batch_size = batch_size
        self.update_target_counter = 0
        self.update_target_period = update_target_period
        self.loss = None

    def choose_action(self, state_tensor):
        """
        :param state_tensor: get the tensor shape, i.e., (c,y,x) shape.
        :return: action: Tensor long
        """
        if self.random.uniform() < self.eps_greedy.get_next_eps():
            return torch.tensor([[self.random.choice(self.num_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # REMOVE: # We do xxx.unsqueeze(0) for making a batch of a single sample. Result: 4 dim state_tensor
                # REMOVE: state_tensor = state_tensor
                actions = self.policy_net(state_tensor)
                action = actions.max(1)[1].view(1, 1)
                return action

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        samples = self.replay_buffer.sample(self.batch_size)
        # Do transpose
        samples = list(zip(*samples))
        # Note: stack/cat - Concatenates sequence of tensors along a new/given dimension.
        states = torch.cat(samples[0])
        actions = torch.cat(samples[1])
        rewards = torch.cat(samples[2])
        states_next = torch.cat(samples[3])
        dones = torch.cat(samples[4])

        Q = self.policy_net(states).gather(1, actions)
        Q_next = self.target_net(states_next).max(1)[0].detach().unsqueeze(1)
        expected_Q = (1 - dones) * (Q_next * self.gamma) + rewards
        # Compute Huber loss
        loss = F.smooth_l1_loss(Q, expected_Q)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_target_network()
        return loss.item()

    def update_target_network(self):
        self.update_target_counter += 1
        if self.update_target_counter > self.update_target_period:
            # print("update_target_counter")
            self.update_target_counter = 0
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # self.target_net.eval()

    def add_to_replay_buffer(self, sample):
        """
        We add the sample with the following convention:
        [state, action, reward, state_next, done].
        It is important since this is how the optimize_model method is processing them.
        :param sample: the sampe to push.
        :return: None
        """
        self.replay_buffer.push(sample)
