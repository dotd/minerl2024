import torch
from torch import nn


class CustomNetwork(nn.Module):
    def __init__(self,
                 num_conv_layers,
                 input_channels,
                 input_width,
                 input_height,
                 intermediate_channels,
                 joint_linear_layers_sizes,
                 discrete_head_sizes,
                 continuous_head_sizes,
                 kernel_size=3,
                 padding=1):
        super(CustomNetwork, self).__init__()

        # Create convolutional layers
        conv_layers = []
        in_channels = input_channels
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = intermediate_channels
        self.conv_block = nn.Sequential(*conv_layers)

        # Calculate the size of the output from conv layers
        # This is needed to determine the input size for linear layers
        dummy_input = torch.randn(1, input_channels, input_height, input_width)
        dummy_output_conv = self.conv_block(dummy_input)
        conv_out_size = dummy_output_conv.view(dummy_output_conv.size(0), -1).size(1)

        # Create joint linear layers
        joint_linear_layers_sizes = [conv_out_size] + joint_linear_layers_sizes
        self.joint_linear_block = self.create_linear_block(joint_linear_layers_sizes, last_activation=nn.ReLU())

        # Create discrete linear layers
        discrete_linear_layers_sizes = [joint_linear_layers_sizes[-1]] + discrete_head_sizes
        self.discrete_linear_block = self.create_linear_block(discrete_linear_layers_sizes, last_activation=nn.Sigmoid())

        # Create continuous linear layers
        continuous_linear_layers_sizes = [joint_linear_layers_sizes[-1]] + continuous_head_sizes
        self.continuous_linear_block = self.create_linear_block(continuous_linear_layers_sizes)

    @staticmethod
    def create_linear_block(self, linear_layers_sizes, last_activation=None):
        joint_linear_layers = []
        for i in range(len(linear_layers_sizes) - 1):
            joint_linear_layers.append(nn.Linear(linear_layers_sizes[i], linear_layers_sizes[i + 1]))
            if i < len(linear_layers_sizes) - 2:
                joint_linear_layers.append(nn.ReLU())
            elif last_activation is not None:
                joint_linear_layers.append(last_activation)
        return nn.Sequential(*joint_linear_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.joint_linear_block(x)
        x1 = self.discrete_linear_block(x)
        x2 = self.continuous_linear_block(x)
        return x1, x2
