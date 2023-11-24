from torch import nn
import torch


class ConstantBias(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        #self.linear = nn.Linear(channels*210*280, 4)
        # I was debugging that piece of code for 2 hours, cause of using
        # magic numbers antipattern and hardcoding value of channels near using
        # variable used for number of channels, take a look:
        # https://en.wikipedia.org/wiki/Magic_number_(programming)
        self.linear = nn.Linear(channels*210*280, channels)  
    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280))
        o_input = torch.split(x, 4, dim=-3)
        output = o_input[0].permute(3, 4, 0, 1, 2) + output
        return output.permute(2, 3, 4, 0, 1)
