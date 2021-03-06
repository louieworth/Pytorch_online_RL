import torch
import torch.nn as nn
import torch.nn.functional as F

# the convolution layer of deepmind
class Deepmind(nn.Module):
    def __init__(self):
        super(Deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)

        nn.init.orthogonal_(self.conv1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.orthogonal_(self.conv2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.orthogonal_(self.conv3.weight, gain=nn.init.calculate_gain("relu"))

        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x



class Net(nn.Module):
    def __init__(self, num_actions):
        super(net, self).__init__()
        self.cnn_layer = Deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)

        nn.init.constant_(self.critic.bias.data, 0)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, x):
        x = self.cnn_layer(x / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi







