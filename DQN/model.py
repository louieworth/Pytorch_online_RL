import torch
import torch.nn as nn
import torch.nn.functional as F

# the convolution layers
class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain("relu"))
        self.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain("relu"))
        self.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain("relu"))

        self.init.constant_(self.conv1.bias.data, 0)
        self.init.constant_(self.conv2.bias.data, 0)
        self.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)

        return x

    def num_flat_features(selfself, x):
        size = x.size()[1:]
        num_features = 1
        for feature in num_features:
            num_features *= feature
        return num_features

class net(nn.Module):
    def __init__(self, num_actions, use_dueling=False):
        super(net, self).__init__()
        # if use the dueling network
        self.use_dueling = use_dueling
        self.cnn_layer = ConvLayer()
        if not use_dueling:
            self.fc = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
        else:
            self.action_fc = nn.Linear(32 * 7 * 7, 256)
            self.state_fc = nn.Linear(32 * 7 * 7 , 256)
            self.action_value = nn.Linear(256, num_actions)
            self.state_value = nn.Linear(256, 1)

    def forward(self, x):
        # TODO
        x = self.cnn_layer(x / 255.0)
        if not self.use_dueling:
            x = F.relu(self.fc(x))
            action_value_out = F.relu(self.action_value(x))
        else:
            # compute action_value
            action_fc = F.relu(self.action_fc(x))
            action_value = F.relu(self.action_value(action_fc))

            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean

            state_fc = F.relu(self.state_fc(x))
            state_value = F.relu(self.state_value(state_fc))
            # Q = V + q - q.mean()
            action_value_out = action_value_center + state_value

        return action_value_out











