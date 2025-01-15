import torch.nn as nn
import torch.nn.init as init


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(self.conv(x))