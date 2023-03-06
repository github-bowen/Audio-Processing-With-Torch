from torch import nn
from torchsummary import summary
import urban_sound_dataset


class CNNNetwork(nn.Module):

    def __init__(self):
        super(CNNNetwork, self).__init__()
        # 4 conv blocks / flatten / linear / softmax

        # 4 convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # equal to the output of self.conv1
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # flatten
        self.flatten = nn.Flatten()

        # linear
        self.linear = nn.Linear(
            in_features=128 * 5 * 4,
            out_features=10  # 10 class
        )

        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        flattened_data = self.flatten(x)
        logits = self.linear(flattened_data)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()

    # (channels, frequency, time)
    summary(cnn, input_size=(1, 64, 44))
