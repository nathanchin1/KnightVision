# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import fen_to_vector


#################eval models
class KingsVisionNN2(nn.Module):
    def __init__(self):
        super(KingsVisionNN2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        #print(f"Shape before flattening: {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(f"Shape2 before flattening: {x.shape}")
        x = x.view(x.size(0),-1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        #print(f"Shape3 before flattening: {x.shape}")
        x = self.fc2(x)
        return x

####################################top moves models

import torch
import torch.nn as nn
import torch.nn.functional as F

class KnightVisionNN(nn.Module):
    def __init__(self, num_moves):
        super(KnightVisionNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.fc1_in_features = 128 * 8 * 8
        self.fc1 = nn.Linear(self.fc1_in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_moves)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = self.fc4(x)
        
        return x



