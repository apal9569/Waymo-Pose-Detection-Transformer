
import torch
import torch.nn as nn
import torch.optim as optim

class outputLayer(nn.Module):
    def __init__(self):
        super(outputLayer, self).__init__()
        self.linearo1 = nn.Linear(128, 2)
        self.linearo2 = nn.Linear(128, 2)
        self.linearo3 = nn.Linear(128, 2)
        self.linearo4 = nn.Linear(128, 2)
        self.linearo5 = nn.Linear(128, 2)
        self.linearo6 = nn.Linear(128, 2)
        self.linearo7 = nn.Linear(128, 2)
        self.linearo8 = nn.Linear(128, 2)
        self.linearo9 = nn.Linear(128, 2)
        self.linearo10 = nn.Linear(128, 2)
        self.linearo11 = nn.Linear(128, 2)
        self.linearo12 = nn.Linear(128, 2)
        self.linearo13 = nn.Linear(128, 2)
        self.linearo14 = nn.Linear(128, 2)
        
    def forward(self, input):
        o1 = self.linearo1(input)
        o2 = self.linearo2(input)
        o3 = self.linearo3(input)
        o4 = self.linearo4(input)
        o5 = self.linearo5(input)
        o6 = self.linearo6(input)
        o7 = self.linearo7(input)
        o8 = self.linearo8(input)
        o9 = self.linearo9(input)
        o10 = self.linearo10(input)
        o11 = self.linearo11(input)
        o12 = self.linearo12(input)
        o13 = self.linearo13(input)
        o14 = self.linearo14(input)
        
        return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14])