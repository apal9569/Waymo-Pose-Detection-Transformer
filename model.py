
import torch
import torch.nn as nn
import torch.optim as optim

class cnnLayer(nn.Module):
    def __init__(self):
        super(cnnLayer, self).__init__()
       
        self.cnn1 = nn.Conv2d(3, 32, 5, stride = 2, padding=2)
        self.cnn2 = nn.Conv2d(32, 64, 5, stride = 2, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(64, 64, 5, stride = 2, padding=2)
        self.cnn4 = nn.Conv2d(64, 64, 5, stride = 2, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.cnn5 = nn.Conv2d(64, 64, 5, stride = 2, padding=2)
        self.cnn6 = nn.Conv2d(64, 64, 5, stride = 2, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.cnn7 = nn.Conv2d(64, 64, 5)
        self.cnn8 = nn.Conv2d(64, 64, 5)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.cnn9 = nn.Conv2d(64, 64, 5)
        self.cnn10 = nn.Conv2d(64, 64, 5)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.cnn11 = nn.Conv2d(64, 32, 5)
        self.cnn12 = nn.Conv2d(32, 16, 5)
        
        self.linear1 = nn.Linear(9216, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear3 = nn.Linear(1024, 128)
        
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
        cnn1 = self.cnn1(input)
        cnn2 = self.cnn2(cnn1)
        batch1 = self.batchnorm1(cnn2)
        cnn3 = self.cnn3(batch1)
        cnn4 = self.cnn4(cnn3)
        batch2 = self.batchnorm2(cnn3)
        cnn5 = self.cnn5(batch2)
        cnn6 = self.cnn6(cnn5)
        batch3 = self.batchnorm3(cnn6)
        cnn7 = self.cnn7(batch3)
        cnn8 = self.cnn8(cnn7)
        batch4 = self.batchnorm4(cnn8)
        cnn9 = self.cnn9(batch4)
        cnn10 = self.cnn10(cnn9)
        batch5 = self.batchnorm5(cnn10)
        cnn11 = self.cnn11(batch5)
        cnn12 = self.cnn12(cnn11)

        x = cnn12.view(-1, 16*16*36)
        # print(x.shape)

        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)

        o1 = self.linearo1(l3)
        o2 = self.linearo2(l3)
        o3 = self.linearo3(l3)
        o4 = self.linearo4(l3)
        o5 = self.linearo5(l3)
        o6 = self.linearo6(l3)
        o7 = self.linearo7(l3)
        o8 = self.linearo8(l3)
        o9 = self.linearo9(l3)
        o10 = self.linearo10(l3)
        o11 = self.linearo11(l3)
        o12 = self.linearo12(l3)
        o13 = self.linearo13(l3)
        o14 = self.linearo14(l3)
        
        return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14])