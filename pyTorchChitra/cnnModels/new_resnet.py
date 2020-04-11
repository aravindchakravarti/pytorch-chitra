import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class NewResNet(nn.Module):
    def __init__(self):
        super(NewResNet, self).__init__()
        
        #------ Preperation Layer
        self.prep_conv = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.prep_bn  = nn.BatchNorm2d(num_features = 64)
     
        #------ First Layer
        self.conv_L1 = nn.Conv2d(64, 128, 3, bias=False, padding=1)
        self.bn_L1  = nn.BatchNorm2d(num_features = 128)
        self.mp_L1  = nn.MaxPool2d(2, 2) 
        
        self.conv_R1 = nn.Conv2d(128,128, 3, bias=False, padding=1)
        self.bn_R1  = nn.BatchNorm2d(num_features = 128)

        #------ Second Layer
        self.conv_L2 = nn.Conv2d(128, 256, 3, bias=False, padding=1)
        self.bn_L2  = nn.BatchNorm2d(num_features = 256)
        self.mp_L2  = nn.MaxPool2d(2, 2)

        #------ Third Layer
        self.conv_L3 = nn.Conv2d(256, 512, 3, bias=False, padding=1)
        self.bn_L3  = nn.BatchNorm2d(num_features = 512)
        self.mp_L3  = nn.MaxPool2d(2, 2) 
        
        self.conv_R2 = nn.Conv2d(512,512, 3, bias=False, padding=1)
        self.bn_R2  = nn.BatchNorm2d(num_features = 512)

        #------ Fourth Layer
        self.mp_L4 = nn.MaxPool2d(4,4)

        #------ FC Layer
        self.convFC = nn.Conv2d(512, 10, 1, bias=False, padding=0)

    def forward(self, img):
        
        #------ Preperation Layer
        prep_layer = F.relu(self.prep_bn(self.prep_conv(img)))

        #------ First Layer
        X = F.relu(self.bn_L1(self.mp_L1(self.conv_L1(prep_layer))))

        R1 = F.relu(self.bn_R1(self.conv_R1(X)))      # ---------
        R1 = self.bn_R1(self.conv_R1(R1))             #          |_ ResNet Block
                                                      #          |
        R1 = F.relu(R1)                               #  --------
        
        layer_1 = X + R1

        #------ Second Layer
        layer_2 = F.relu(self.bn_L2(self.mp_L2(self.conv_L2(layer_1))))

        #------ Third Layer
        X = F.relu(self.bn_L3(self.mp_L3(self.conv_L3(layer_2))))

        R2 = F.relu(self.bn_R2(self.conv_R2(X)))      #  ----------
        R2 = self.bn_R2(self.conv_R2(R2))             #            |_ ResNet Block
                                                      #            |
        R2 = F.relu(R2)                               #  ----------

        layer_3 = X + R2

        #------- Fourth Layer
        Layer_4 = self.mp_L4(layer_3)

        #------- FC layer
        Layer_4 = self.convFC(Layer_4)
        Layer_4 = Layer_4.view(-1, 10)                           # Don't want 10x1x1..

        return F.log_softmax(Layer_4)
       
       
def PrintMyFileVersion():
    print('Hello there it is 1.4')
        