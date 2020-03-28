import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class plainArch(nn.Module):
    def __init__(self):
        super(plainArch, self).__init__()
        self.conv01 = nn.Conv2d(3, 16, 3, bias=False, padding=1)        
        self.batch01 = nn.BatchNorm2d(num_features=16)    

        # ---- Lets take a skip connection
        self.skip_conv1 = nn.Conv2d(16, 16, 3, padding=0, dilation=2)
        
        self.conv02 = nn.Conv2d(16, 16, 3, bias=False,padding=1)        
        self.batch02 = nn.BatchNorm2d(num_features=16)    
        self.conv03 = nn.Conv2d(16, 16, 3, bias=False,padding=1)       
        self.batch03 = nn.BatchNorm2d(num_features=16)   
        self.conv04 = nn.Conv2d(16, 16, 3, bias=False,padding=1)        
        self.batch04 = nn.BatchNorm2d(num_features=16)   
        self.pool01 = nn.MaxPool2d(2, 2)                                #O=16
        self.conv05 = nn.Conv2d(16, 16, 1, bias=False)
        
        self.conv11 = nn.Conv2d(16, 32, 3, bias=False, padding=1)       
        self.batch11 = nn.BatchNorm2d(num_features=32)
        self.conv12 = nn.Conv2d(32, 32, 3, bias=False, padding=1)      
        self.batch12 = nn.BatchNorm2d(num_features=32)
        self.conv13 = nn.Conv2d(32, 32, 3, bias=False, padding=1)       
        self.batch13 = nn.BatchNorm2d(num_features=32)
        self.conv14 = nn.Conv2d(32, 32, 3, bias=False, padding=1)       
        self.batch14 = nn.BatchNorm2d(num_features=32)
        self.pool11 = nn.MaxPool2d(2, 2)                                #O=8
        self.conv15 = nn.Conv2d(32, 32, 1, bias=False)

        self.conv21 = nn.Conv2d(32, 64, 3, bias=False, padding=1)       
        self.batch21 = nn.BatchNorm2d(num_features=64)
        self.conv22 = nn.Conv2d(64, 64, 3, bias=False, padding=1)       
        self.batch22 = nn.BatchNorm2d(num_features=64)
        self.conv23 = nn.Conv2d(64, 64, 3, bias=False, padding=1)       
        self.batch23 = nn.BatchNorm2d(num_features=64)
        self.conv24 = nn.Conv2d(64, 64, 3, bias=False, padding=1)       
        self.batch24 = nn.BatchNorm2d(num_features=64)
        self.pool21 = nn.MaxPool2d(2, 2)                                #O=4
        self.conv25 = nn.Conv2d(64, 64, 1, bias=False)

        self.conv31 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, bias = False, padding = 1)
        self.convPV1= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias = False, padding = 0)       
        self.batch31 = nn.BatchNorm2d(num_features=128)
        self.conv32 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, bias = False, padding = 1)
        self.convPV2= nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, bias = False, padding = 0)       
        self.batch32 = nn.BatchNorm2d(num_features=256)

       
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.convx3 = nn.Conv2d(256, 10, 1, bias=False, padding=0)

    def forward(self, x):
        x = self.batch01(F.relu(self.conv01(x)))

        # ---- Lets take a skip connection
        skip_channels = self.skip_conv1(self.skip_conv1(self.skip_conv1(self.skip_conv1(x))))

        x = self.batch02(F.relu(self.conv02(x)))
        x = self.batch03(F.relu(self.conv03(x)))
        x = self.batch04(F.relu(self.conv04(x)))
        x = self.pool01(x)
        x = self.conv05(x)
        # ----------------------------------------------------------
        
        # ---- Lets add the skip connection here
        x = skip_channels + x

        x = self.batch11(F.relu(self.conv11(x)))
        x = self.batch12(F.relu(self.conv12(x)))
        x = self.batch13(F.relu(self.conv13(x)))
        x = self.batch14(F.relu(self.conv14(x)))
        x = self.pool11(x)
        x = self.conv15(x)
        # ----------------------------------------------------------

        x = self.batch21(F.relu(self.conv21(x)))
        x = self.batch22(F.relu(self.conv22(x)))
        x = self.batch23(F.relu(self.conv23(x)))
        x = self.batch24(F.relu(self.conv24(x)))
        x = self.pool21(x)
        x = self.conv25(x)
        # ----------------------------------------------------------

        x = self.batch31(F.relu(self.convPV1(F.relu(self.conv31(x)))))
        x = self.batch32(F.relu(self.convPV2(F.relu(self.conv32(x)))))


        x = self.avg_pool(x)
        x = self.convx3(x)
        x = x.view(-1, 10)                           # Don't want 10x1x1..
        return F.log_softmax(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def displayModelSummary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))