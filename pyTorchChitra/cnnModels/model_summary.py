import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def displayModelSummary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))