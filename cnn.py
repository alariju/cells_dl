from torch import nn
from torchvision import models
import torch.nn.functional as F


class DenseNET121(nn.Module):
    def __init__(self):
        super(DenseNET121, self).__init__()
        self.dense_net_121 = models.densenet121(pretrained=True)
        for param in self.dense_net_121.parameters():
            param.requires_grad = True
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(1024, 256),
                                                      nn.ReLU(),
                                                      nn.Dropout(0.2),
                                                      nn.Linear(256, 2),
                                                      nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        output = self.dense_net_121(input_data)
        return output


class ResNET50(nn.Module):

    def __init__(self):
        super(ResNET50, self).__init__()
        res_net_50 = models.resnet50(pretrained=True)
        self.res_net_50_convolution = nn.Sequential(*list(res_net_50.children())[:-2])
        for param in self.res_net_50_convolution.parameters():
            param.requires_grad = True
        self.fully_connected_layer = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.res_net_50_convolution(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(-1, 2048)
        x = F.relu(self.fully_connected_layer(x))
        x = F.dropout(x, training=self.training)
        output = F.log_softmax(x, dim=1)
        return output
