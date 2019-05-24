from torch import nn
from torchvision import models
import torch.nn.functional as F


class ResNET50(nn.Module):

    def __init__(self):
        super(ResNET50, self).__init__()
        # se importan las capas de resnet 50, excepto el top model
        # se importan los pesos de imagenet
        res_net_50 = models.resnet50(pretrained=True)
        self.res_net_50_convolution = nn.Sequential(*list(res_net_50.children())[:-2])
        # se asegura que todas las capas sean entrenables
        for param in self.res_net_50_convolution.parameters():
            param.requires_grad = True
        # la salida del promediado global es 2048 promedios de los 2048 feature maps que genera resnet
        self.fully_connected_layer = nn.Linear(2048, 2)
        # 2 clases, perro o gato

        self.dense_net_121 = models.densenet121(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in self.dense_net_121.parameters():
            param.requires_grad = True

        self.dense_net_121.classifier = nn.Sequential(nn.Linear(1024, 256),
                                                      nn.ReLU(),
                                                      nn.Dropout(0.2),
                                                      nn.Linear(256, 2),
                                                      nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = self.dense_net_121(x)
        return output

    def forward2(self, x):
        # Apila las capas
        # Primero aplica las capas de Resnet50
        x = self.res_net_50_convolution(x)

        # outputs 2048 activation maps of 8x8
        # https://resources.wolframcloud.com/NeuralNetRepository/resources/ResNet-50-Trained-on-ImageNet-Competition-Data
        # Promediado de los feature maps
        x = F.avg_pool2d(x, x.size()[2:])
        # Aplanado
        x = x.view(-1, 2048)
        # Se pasa por la capa completamente conectada
        x = F.relu(self.fully_connected_layer(x))
        # Dropout para la regularizacion del modelo
        x = F.dropout(x, training=self.training)
        # Salida usando funcion softmax
        output = F.log_softmax(x, dim=1)
        return output
