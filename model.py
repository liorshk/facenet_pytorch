import torch
import torch.nn as nn
from torchvision.models import resnet18
import math

class FaceModel(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False):
        super(FaceModel, self).__init__()

        self.model = resnet18(pretrained)

        self.embedding_size = embedding_size

        self.model.fc = nn.Linear(512*3*3, self.embedding_size)

        self.model.classifier = nn.Linear(self.embedding_size, num_classes)


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res
