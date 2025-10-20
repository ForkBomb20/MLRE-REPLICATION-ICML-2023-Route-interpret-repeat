import torch.nn as nn
from torchvision import models


class Residual(nn.Module):
    def __init__(self, dataset, pre_trained=True, n_class=200, model_choice="ResNet50"):
        super(Residual, self).__init__()
        self.model_choice = model_choice
        self.feature_store = {}
        self.n_class = n_class
        self.feat_dim = self._model_choice(pre_trained, model_choice)
        self.avgpool = None
        self.fc = None
        self.dataset = dataset

        if dataset == "cub" and (model_choice == "ResNet50" or model_choice == "ResNet101"):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=n_class)

    def forward(self, x):
        if self.model_choice == "ResNet50" or self.model_choice == "ResNet101":
            if self.dataset == "cub":
                return self.fc(self.avgpool(x).reshape(-1, self.feat_dim * 1 * 1))

    @staticmethod
    def _model_choice(pre_trained, model_choice):
        if model_choice == "ResNet50":
            return models.resnet50(pretrained=pre_trained).fc.weight.shape[1]
        elif model_choice == "ResNet101":
            return models.resnet101(pretrained=pre_trained).fc.weight.shape[1]
