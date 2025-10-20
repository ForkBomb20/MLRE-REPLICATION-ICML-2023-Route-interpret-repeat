import torch
from torchvision import models


class G(torch.nn.Module):
    def __init__(
            self,
            pre_trained,
            model_choice,
            dataset,
            hidden_nodes,
            concept_size=0
    ):
        super(G, self).__init__()

        self.g_model_op_size = self._get_op_size(pre_trained, model_choice)
        self.g_model_ip_size = self._get_ip_size(dataset, model_choice, concept_size)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.g_model_ip_size, out_features=hidden_nodes, bias=True),
            # torch.nn.Linear(in_features=self.g_model_ip_size, out_features=self.g_model_op_size, bias=True),
            # torch.nn.Linear(in_features=self.g_model_op_size, out_features=self.g_model_op_size, bias=True),
            torch.nn.Linear(in_features=hidden_nodes, out_features=self.g_model_op_size, bias=True)
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _get_op_size(pre_trained, model_choice):
        if model_choice == "ResNet50":
            return models.resnet101(pretrained=pre_trained).fc.weight.shape[1] * 7 * 7
        elif model_choice == "ResNet101":
            return models.resnet101(pretrained=pre_trained).fc.weight.shape[1] * 14 * 14

    @staticmethod
    def _get_ip_size(dataset, model_choice, concept_size=0):
        if dataset == "cub" and model_choice == "ResNet101":
            return 21168
