"""
This is the implementation of the hierarchical classification technique proposed in 
La Grassa, R., Gallo, I. & Landro, N. Learn class hierarchy using convolutional 
neural networks. Appl Intell 51, 6622-6632 (2021). 
doi: https://doi.org/10.1007/s10489-020-02103-6
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CenterLossNN(nn.Module):
    def __init__(self, num_classes, num_classes_per_level, lambda_values):
        super(CenterLossNN, self).__init__()
        self.num_classes = num_classes
        self.lambda_values = lambda_values

        self.base_model = models.resnet18(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.feature_dim = self.base_model.fc.in_features

        # self.base_model.fc = nn.Identity()  # Replace the final fully connected layer

        # Get feature dimension from base model output
        base_feature_dim = self.base_model.fc.in_features
        self.fc_layers = nn.ModuleList([])
        for num_class in num_classes_per_level:
            self.fc_layers.append(nn.Linear(base_feature_dim, num_class))
            base_feature_dim = num_class

    def forward(self, x, labels):
        features = self.base_model(x)

        batch_centers = torch.zeros(self.num_classes, self.feature_dim, device=x.device)
        for i in range(self.num_classes):
            class_mask = (labels == i)
            if class_mask.any():
                batch_centers[i] = features[class_mask].mean(dim=0)

        center_loss = torch.tensor(0.0, device=x.device)
        for i in range(features.size(0)):
            center_loss += torch.sum((features[i] - batch_centers[labels[i].item()]) ** 2)

        classification_losses = []
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            loss = nn.CrossEntropyLoss()(x, labels)
            classification_losses.append(loss)

        classification_losses = torch.stack(classification_losses)

        total_loss = self.lambda_values[0] * center_loss + torch.dot(self.lambda_values[1:], classification_losses)

        return total_loss
