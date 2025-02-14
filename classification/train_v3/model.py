import torchvision.models as models
import torch.nn as nn


def build_model(model = 'efficientnetb0',pretrained=True, fine_tune=True, num_classes=6):
    num_features = 0
    weights = None
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        if model == 'efficientnetb0':
            weights = models.EfficientNet_B0_Weights.DEFAULT
        elif model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
    else:
        print('[INFO]: Not loading pre-trained weights')

    # Choose model
    if model == 'efficientnetb0':
        model = models.efficientnet_b0(weights=weights)
        num_features = 1280
    elif model == 'resnet18':
        model = models.resnet18(weights=weights)
        num_features = 512
    else:
        raise AttributeError('Invalid model name')

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    return model