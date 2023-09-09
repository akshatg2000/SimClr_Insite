import torchvision
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.models import resnet50,ResNet50_Weights


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": resnet18(weights=ResNet18_Weights.DEFAULT),
        "resnet50": resnet50(weights=ResNet50_Weights.DEFAULT)
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
