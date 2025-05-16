from .wideresnet import WideResNet
from .resnets import get_resnet

def get_model(model_name:str):
    if model_name.startswith("WideResNet-22"):
        return WideResNet()
    else:
        return get_resnet(model_name)

