import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Adapted for CIFAR: 3x3 kernel, stride 1
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Adapted for CIFAR: Output size is 4x4 here if input is 32x32
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def get_resnet(model_name, pretrained=False, imagenet=False):

    if model_name.startswith("ResNet18"):
        net = ResNet18()
    elif model_name.startswith("ResNet34"):
        net = ResNet34()
    elif model_name.startswith("ResNet50"):
        net = ResNet50()
    elif model_name.startswith("ResNet101"):
        net = ResNet101()
    elif model_name.startswith("ResNet152"):
        net = ResNet152()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return net


# net = ResNet34()
#
#
# def set_sparsity(model, ratio):
#     p_total = 0
#     p_used = 0
#     dystil_exclude = ["conv1.weight"]
#     for name, p in model.named_parameters():
#         p_total += p.numel()
#         if not ("bias" in name or name in dystil_exclude or "bn" in name):
#             p.dystil_k = int(p.numel() * ratio)
#             print(name, p.numel(), p.shape, p.dystil_k)
#             p_used += p.dystil_k
#         else:
#             p_used += p.numel()
#
#     print("Target Sparsity:", 1 - p_used / p_total)
#
#
# from tutils import (
#     initialize_sparse,
#     print_model_sparsity,
#     calculate_sparsity_per_layer,
# )
#
# set_sparsity(net, 0.125)
# print_model_sparsity(net)
# initialize_sparse(net)
# # print_model_sparsity(net)
#
# overall_sparse, _ = calculate_sparsity_per_layer(net)
# print(f"Overall sparsity: {overall_sparse:.2f}%")


# print(ResNet34())
# model = models.resnet34(
#     weights=None, num_classes=10
# )  # Ensure not pretrained, set num_classes
# # print(model)
# print(model(torch.randn(1, 3, 228, 228)).shape)
#
# # --- Modify the stem for CIFAR-10 ---
# # Replace the first conv layer
# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# # Remove the initial max pooling layer (replace with an identity layer)
# model.maxpool = nn.Identity()
#
# print(model(torch.randn(1, 3, 32, 32)).shape)
# # print(model)
# # print(list(model.parameters()))
# # print(list(model.modules()))
# from torchsummary import summary
#
# summary(model, (3, 32, 32))
