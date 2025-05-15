import torch
import models

net = models.ResNet34().cuda()

state = torch.load("./checkpoint/ResNet34_CIFAR10_IHT_0.1_best.pth")
net.load_state_dict(state["net"])
state["acc"]


from tutils import calculate_sparsity_per_layer

res, per_layer = calculate_sparsity_per_layer(net)
res
per_layer
sparse_weights  = [l for l in per_layer if per_layer[l]>0]
n = sparse_weights[0].split("_")[1]
n

state["net"].keys()
[k for k in state["net"].keys() if "shortcut" in k]
[ p[0] for p in net.named_parameters() if "shortcut" in p[0]]

w = state["net"][n]
w.shape
s = w.abs().sum(dim=(2,3))
s.shape
z = w.abs().sum(dim=(1,2,3))

s.sum(0)
s.sum(1)
net
s.sum(0)



torch.sum(s.sum(0)>0)
torch.sum(s.sum(1)>0)
b = state["net"][n.replace("weight", "bias")]
# w = torch.clip(w,min=1e-10)
w.masked_fill_(w.abs()<1e-10, 0.0)

torch.sum(s.sum(0)>0)

torch.where(s.sum(0)>0)

w[1]
s[:, 1]
w[2,1]
w[2]


[n[0] for n in net.named_modules()]

a= list(net.layer2.modules())
a
short = net.layer2[0].shortcut
net.layer2[0]
w =net.layer2[0].conv1.weight.detach()
net.layer2[0]
w_s = short[0].weight.detach()
w_s.masked_fill_(w_s.abs()<1e-10, 0.0)
torch.where(w_s.abs().sum((2,3)).sum(1)> 0)
torch.sum(w_s.abs().sum((2,3)).sum(1)> 0)

w.masked_fill_(w.abs()<1e-10, 0.0)
s = w.abs().sum((2,3))
torch.where(s.sum(0)>0)
torch.where(w_s.abs().sum((2,3)).sum(0)> 0)
torch.sum(s.sum(0)>0)


torch.sum(s>0)
s.numel()
torch.sum(w.abs()>0)/w.numel()


from torchsummary import summary
summary(net)
print(res)
res*21282122/100.0
