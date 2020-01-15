import torch
import torchvision
print(torch.cuda.is_available())

a = torch.Tensor(5,3)
a=a.cuda()
print(a)