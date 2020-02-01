import torch
import torchvision
# print(torch.cuda.is_available())
#
# a = torch.Tensor(5,3)
# a=a.cuda()
# print(a)
layers=[1,2,3,4,5,6,7,8,9]
for l in layers[::-1]:
    print(l)