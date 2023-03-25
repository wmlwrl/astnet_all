import torch
path = '/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/ped2.pth'
pretrained_dict = torch.load(path)
for k, v in pretrained_dict.items():  # k 参数名 v 对应参数值
        print(k)
