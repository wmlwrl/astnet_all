import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor


input_grad = []
output_grad = []

def save_gradient(module, grad_input, grad_output):
    input_grad.append(grad_input)
    # print(f"{module.__class__.__name__} input grad:\n{grad_input}\n")
    output_grad.append(grad_output)
    # print(f"{module.__class__.__name__} output grad:\n{grad_output}\n")


def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_feature_map(model, input_tensor):

    x = model.conv1(input_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


# prepare image
image_path = '/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/000.jpg'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# input = torch.rand([8, 3, 224, 224], dtype=torch.float, requires_grad=True)

model = resnet50(pretrained=True)
last_layer = model.layer4[-1]
last_layer.conv3.register_full_backward_hook(save_gradient)
# print(model)

# get output and feature
feature = get_feature_map(model, input)
output = model(input)
# print("feature.shape:", feature.shape)  # torch.Size([8, 2048, 7, 7])
# print("output.shape:", output.shape)    # torch.Size([8, 1000])

# cal grad
output[0][0].backward()
gard_info = input.grad
# print("gard_info.shape: ", gard_info.shape)     # torch.Size([8, 3, 224, 224])

# print("input_grad.shape:", input_grad[0][0].shape)      # torch.Size([8, 512, 7, 7])
# print("output_grad.shape:", output_grad[0][0].shape)    # torch.Size([8, 2048, 7, 7])

feature_grad = output_grad[0][0]
feature_weight = einops.reduce(feature_grad, 'b c h w -> b c', 'mean')
grad_cam = feature * feature_weight.unsqueeze(-1).unsqueeze(-1)     # (b c h w) * (b c 1 1) -> (b c h w)
grad_cam = F.relu(torch.sum(grad_cam, dim=1)).unsqueeze(dim=1)      # (b c h w) -> (b h w) -> (b 1 h w)
grad_cam = F.interpolate(grad_cam, size=(224, 224), mode='bilinear')    # (b 1 h w) -> (b 1 224 224) -> (224 224)
grad_cam = grad_cam[0, 0, :]
# print(grad_cam.shape)    # torch.Size([224, 224])

cam_image = show_cam_on_image(rgb_img, grad_cam.detach().numpy())
cv2.imwrite('./result/test.jpg', cam_image)
