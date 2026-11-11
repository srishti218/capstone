"""Architectures and preprocessing shared by training and inference."""
import numpy as np
import torch
from PIL import Image
from torchvision.models import DenseNet121_Weights, densenet121

from .dicom import to_gray_image

CLASS_SIZE = 224
ROI = (48, 64, 64)
CLASS_VERSION = "densenet121-rsna-v1"
SEG_VERSION = "monai-unet3d-msd-lung-v1"


def classifier(pretrained=False):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    return model


def segmenter():
    from monai.networks.nets import UNet

    return UNet(spatial_dims=3, in_channels=1, out_channels=1,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)


def resize_cxr(pixels):
    return to_gray_image(pixels).resize((CLASS_SIZE, CLASS_SIZE), Image.Resampling.BILINEAR)


def class_tensor(image):
    gray = torch.from_numpy(np.array(image, dtype=np.float32) / 255)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (gray[None] - mean) / std


def classify_with_cam(model, image):
    """Explain the predicted class at the last convolutional feature map."""
    activation = None

    def capture(_module, _inputs, output):
        nonlocal activation
        activation = output

    handle = model.features.norm5.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(image.requires_grad_(True))
            positive = bool(logits.detach().item() >= 0)
            score = logits if positive else -logits
            gradient, = torch.autograd.grad(score.sum(), activation)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (activation * weights).sum(dim=1).relu().detach()[0]
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-8)
            return float(logits.detach().sigmoid().item()), cam.cpu().numpy()
    finally:
        handle.remove()


@torch.inference_mode()
def segment_volume(model, image, device):
    from monai.inferers import sliding_window_inference

    return sliding_window_inference(
        image, ROI, 2, model, overlap=0.25, mode="gaussian",
        sw_device=device, device="cpu",
    ).sigmoid()
