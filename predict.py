import torch
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib as plt
from matplotlib.pyplot import subplots
from .train import checkpoint_file
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

image_path = ''

def load_checkpoint(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location='cpu')
    model = models.densenet161(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    size = 256, 256
    crop_size = 224
    left = 0
    upper = 0
    with Image.open(image) as im:
        im.thumbnail(size)
        pil_image = im.crop((left, upper, crop_size, crop_size))
        np_image = np.array(pil_image).astype(np.float)
        np_image = (np_image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np_image = np_image.transpose(2, 0, 1)
    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = subplots()

    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        image = process_image(image_path)
        image_tensor = torch.from_numpy(image)
        log_ps = model(image_tensor.unsqueeze(0).to(device).float())
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class


probs, classes = predict(image_path, load_checkpoint(checkpoint_file))


def sanity_check(img, probs, classes):
    ps = probs.cpu()
    ps = ps.numpy().squeeze()
    fig, (ax1, ax2) = subplots(figsize=(10, 4), ncols=2)

    ax2_yticks = [cat_to_name[str(int(cat))] for cat in np.nditer(classes.cpu().numpy())]
    with Image.open(img) as im:
        ax1.imshow(im)

    ax1.axis('off')
    ax2.barh(np.arange(5), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(ax2_yticks)

    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 0.3)

    fig.tight_layout()


sanity_check(image_path, probs, classes)