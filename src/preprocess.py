from PIL import Image
import numpy as np
import torch

def preprocess_image(image: Image.Image):
    """Simple fix - just use original pixel values"""
    image = image.convert("L")
    image = image.resize((48, 48))
    img_array = np.array(image, dtype=np.float32)
    flat = img_array.flatten()
    tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
    return tensor
