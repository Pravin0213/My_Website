import numpy as np
from PIL import Image
from io import BytesIO
import base64
import pickle
from scipy.ndimage import zoom

class Network:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a.flatten()
    

def center_image(pixels):
    """Center the digit within the 28x28 image."""
    img = pixels.reshape(280, 280)
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)

    # Find bounding box
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    # Crop the bounding box
    cropped = img[top:bottom+1, left:right+1]

    # Resize cropped image to smaller square
    from scipy.ndimage import zoom
    h, w = cropped.shape
    scale = 20 / max(h, w)
    resized = zoom(cropped, scale)

    # Pad back to 28x28, centered
    new_img = np.zeros((28, 28))
    h2, w2 = resized.shape
    top_pad = (28 - h2) // 2
    left_pad = (28 - w2) // 2
    new_img[top_pad:top_pad+h2, left_pad:left_pad+w2] = resized

    new_img = np.clip(new_img, 0, 1)
    new_img = np.round(new_img, 2)
    return new_img.reshape(-1, 1)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

# Load your weights and biases

import os

# Get the directory of the current file (model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'weights.pkl'), 'rb') as f:
    weights = pickle.load(f)
with open(os.path.join(BASE_DIR, 'biases.pkl'), 'rb') as f:
    biases = pickle.load(f)

net = Network(weights, biases)

