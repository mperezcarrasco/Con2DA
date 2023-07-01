import numpy as np
import os
import os.path
from PIL import Image


def load_img(path):
    """Return an image given its respective path."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def data_fromlist(img_list):
    """Return a list of images paths and their respective labels"""
    with open(img_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(img_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list
