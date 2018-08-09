import glob
from functools import partial
import os

import cv2
import numpy as np
import torch

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.png'))

def get_dirs_in(path):
    dirs = [dir for dir in os.listdir(path) if not dir.startswith('.')]
    return np.sort(dirs)

def get_label_mapping():
    return {
        'real': 0,
        'spoof': 1
    }

def get_image_and_label_pairs(path):
    image_paths_and_labels = list()

    for label in get_dirs_in(path):
        image_paths = get_images_in(os.path.join(path, label))
        mapping = get_label_mapping()
        # pylint: disable=cell-var-from-loop
        image_paths_and_labels.extend(map(lambda path: [path, mapping[label]], image_paths))

    return image_paths_and_labels

def normalize(image):
    image = image.astype(np.float32)
    image /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    return image

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def read_image_cached(cache, preprocess, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = preprocess(read_image(path))
        cache[path] = image
        return image

def crop_random(size, image):
    top_x = np.random.randint(image.shape[0] - size[0])
    top_y = np.random.randint(image.shape[1] - size[1])
    return np.array(image[top_x:top_x + size[0], top_y:top_y + size[1]])

def resize(size, image):
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def fliplr(image):
    return np.fliplr(image)

def pipeline(cache, path_and_label):
    path, label = path_and_label
    image = read_image_cached(cache, partial(resize, (248, 248)), path)
    image = crop_random((224, 224), image)
    image = fliplr(image) if np.random.rand() < .5 else image
    image = normalize(image)
    image = channels_first(image)
    return (image, label)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj, dtype=np.float32):
    tensor = torch.Tensor(torch.from_numpy(obj.astype(dtype)))
    return as_cuda(tensor)

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
