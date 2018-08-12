import os
import glob
from functools import partial

import cv2
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

def get_train_validation_holdout_split(records):
    np.random.shuffle(records)
    n = len(records)
    train = records[:int(n * .6)]
    validation = records[int(n * .6):int(n * .75)]
    holdout = records[int(n * .75)]
    return train, validation, holdout

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

def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier('fspoofing/haar.xml')
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, width, height = faces[0]
        return image[y:y + height, x:x + width]
    else:
        return image

def pipe(funcs, arg):
    for func in funcs: arg = func(arg)
    return arg

def pipeline(cache, path_and_label):
    path, label = path_and_label
    # TODO AS: Face detection eats too much time
    # preprocess = partial(pipe, [
    #    crop_face,
    #    partial(resize, (224, 224))
    # ])
    # TODO AS: Doesn't seem to converge with augs yet
    # image = crop_random((224, 224), image)
    # image = fliplr(image) if np.random.rand() < .5 else image
    preprocess = partial(resize, (224, 224))
    image = read_image_cached(cache, preprocess, path)
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

def print_confusion_matrix(outputs, gt):
    labels = np.argmax(outputs, axis=1)
    true_negatives = sum(labels[gt == 0] == 0)
    false_positives = sum(labels[gt == 0] == 1)

    true_positives = sum(labels[gt == 1] == 1)
    false_negatives = sum(labels[gt == 1] == 0)

    tqdm.write(tabulate([
        ['Pred Real', true_negatives, false_negatives],
        ['Pred Spoof', false_positives, true_positives]
    ], headers=['True Real', 'True Spoof'], tablefmt='grid'))

if __name__ == '__main__':
    import pdb; pdb.set_trace()
