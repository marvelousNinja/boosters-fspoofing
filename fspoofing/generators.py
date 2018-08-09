import numpy as np

from fspoofing.utils import get_image_and_label_pairs
from fspoofing.utils import pipeline

def get_validation_generator():
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_val')
    while True:
        np.random.shuffle(path_and_label_pairs)
        images = []
        labels = []

        for image, label in map(pipeline, path_and_label_pairs):
            images.append(image)
            labels.append(label)

            if len(images) >= 16:
                yield np.stack(images), np.array(labels)
                images = []
                labels = []


def get_train_generator():
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_train')
    while True:
        np.random.shuffle(path_and_label_pairs)
        images = []
        labels = []

        for image, label in map(pipeline, path_and_label_pairs):
            images.append(image)
            labels.append(label)

            if len(images) >= 16:
                yield np.stack(images), np.array(labels)
                images = []
                labels = []
