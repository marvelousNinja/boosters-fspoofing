import math
from functools import partial

import numpy as np

from fspoofing.utils import get_image_and_label_pairs
from fspoofing.utils import get_images_in
from fspoofing.utils import get_train_validation_holdout_split
from fspoofing.utils import pipeline

class DataGenerator:
    def __init__(self, records, batch_size, transform, shuffle=True):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.records)
        batch = []

        for output in map(self.transform, self.records):
            batch.append(output)

            if len(batch) >= self.batch_size:
                split_outputs = list(zip(*batch))
                yield map(np.stack, split_outputs)
                batch = []

        if len(batch) > 0:
            split_outputs = list(zip(*batch))
            yield map(np.stack, split_outputs)

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

def get_validation_generator(batch_size, limit=None):
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_train')
    path_and_label_pairs += get_image_and_label_pairs('data/IDRND_FASDB_val')
    _, path_and_label_pairs, _ = get_train_validation_holdout_split(path_and_label_pairs)
    transform = partial(pipeline, {})
    return DataGenerator(path_and_label_pairs[:limit], batch_size, transform)

def get_train_generator(batch_size, limit=None):
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_train')
    path_and_label_pairs += get_image_and_label_pairs('data/IDRND_FASDB_val')
    path_and_label_pairs, _, _ = get_train_validation_holdout_split(path_and_label_pairs)
    transform = partial(pipeline, {})
    return DataGenerator(path_and_label_pairs[:limit], batch_size, transform)

def get_test_generator(batch_size, limit=None):
    paths = get_images_in('data/test')
    path_and_label_pairs = list(map(lambda path: (path, 0), paths))
    transform = partial(pipeline, {})
    return DataGenerator(path_and_label_pairs[:limit], batch_size, transform, shuffle=False)
