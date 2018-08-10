import math
from functools import partial

import numpy as np

from fspoofing.utils import get_image_and_label_pairs
from fspoofing.utils import pipeline

class DataGenerator:
    def __init__(self, records, batch_size, transform):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform

    def __iter__(self):
        np.random.shuffle(self.records)
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
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_val')[:limit]
    transform = partial(pipeline, {})
    return DataGenerator(path_and_label_pairs, batch_size, transform)

def get_train_generator(batch_size, limit=None):
    path_and_label_pairs = get_image_and_label_pairs('data/IDRND_FASDB_train')[:limit]
    transform = partial(pipeline, {})
    return DataGenerator(path_and_label_pairs, batch_size, transform)
