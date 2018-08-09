import gc

import numpy as np
from tqdm import tqdm

from fspoofing.utils import from_numpy
from fspoofing.utils import to_numpy

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs,
        num_batches,
        validation_batches,
        after_validation=None
    ):

    for _ in tqdm(range(num_epochs)):
        train_loss = 0
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            inputs, gt = next(train_generator)
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            loss = loss_fn(model(inputs), gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= num_batches

        val_loss = 0
        all_preds = []
        all_gt = []
        for _ in tqdm(range(validation_batches)):
            inputs, gt = next(validation_generator)
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            outputs = model.eval()(inputs)
            val_loss += loss_fn(outputs, gt).item()

            all_preds.append(to_numpy(outputs))
            all_gt.append(to_numpy(gt))
        val_loss /= validation_batches

        tqdm.write(f'train loss {train_loss:.5f} - val loss {val_loss:.5f}')
        if after_validation: after_validation(inputs, np.concatenate(all_preds), np.concatenate(all_gt))
