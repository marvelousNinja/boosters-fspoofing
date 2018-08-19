from functools import partial

import torch
import torchvision
import numpy as np
from fire import Fire
from tqdm import tqdm

from fspoofing.generators import get_train_generator
from fspoofing.generators import get_validation_generator
from fspoofing.model_checkpoint import load_checkpoint
from fspoofing.model_checkpoint import ModelCheckpoint
from fspoofing.training import fit_model
from fspoofing.utils import as_cuda
from fspoofing.utils import confusion_matrix

def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels.long())

def after_validation(model_checkpoint, val_loss, outputs, gt):
    tqdm.write(confusion_matrix(outputs, gt, [0, 1]))
    model_checkpoint.step(val_loss)

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001, checkpoint_path=None):
    np.random.seed(1991)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, momentum=.95, nesterov=True)
    model_checkpoint = ModelCheckpoint(model, 'resnet18', tqdm.write)

    fit_model(
        model=model,
        train_generator=get_train_generator(batch_size, limit),
        validation_generator=get_validation_generator(batch_size, limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        after_validation=partial(after_validation, model_checkpoint)
    )

def prof():
    import profile
    import pstats
    profile.run('fit()', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
