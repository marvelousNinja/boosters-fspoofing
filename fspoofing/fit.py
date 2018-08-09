import torch
import torchvision

import numpy as np
import torch
from fire import Fire

from fspoofing.generators import get_train_generator
from fspoofing.generators import get_validation_generator
from fspoofing.training import fit_model
from fspoofing.utils import as_cuda

def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels.long())

def fit():
    np.random.seed(1991)
    model = as_cuda(torchvision.models.squeezenet1_1(num_classes=2))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), .001)

    fit_model(
        model=model,
        train_generator=get_train_generator(),
        validation_generator=get_validation_generator(),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=100,
        num_batches=515, # 8231 in train set
        validation_batches=63, # 1007 in val set
        after_validation=None
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
