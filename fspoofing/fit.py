import torch
import torchvision
import numpy as np
from fire import Fire

from fspoofing.generators import get_train_generator
from fspoofing.generators import get_validation_generator
from fspoofing.training import fit_model
from fspoofing.utils import as_cuda
from fspoofing.utils import print_confusion_matrix

def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels.long())

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001):
    np.random.seed(1991)
    model = as_cuda(torchvision.models.squeezenet1_1(num_classes=2))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)

    fit_model(
        model=model,
        train_generator=get_train_generator(batch_size, limit),
        validation_generator=get_validation_generator(batch_size, limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        after_validation=print_confusion_matrix
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
