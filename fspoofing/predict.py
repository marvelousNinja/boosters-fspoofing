import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from fspoofing.model_checkpoint import load_checkpoint
from fspoofing.utils import get_images_in
from fspoofing.utils import as_cuda
from fspoofing.utils import from_numpy
from fspoofing.utils import to_numpy
from fspoofing.generators import get_test_generator

def predict(checkpoint_path, batch_size=8, limit=None):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    model.eval()

    all_outputs = []
    test_generator = get_test_generator(batch_size, limit)
    for inputs, gt in tqdm(test_generator, total=len(test_generator)):
        inputs, gt = from_numpy(inputs), from_numpy(gt)
        outputs = model(inputs)
        all_outputs.append(to_numpy(torch.nn.functional.softmax(outputs, dim=1)[:, 1]))
    all_outputs = np.concatenate(all_outputs)
    ids = list(map(lambda path: path.split('/')[-1], get_images_in('data/test')))[:limit]
    df = pd.DataFrame({ 'id': ids, 'prob': all_outputs })
    df.to_csv('./data/submissions/__latest.csv', index=False)

if __name__ == '__main__':
    Fire(predict)
