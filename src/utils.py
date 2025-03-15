import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from torch.utils.data.sampler import Sampler
from torch.utils.data import SubsetRandomSampler
from statistics import mean
from monai.metrics import DiceMetric
import numpy as np
    
class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

class SamplingStrategy:
    """ 
    Sampling Strategy wrapper class
    """
    def __init__(self, dset, train_idx, model, device, args):
        self.dset = dset
        self.train_idx = np.array(train_idx)
        self.model = model
        self.device = device
        self.args = args
        self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
    
    def query(self, n):
        pass
    
    def custom_collate(self, batch):
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets
