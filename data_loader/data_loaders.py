from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.multi_pose import MultiPoseDataset
from data_loader.centerface_hp import FACEHP
import pickle

class CenterFaceDataset(FACEHP, MultiPoseDataset):
    pass

class DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, key, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        opt = None
        with open('data_loader/opt.pkl', 'rb') as f:
            opt = pickle.load(f)
        self.dataset = CenterFaceDataset(opt, key)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
