import os

from torchvision import datasets, transforms

from .med2vec_dataset import Med2VecDataset
from .med2vec_dataset import collate_fn as med2vec_collate
from base import BaseDataLoader


class Med2VecDataLoader(BaseDataLoader):
    """
    Med2Vec Dataloader
    """
    def __init__(self, data_dir, num_codes, batch_size, shuffle, validation_split, num_workers, file_name=None,
                 training=True):
        self.data_dir = data_dir
        self.num_codes = num_codes
        path_file = os.path.expanduser(data_dir)

        if file_name is not None:
            path_file = os.path.join(path_file, file_name)
        else:
            path_file = os.path.join(path_file, 'med2vec.seqs')
        self.data_dir = path_file
        self.train = training
        self.dataset = Med2VecDataset(self.data_dir, self.num_codes, self.train)
        super(Med2VecDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                                                collate_fn=med2vec_collate)

