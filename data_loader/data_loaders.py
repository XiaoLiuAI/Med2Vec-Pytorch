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
        """
        把dataset参数/属性固定为 Med2VecDataset，不再作为一个参数传入，同时因为Med2VecDataset本身是一次性读入所有数据到
        内存中，在num_worker不为0的时候(多进程读取数据),会直接炸内存. 但是，Med2VecDataset已经读取所有数据到内存中，此时也
        不需要多进程数据读取
        :param data_dir:
        :param num_codes:
        :param batch_size:
        :param shuffle:
        :param validation_split:
        :param num_workers:
        :param file_name:
        :param training:
        """
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

