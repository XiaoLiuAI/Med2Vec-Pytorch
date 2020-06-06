import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    在torch本身的 dataloader 基础上包装了 validation split 这个功能,
    通过额外的 api 返回 validation set 的 dataloader
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """
        通过不同的 sampler 来实现数据集分割
        :param split:
        :return:
        """
        if split == 0.0:
            return None, None
        idx_full = np.arange(self.n_samples)
        # shuffle indexes only if shuffle is true
        # added for med2vec dataset where order matters
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        if (self.shuffle):
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler = SequentialSampler(train_idx)
            valid_sampler = SequentialSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

