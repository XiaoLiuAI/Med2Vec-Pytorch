#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import os
import pickle

import torch
import torch.utils.data as data


class Med2VecDataset(data.Dataset):

    def __init__(self, root, num_codes, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.num_codes = num_codes
        if download:
            raise ValueError('cannot download')

        self.train_data = pickle.load(open(root, 'rb'))
        self.test_data = []

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        x, ivec, jvec, d = self.preprocess(self.train_data[index])
        return x, ivec, jvec, d

    def preprocess(self, seq):
        """ create one hot vector of idx in seq, with length self.num_codes

            Args:
                seq: list of ideces where code should be 1

            Returns:
                x: one hot vector
                ivec: vector for learning code representation
                jvec: vector for learning code representation
        """
        x = torch.zeros((self.num_codes, ), dtype=torch.long)

        ivec = []
        jvec = []
        d = []
        if seq == [-1]:  # masked, separator between patient
            return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

        x[seq] = 1  # one-hot
        for i in seq:
            for j in seq:
                if i == j:
                    continue
                ivec.append(i)
                jvec.append(j)  # code to code coordination, code pairs in one visit
        return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d  # d 没有任何操作


def collate_fn(data):
    """ Creates mini-batch from x, ivec, jvec tensors

    We should build custom collate_fn, as the ivec, and jvec have varying lengths. These should be appended
    in row form

    Args:
        data: list of tuples contianing (x, ivec, jvec)

    Returns:
        x: one hot encoded vectors stacked vertically
        ivec: long vector
        jvec: long vector
        mask:
        d:
    """

    x, ivec, jvec, d = zip(*data)  # x 是one-hot, 1 x num_vocab -stack-> n x num_vocab
    x = torch.stack(x, dim=0)  # list of tensor to tensor with additional dimension
    mask = torch.sum(x, dim=1) > 0
    mask = mask[:, None]  # additional dimension
    ivec = torch.cat(ivec, dim=0)  # list of list 接起来变成一个list
    jvec = torch.cat(jvec, dim=0)
    d = torch.stack(d, dim=0)

    return x, ivec, jvec, mask, d


def get_loader(root, num_codes, train=True, transform=None, target_transform=None, download=False, batch_size=1000,
               shuffle=False, num_workers=0):
    """ returns torch.utils.data.DataLoader for Med2Vec dataset """
    med2vec = Med2VecDataset(root, num_codes, train, transform, target_transform, download)
    data_loader = torch.utils.data.DataLoader(dataset=med2vec, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
