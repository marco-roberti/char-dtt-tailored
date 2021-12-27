# coding=utf-8
import enum
import string
from enum import Enum
from typing import List

import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetType(Enum):
    TRAIN = enum.auto()
    DEV = enum.auto()  # tuning
    TEST = enum.auto()  # NO tuning


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.vocabulary = string.printable
        self.pad_token = len(self.vocabulary)
        self.sos_token = len(self.vocabulary) + 1
        self.eos_token = len(self.vocabulary) + 2

        self.print_vocabulary = self.vocabulary + '#__'

    def __getitem__(self, index):
        return self._x[index], self._y[index]

    def __len__(self):
        return len(self._x)

    def __repr__(self) -> str:
        return f'Dataset {self.__class__.__name__}\n' \
               f'\tNumber of instances: {self.__len__()}\n'

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda example: len(example[0]), reverse=True)

        lengths_x = [len(example[0]) for example in batch]
        max_len_x = max(lengths_x)

        lengths_y = [len(example[1]) for example in batch]
        max_len_y = max(lengths_y)

        xs = torch.tensor([self._pad_sequence(x, max_len_x) for x, _ in batch]).to(device)
        ys = torch.tensor([self._pad_sequence(y, max_len_y) for _, y in batch]).to(device)
        lengths_x = torch.tensor(lengths_x)

        return (xs, lengths_x), ys

    def sort(self):
        """Sorts the examples by mr"""
        data_zip = list(zip(self._x, self._y))
        data_zip.sort(key=lambda example: example[0])
        unzip = list(zip(*data_zip))
        self._x = list(unzip[0])
        self._y = list(unzip[1])
        return self

    def to_list(self, sequence: str) -> List[int]:
        return [self.sos_token] + [self.vocabulary.index(c) for c in sequence] + [self.eos_token]

    def to_string(self, sequence):
        return ''.join([self.print_vocabulary[i] for i in sequence if i != self.pad_token])

    def vocabulary_size(self):
        return len(self.print_vocabulary)

    def _pad_sequence(self, sequence, length):
        pad_length = length - len(sequence)
        return sequence + [self.pad_token for _ in range(pad_length)]

    def _string_to_char(self, x_string, y_string):
        self._x = [[self.sos_token] + [self.vocabulary.index(c) for c in s] + [self.eos_token] for s in x_string]
        self._y = [[self.sos_token] + [self.vocabulary.index(c) for c in s] + [self.eos_token] for s in y_string]
