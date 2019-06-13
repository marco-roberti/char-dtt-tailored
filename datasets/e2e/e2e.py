# coding=utf-8
import csv
import os.path
from typing import List, Tuple

from datasets.base import BaseDataset, DatasetType


class E2E(BaseDataset):
    _csv_for = {
        DatasetType.TRAIN: os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trainset.csv'),
        DatasetType.DEV:   os.path.join(os.path.dirname(os.path.realpath(__file__)), 'devset.csv'),
        DatasetType.TEST:  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testset.csv')
    }

    def __init__(self, which_set: DatasetType):
        super(E2E, self).__init__()
        self.which_set = which_set

        x_string, y_string = self._read()
        self._string_to_char(x_string, y_string)

    def __repr__(self) -> str:
        return f'E2E Challenge Dataset, {self.which_set.name.lower()} set\n' \
               f'\tNumber of instances: {self.__len__()}\n'

    def _read(self) -> Tuple[List[str], List[str]]:
        file = self._csv_for[self.which_set]
        mr = []
        ref = []
        with open(file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader)
            for row in reader:
                mr.append(row[0])
                ref.append(row[1])
        return mr, ref
