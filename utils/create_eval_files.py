# coding=utf-8
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dtt_datasets.base import DatasetType
from dtt_datasets.e2e.e2e import E2E
from dtt_datasets.e2e.e2e_plus import E2EPlus
from dtt_datasets.hotel_e2e.hotel import Hotel
from dtt_datasets.restaurant_e2e.restaurant import Restaurant
from models.eda import EDA
from models.eda_c import EDA_C
from models.eda_cs import EDA_CS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    which_set = {'dev': DatasetType.DEV, 'test': DatasetType.TEST}[args.subset]
    dataset = {
        'e2e+': E2EPlus,
        'e2e': E2E,
        'hotel': Hotel,
        'restaurant': Restaurant
    }[args.dataset](which_set)
    dataset.sort()
    loader = DataLoader(dataset, collate_fn=dataset.collate_fn)

    model = {
        'eda_cs': EDA_CS,
        'eda_c': EDA_C,
        'eda': EDA
    }[args.model](dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        with open(f'{args.checkpoint}.{which_set.name.lower()}.output', 'w') as outputs, \
                open(f'{args.checkpoint}.{which_set.name.lower()}.references', 'w') as references:
            last_x = None
            for (x, lengths), y in tqdm(loader):
                list_x = x.squeeze().tolist()
                if list_x != last_x:
                    y_prob = model((x, lengths))[0]
                    y_tensor = y_prob.argmax(1)
                    y_list = y_tensor.tolist()
                    y_ = dataset.to_string(y_list).strip('_')
                    outputs.write(y_ + '\n')
                    if references.tell() > 0:
                        references.write('\n')

                y = dataset.to_string(y.squeeze().tolist()).strip('_')
                references.write(y + '\n')
                last_x = list_x


if __name__ == '__main__':
    parser = ArgumentParser(description='Create evaluation files for a dataset using a trained model')
    parser.add_argument('checkpoint', type=str, help='The state_dict of a trained model')
    parser.add_argument('--subset', choices=('dev', 'test'), default='test', help='Which set to use, dev or test.')
    parser.add_argument('--dataset', choices=('e2e+', 'e2e', 'hotel', 'restaurant'), default='e2e+')
    parser.add_argument('--model', choices=('eda_cs', 'eda_c', 'eda'), default='eda_cs')
    main(parser.parse_args())
