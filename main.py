# coding=utf-8
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dtt_datasets.base import DatasetType
from dtt_datasets.e2e.e2e import E2E
from dtt_datasets.e2e.e2e_plus import E2EPlus
from dtt_datasets.hotel_e2e.hotel import Hotel
from dtt_datasets.restaurant_e2e.restaurant import Restaurant
from models.eda import EDA
from models.eda_c import EDA_C
from models.eda_cs import EDA_CS
from utils.train import train, train_switching_grus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dataset = {
        'e2e+': E2EPlus,
        'e2e': E2E,
        'hotel': Hotel,
        'restaurant': Restaurant
    }[args.dataset](DatasetType.TRAIN)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)

    model = {
        'eda_cs': EDA_CS,
        'eda_c': EDA_C,
        'eda': EDA
    }[args.model](dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token)

    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = NLLLoss()

    if args.model == 'eda_cs':
        losses = train_switching_grus(data_loader, model, optimizer, criterion,
                                      dataset.vocabulary_size(), args.n_epochs, args.epoch, clip_norm=args.clip_norm)
    else:
        losses = train(data_loader, model, optimizer, criterion,
                       dataset.vocabulary_size(), args.n_epochs, args.epoch, clip_norm=args.clip_norm)
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Utility script to (load and) train a model.')
    parser.add_argument('--dataset', choices=('e2e+', 'e2e', 'hotel', 'restaurant'), default='e2e+')
    parser.add_argument('--model', choices=('eda_cs', 'eda_c', 'eda'), default='eda_cs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--clip_norm', type=int, default=1)
    parser.add_argument('-c', '--checkpoint', help='The state_dict of a trained model (use this for transfer learning)')
    parser.add_argument('-e', '--epoch', help='The epoch index to start with', default=0, type=int)
    main(parser.parse_args())
