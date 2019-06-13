# coding=utf-8
from argparse import ArgumentParser

import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.base import DatasetType
from datasets.e2e.e2e import E2E
from datasets.e2e.e2e_plus import E2EPlus
from datasets.hotel_e2e.hotel import Hotel
from datasets.restaurant_e2e.restaurant import Restaurant
from models.eda import EDA
from models.eda_c import EDA_C
from models.eda_cs import EDA_CS
from utils.train import train, train_switching_grus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
n_epochs = 100
learning_rate = 0.0001
clip_norm = 1


def main(args):
    # TODO use your dataset
    dataset = E2EPlus(DatasetType.TRAIN)
    # dataset = E2E(DatasetType.TRAIN)
    # dataset = Hotel(DatasetType.TRAIN)
    # dataset = Restaurant(DatasetType.TRAIN)

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True)

    # TODO use your model
    model = EDA_CS(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token)
    # model = EDA_C(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token)
    # model = EDA(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token)

    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = NLLLoss()

    # TODO choose proper training process
    losses = train_switching_grus(data_loader, model, optimizer, criterion, dataset.vocabulary_size(), n_epochs,
                                  args.epoch, clip_norm=clip_norm)
    # losses = train(data_loader, model, optimizer, criterion, dataset.vocabulary_size(), n_epochs, args.epoch,
    #                clip_norm=clip_norm)
    print(losses)


if __name__ == '__main__':
    parser = ArgumentParser(description='Utility script to (load and) train a model.')
    parser.add_argument('-m', '--model', help='The state_dict of a trained model (use this for transfer learning)',
                        type=str)
    parser.add_argument('-e', '--epoch', help='The epoch index to start with', default=0, type=int)
    main(parser.parse_args())
