# coding=utf-8
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from datasets.base import DatasetType
from datasets.e2e.e2e_plus import E2EPlus
from datasets.e2e.e2e import E2E
from datasets.hotel_e2e.hotel import Hotel
from datasets.restaurant_e2e.restaurant import Restaurant
from models.eda import EDA
from models.eda_c import EDA_C
from models.eda_cs import EDA_CS
from utils.plot import show_attention, show_attention_eda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # TODO use your dataset
    which_set = DatasetType.TEST
    dataset = E2EPlus(which_set)
    # dataset = E2E(which_set)
    # dataset = Hotel(which_set)
    # dataset = Restaurant(which_set)
    loader = DataLoader(dataset, shuffle=True, collate_fn=dataset.collate_fn)

    model = EDA_CS(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    # model = EDA_C(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    # model = EDA(dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        for x, y_ref in loader:
            x_str = dataset.to_string(x[0][0].tolist())
            y_ref_str = dataset.to_string(y_ref[0].tolist())

            y_prob, attention, p_gens = model(x)
            y_tensor = y_prob.argmax(1)
            y_list = y_tensor.tolist()
            y_str = dataset.to_string(y_list)
            attention = attention[:, 0].tolist()
            p_gens = [p_gens.squeeze().tolist()]

            print()
            print(f'> {x_str}')
            print(f'= {y_ref_str}')
            print(f'< {y_str}')
            show_attention(f'{x_str}', y_str, attention, p_gens)
            # show_attention_eda(x_str, y_str, attention)


if __name__ == '__main__':
    parser = ArgumentParser(description='Check a trained model on a dataset')
    parser.add_argument('model', type=str, help='The state_dict of a trained model')
    # parser.add_argument('input_string', type=str, help='The model input string')
    main(parser.parse_args())
