# coding=utf-8
from argparse import ArgumentParser

import torch

from dtt_datasets.base import BaseDataset
from models.eda import EDA
from models.eda_c import EDA_C
from models.eda_cs import EDA_CS
from utils.plot import show_attention, show_attention_eda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dataset = BaseDataset()
    model = {
        'eda_cs': EDA_CS,
        'eda_c': EDA_C,
        'eda': EDA
    }[args.model](dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    x_str = "name[Fried RNNs], food[Martian], customer rating[5 out of 5], priceRange[high], near[Venice Beach]"
    x_list = dataset.to_list(x_str)
    x = torch.tensor(x_list).unsqueeze(0)

    with torch.no_grad():
        y_prob, attention, p_gens = model((x, [x.size(1)]))
        y_tensor = y_prob.argmax(1).squeeze()
        y_list = y_tensor.tolist()
        y_str = dataset.to_string(y_list)
        attention = attention.squeeze().tolist()
        if p_gens is not None:
            p_gens = [p_gens.squeeze().tolist()]

    print()
    print(f'> {x_str}')
    print(f'< {y_str}')
    if args.model != 'eda':
        show_attention(f'_{x_str}_', y_str, attention, p_gens)
    else:
        show_attention_eda(f'_{x_str}_', y_str, attention)


if __name__ == '__main__':
    # Eventually parse arguments here and call main(args)
    parser = ArgumentParser(description='Check a trained model with custom inputs')
    parser.add_argument('checkpoint', type=str, help='The state_dict of a trained model')
    parser.add_argument('--model', choices=('eda_cs', 'eda_c', 'eda'), default='eda_cs')
    main(parser.parse_args())
