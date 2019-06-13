# coding=utf-8
from typing import List

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def show_attention(x_str: str, y_str: str, attention: List[int], p_gens: List[int], pgen_cmap='bwr'):
    gs = GridSpec(8, 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[:-1, 0])
    ax1.matshow(attention, cmap='Greys_r')
    ax1.set_xticks(range(len(x_str)))
    ax1.set_yticks(range(len(y_str)))
    ax1.set_xticklabels(x_str)
    ax1.set_yticklabels(y_str)

    ax2 = fig.add_subplot(gs[-1, 0])
    ax2.matshow(p_gens, cmap=pgen_cmap, vmin=0, vmax=1)
    ax2.set_xticks(range(len(y_str)))
    ax2.set_yticks([0])
    ax2.set_xticklabels(y_str)
    ax2.set_yticklabels('P')

    plt.show()


def show_attention_eda(x_str: str, y_str: str, attention: List[int]):
    ax = plt.figure().add_subplot(111)
    ax.matshow(attention, cmap='Greys_r')
    ax.set_xticks(range(len(x_str)))
    ax.set_yticks(range(len(y_str)))
    ax.set_xticklabels(x_str)
    ax.set_yticklabels(y_str)

    plt.show()
