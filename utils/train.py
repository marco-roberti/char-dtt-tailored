# coding=utf-8
import os
import time
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(loader: DataLoader, model: Module, optimizer: Optimizer, criterion, num_classes: int, n_epochs: int,
          first_epoch=0, save_epochs=True, clip_norm=None):
    print(f'Training exactly {count_parameters(model)} parameters.')

    batches_per_epoch = len(loader)
    total_batches = batches_per_epoch * (n_epochs - first_epoch)
    epoch_losses = []

    model.train()
    start = time.time()
    save_folder = f'trained_nets/{start:.0f}'
    for epoch in range(first_epoch + 1, n_epochs + 1):
        epoch_loss = 0
        for i, (x, y) in enumerate(tqdm(loader, leave=False)):
            epoch_loss += optimization_step(x, y, model, optimizer, criterion, num_classes, clip_norm)[0]

        # Computing time stats
        now = time.time()
        elapsed = now - start
        h_el, m_el, s_el = sec_to_hhmmss(elapsed)
        batches_completed = batches_per_epoch * (epoch - first_epoch)
        batches_remaining = total_batches - batches_completed
        prev = elapsed / batches_completed * batches_remaining
        h_prev, m_prev, s_prev = sec_to_hhmmss(prev)

        if save_epochs:
            save(model, name=f'epoch{epoch:02d}_loss{epoch_loss:6.4f}', folder=save_folder)

        epoch_losses.append(epoch_loss)

        print(f'[epoch {epoch:2d}/{n_epochs}, '
              f'{h_el:2d}h {m_el:02d}m {s_el:02d}s elapsed, '
              f'{h_prev:2d}h {m_prev:02d}m {s_prev:02d}s remaining] '
              f'Train loss: {epoch_loss:9.6f}')

    if not save_epochs:
        save(model, name=f'Final_{n_epochs}epochs_loss{epoch_losses[-1]:6.4f}', folder=save_folder)
    model.eval()
    return epoch_losses


def train_switching_grus(loader: DataLoader, model: Module, optimizer: Optimizer, criterion, num_classes: int,
                         n_epochs: int, first_epoch=0, save_epochs=True, clip_norm=None):
    print(f'Training exactly {count_parameters(model)} parameters.')

    batches_per_epoch = len(loader)
    total_batches = batches_per_epoch * (n_epochs - first_epoch)
    epoch_losses = []

    model.train()
    start = time.time()
    save_folder = f'trained_nets/{start:.0f}'
    for epoch in range(first_epoch + 1, n_epochs + 1):
        epoch_loss = 0
        for i, (x, y) in enumerate(tqdm(loader, leave=False)):
            # Forward
            step_loss, y_ = optimization_step(x, y, model, optimizer, criterion, num_classes, clip_norm, reverse=False)
            epoch_loss += step_loss

            # Backward
            x_bw = x[0]
            y_bw_data = y_.argmax(2)
            y_bw_lengths = sequence_lengths(y_bw_data, model.eos_token)
            y_bw_data = y_bw_data[:, :max(y_bw_lengths)]  # avoid padding whole columns
            y_bw, x_bw = aligned_sort(y_bw_data, y_bw_lengths, x_bw)

            optimization_step(y_bw, x_bw, model, optimizer, criterion, num_classes, clip_norm, reverse=True)

        # Computing time stats
        now = time.time()
        elapsed = now - start
        h_el, m_el, s_el = sec_to_hhmmss(elapsed)
        batches_completed = batches_per_epoch * (epoch - first_epoch)
        batches_remaining = total_batches - batches_completed
        prev = elapsed / batches_completed * batches_remaining
        h_prev, m_prev, s_prev = sec_to_hhmmss(prev)

        if save_epochs:
            save(model, name=f'epoch{epoch:02d}_loss{epoch_loss:6.4f}', folder=save_folder)

        epoch_losses.append(epoch_loss)

        print(f'[epoch {epoch:2d}/{n_epochs}, '
              f'{h_el:2d}h {m_el:02d}m {s_el:02d}s elapsed, '
              f'{h_prev:2d}h {m_prev:02d}m {s_prev:02d}s remaining] '
              f'Train loss: {epoch_loss:9.6f}')

    if not save_epochs:
        save(model, name=f'Final_{n_epochs}epochs_loss{epoch_losses[-1]:6.4f}', folder=save_folder)
    model.eval()
    return epoch_losses


def aligned_sort(tensor_1, lengths, tensor_2):
    data_zip = list(zip(tensor_1, lengths, tensor_2))
    data_zip.sort(key=lambda example: example[1], reverse=True)
    unzip = list(zip(*data_zip))
    return (torch.stack(unzip[0]), list(unzip[1])), torch.stack(unzip[2])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model: Module, name=str(round(time.time())), folder='./Saved_models') -> None:
    # Create folder if it doesn't exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, name), 'wb') as file:
        torch.save(model.state_dict(), file)


def sec_to_hhmmss(seconds: float) -> Tuple[int, int, int]:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def sequence_lengths(sequences, eos_token):
    max_len = sequences.size(1)
    return [_sequence_length(s, eos_token, max_len) for s in sequences]


def optimization_step(x, y, model, optimizer, criterion, num_classes, clip_norm, reverse=None):
    optimizer.zero_grad()
    y_ = model(x, y, reverse=reverse) if reverse else model(x, y)
    loss = criterion(y_.view(-1, num_classes), y.view(-1))
    loss.backward()
    if clip_norm:
        clip_grad_norm_(model.parameters(), clip_norm)  # Gradient clipping
    optimizer.step(None)
    return loss.item() / y.size(1), y_  # Regularizes loss according to sequence length


def _sequence_length(s, eos_token, max_len):
    eos_position: Tensor = (s == eos_token)
    # reshape(-1) allows to deal with a non-scalar tensor
    return eos_position.nonzero().reshape(-1)[0] if eos_position.sum() > 0 else max_len
