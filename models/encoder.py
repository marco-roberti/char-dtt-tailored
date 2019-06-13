# coding=utf-8
from torch.nn import Module, Embedding, GRU
from torch.nn.functional import pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, enc_input, lengths):
        # [B x T x D]
        enc_input = self.embedding(enc_input)
        # [sum of len seqs x D]
        enc_input = pack_padded_sequence(enc_input, lengths, batch_first=True)
        # output packed [sum of len seqs x H*num_dir], hidden [num_dir*num_lay x B x H]
        outputs, hidden = self.gru(enc_input)
        # outputs [max len x B x num_dir*H]
        outputs, output_lengths = pad_packed_sequence(outputs)
        # outputs [max len x B x num_dir*H], hidden [num_dir*num_lay x B x H]
        return outputs, output_lengths, hidden


def forward_encoder(enc_input, lengths, embedding, gru):
    # [B x T x D]
    enc_input = embedding(enc_input)
    enc_input = pad(enc_input, (2 * gru.hidden_size, 0))
    # [sum of len seqs x D]
    enc_input = pack_padded_sequence(enc_input, lengths, batch_first=True)
    # output packed [sum of len seqs x H*num_dir], hidden [num_dir*num_lay x B x H]
    outputs, hidden = gru(enc_input)
    # outputs [max len x B x num_dir*H]
    outputs, output_lengths = pad_packed_sequence(outputs)
    # outputs [max len x B x num_dir*H], hidden [num_dir*num_lay x B x H]
    return outputs, output_lengths, hidden
