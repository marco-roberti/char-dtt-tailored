# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import pad


class AttentionBahdanauWithLastChar(nn.Module):
    """
    attention with linear mechanism, Bahdanau, plus last char info
    """

    def __init__(self, hidden_size, attention_size, embedding_size, pad_token):
        super(AttentionBahdanauWithLastChar, self).__init__()
        self.pad_token = pad_token

        self.linear_enc = nn.Linear(2 * hidden_size, attention_size, bias=False)
        self.linear_dec = nn.Linear(hidden_size, attention_size)
        self.linear_last_char = nn.Linear(embedding_size, attention_size)

        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_state, encoder_annotations, last_char_index):
        att_decoder_state = self.linear_dec(decoder_state)  # [Bx1xAtt]
        att_encoder_annotations = self.linear_enc(encoder_annotations)
        att_last_char = self.linear_last_char(last_char_index)
        att_encoder_annotations.transpose_(0, 1)  # [BxTxAtt]

        # attention scores
        att_decoder_state = att_decoder_state.expand_as(att_encoder_annotations)
        att_last_char = att_last_char.expand_as(att_encoder_annotations)

        # get the mask
        mask_tensor = att_encoder_annotations.data != self.pad_token
        mask = Variable(mask_tensor.type_as(att_encoder_annotations.data), requires_grad=False)

        attention_sum = (att_decoder_state + att_encoder_annotations + att_last_char) * mask  # [BxTxAtt]
        attention_sum = torch.tanh(attention_sum)

        scores = self.v(attention_sum)

        # prob distribution over scores
        focus_prob = masked_stable_softmax(scores, self.pad_token)
        return focus_prob  # [BxTx1]


def masked_stable_softmax(x, pad_token):
    """
    masked softmax for batches with shifted un-normalize probabilities for stability.
    """

    # get ones where x values are non-zero
    mask_tensor = x.data != pad_token
    mask = Variable(mask_tensor.type_as(x.data), requires_grad=False)

    # find max values and shift to deal with numerical instability
    maxes, _ = torch.max(x, dim=1, keepdim=True)
    shift_x = x - maxes

    # mask element's exp
    exponents = torch.exp(shift_x) * mask

    # get sum of exponentials over every sample in the batch
    exponentials_sum = torch.sum(exponents, dim=1).expand(exponents.squeeze(2).size()).unsqueeze(2) + 0.0000001

    softmax = exponents / exponentials_sum

    return softmax


def masked_softmax(x, lengths, softmax):
    """
    computes the PyTorch softmax for every sample in the batch.
    probably is more stable than the masked_stable_softmax but slows down performance too much
    """
    probabilities = []
    max_len = lengths[0]

    for b in range(x.size()[0]):
        # get the sample length
        length = lengths[b]

        # get the cropped sample
        sample = x[b][:length, :]

        soft = softmax(sample)

        # pad to original max_string_length
        pad_param = (0, 0, 0, max_len - length)
        soft = pad(soft, pad_param, "constant", 0)
        probabilities.append(soft)

    probabilities = torch.stack(probabilities)

    return probabilities
