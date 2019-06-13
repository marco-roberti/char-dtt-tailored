# coding=utf-8
import torch
from torch import nn

from models.attention import AttentionBahdanauWithLastChar
from models.decoder import _shift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmptyDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, attention_size, pad_token):
        super(EmptyDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention = AttentionBahdanauWithLastChar(hidden_size, attention_size, embedding_size, pad_token)
        self.out = nn.Linear(4 * hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, last_char_index, decoder_output, decoder_hidden, encoder_annotations, embedding, gru):
        last_char_index = last_char_index.long().squeeze(1)
        last_char_index = embedding(last_char_index)

        # getting attention weights over encoder_annotations
        focus_prob = self.attention(decoder_output, encoder_annotations, last_char_index)  # [BxTx1]
        context = focus_prob.transpose(1, 2) @ encoder_annotations.transpose(0, 1)  # [Bx1xT] @ [BxTx2*H] = [Bx1x2*H]

        # concatenate context with last_char_index
        dec_input = torch.cat((context, last_char_index), 2)  # [Bx1x(2*H+Emb)]

        # decoder_output = [Bx1xH], decoder_hidden = [num_lay x 1 x (2*H+Emb)]
        decoder_output, decoder_hidden = gru(dec_input, decoder_hidden)

        # concatenate output with context before prediction
        out_context = torch.cat((decoder_output, context), 2)  # [Bx1x(3*H)]
        un_normalize_log_prob = self.out(out_context)
        prob = self.log_softmax(un_normalize_log_prob)  # [B x 1 x alpLen]
        return focus_prob, decoder_output, decoder_hidden, prob

    def init_first_context(self, batch_size):
        context = torch.zeros(batch_size, 1, 2 * self.hidden_size)
        return context.to(device)

    def init_first_hidden(self, batch_size):
        hid = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()
        return hid.to(device)

    def init_first_output(self, batch_size):
        out = torch.zeros(batch_size, 1, 2 * self.hidden_size).float()
        return out.to(device)


class EmptyDecoderAndPointer(EmptyDecoder):

    def __init__(self, output_size, embedding_size, hidden_size, attention_size, pad_token, shift_focus=False):
        super().__init__(output_size, embedding_size, hidden_size, attention_size, pad_token)
        self.softmax = nn.Softmax(dim=2)

        self.out_p = nn.Linear(2 * hidden_size, 1)
        self.last_char_p = nn.Linear(embedding_size, 1)
        self.context_p = nn.Linear(hidden_size * 2, 1)
        # self.out_context_p = nn.Linear(hidden_size * 3, 1)
        self.gen_copy_switcher = nn.Linear(3, 1)  # it decides to change p_gen or not
        self.p_gen_l = nn.Linear(1, 1)
        self.shift_focus = shift_focus

    def forward(self, last_char_index, decoder_output, decoder_hidden, encoder_annotations, pairs_to_string, p_gen,
                embedding, gru):
        focus_prob, decoder_output, decoder_hidden, prob = super().forward(
            last_char_index, decoder_output, decoder_hidden, encoder_annotations, embedding, gru)
        # TODO occhio ai calcoli ripetuti!

        last_char_index = last_char_index.long().squeeze(1)
        last_char_index = embedding(last_char_index)
        lcp = self.last_char_p(last_char_index)
        op = self.out_p(decoder_output)
        context = focus_prob.transpose(1, 2) @ encoder_annotations.transpose(0, 1)  # [Bx1xT] @ [BxTx2*H] = [Bx1x2*H]
        cp = self.context_p(context)
        # ocp = self.out_context_p(out_context)
        p_g = self.p_gen_l(p_gen)

        # p_g_input_concat = torch.cat((lcp, , p_g), 2)  # [Bx1x3]
        # switcher = self.gen_copy_switcher(p_g_input_concat)

        # p_gen = torch.sigmoid(switcher)
        p_gen = torch.sigmoid(lcp + op + cp + p_g)
        p_gen = p_gen.squeeze(2)

        # get the P_vocab running a softmax over un-normalized_log_prob
        out_context = torch.cat((decoder_output, context), 2)  # [Bx1x(3*H)]
        un_normalize_log_prob = self.out(out_context)
        gen_prob = self.softmax(un_normalize_log_prob)  # [B x 1 x alpLen]

        # multiply generation probability for Pgen
        gen_prob = gen_prob.squeeze(1)
        scaled_gen_prob = p_gen * gen_prob

        focus_prob = focus_prob.squeeze(2)

        if self.shift_focus:
            # shift focus_prob
            focus_prob = _shift(focus_prob)

        # multiply copy probability for 1 - Pgen
        focus_prob_p = (1 - p_gen) * focus_prob

        # add probabilities together
        # scatter adds every char probability in focus_prob with gen_prob
        final_prob = scaled_gen_prob.scatter_add(1, pairs_to_string, focus_prob_p)
        final_prob = final_prob.unsqueeze(1)
        final_prob = torch.log(final_prob)

        # multiply focus_prob with context
        # shifted_context = focus_shift @ encoder_annotations.transpose(0, 1)  # [Bx1xT] @ [BxTx2*H] = [Bx1x2*H]

        p_gen = p_gen.unsqueeze(1)

        return focus_prob, decoder_output, decoder_hidden, final_prob, gen_prob, p_gen,  # shifted_context
