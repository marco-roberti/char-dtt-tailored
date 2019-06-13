# coding=utf-8

import torch
from torch.nn import Module, Embedding, GRU

from models import decoder
from models.decoder import length_penalty
from models.defaults import default_eda, default_attention, default_embedding, default_gru
from models.empty_decoder import EmptyDecoderAndPointer
from models.encoder import forward_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EDA_CS(Module):
    def __init__(self, vocabulary_size, sos_token, eos_token, pad_token, attention_size=default_attention['size'],
                 embedding_size=default_embedding['size'], hidden_size=default_gru['hidden_size'],
                 num_layers=default_gru['num_layers'], dropout=default_gru['dropout'], shift_focus=True):
        super().__init__()
        self.attention_size = attention_size
        self.vocabulary_size = vocabulary_size

        self.embedding = Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.gru1 = GRU(2 * hidden_size + embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                        dropout=dropout, bidirectional=True)
        self.gru2 = GRU(2 * hidden_size + embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                        dropout=dropout, bidirectional=True)
        self.decoder = EmptyDecoderAndPointer(vocabulary_size, embedding_size, hidden_size, attention_size, pad_token,
                                              shift_focus=shift_focus)
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, x, y=None, reverse=False, beam_size=default_eda['beam_size'], alpha=default_eda['alpha'],
                max_len=default_eda['string_max_length']):
        assert y is not None if self.training else y is None
        # Choose gru
        if reverse:
            gru_enc = self.gru2
            gru_dec = self.gru1
        else:
            gru_enc = self.gru1
            gru_dec = self.gru2

        # Unpack input
        x, lengths = x

        # encoder pass
        enc_annotations, annotations_len, enc_hidden = forward_encoder(x, lengths, self.embedding, gru_enc)

        return self.decode_training(x, y, gru_dec, enc_hidden, enc_annotations) if self.training \
            else self.decode_eval(x, beam_size, alpha, gru_dec, enc_hidden, enc_annotations, max_len)

    def decode_training(self, x, y, gru_dec, enc_hidden, enc_annotations):
        batch_size = x.size(0)
        # decoder first hidden is last encoder hidden
        dec_hidden = enc_hidden

        # decoder first output (needed for attention) init to zeros
        dec_output = self.decoder.init_first_output(batch_size).float()

        # init first decoder input <sos>
        last_char_index = decoder.init_first_input_index(batch_size, self.sos_token).double()

        p_gen = decoder.init_first_p_gen(batch_size)

        first_prob = torch.zeros((batch_size, 1, self.vocabulary_size)).to(device)
        first_prob[:, 0, self.sos_token] = 1
        probabilities = [first_prob]

        max_len = y.size(1)
        for pos in range(1, max_len):
            # DecoderAndPointer
            att, dec_output, dec_hidden, prob, _, p_gen = self.decoder(last_char_index,
                                                                       dec_output[:, :, :gru_dec.hidden_size],
                                                                       # only fw
                                                                       dec_hidden, enc_annotations, x, p_gen,
                                                                       self.embedding, gru_dec)
            probabilities.append(prob)

            # find next char [Bx1x1]
            target = [ref[pos] for ref in y]
            last_char_index = torch.tensor(target).double().view(batch_size, 1, 1).to(device)
        return torch.stack(probabilities, dim=1).squeeze()

    def decode_eval(self, x, beam_size, alpha, gru_dec, enc_hidden, enc_annotations, max_len):
        # beam element: <likelihood, probabilities, last_index, attentions, dec_output, dec_hidden, p_gens>
        beam = [(torch.tensor([0.]).to(device),
                 [torch.tensor([0. if i != self.sos_token else 1. for i in range(self.vocabulary_size)]).to(device)],
                 self.sos_token,
                 [torch.zeros((1, x.size(1))).to(device)],
                 self.decoder.init_first_output(1).float(),
                 # decoder first hidden is last encoder hidden
                 enc_hidden,
                 [decoder.init_first_p_gen(1)])]
        for pos in range(1, max_len):
            new_elements = []
            for beam_elem in beam:
                # Check if this beam element is complete
                if beam_elem[2] == self.eos_token:
                    new_elements.append(beam_elem)
                    continue
                likelihood, probs, last_index, attentions, dec_output, dec_hidden, p_gens = beam_elem
                # DecoderAndPointer
                att, dec_output, dec_hidden, prob, _, p_gen = self.decoder(
                    torch.tensor([[[last_index]]], dtype=torch.float64).to(device),
                    dec_output[:, :, :gru_dec.hidden_size],
                    # only fw
                    dec_hidden, enc_annotations, x, p_gens[-1],
                    self.embedding, gru_dec)

                probs = probs + [prob.squeeze()]
                attentions = attentions + [att]
                p_gens = p_gens + [p_gen]

                # Expand beam
                best_probs, top_indices = prob.topk(beam_size, 2)
                top_indices = top_indices.squeeze().tolist() if beam_size > 1 else [top_indices.item()]

                for i in range(beam_size):
                    # Update list of ended sentences
                    next_char = top_indices[i]
                    new_elements.append((likelihood + best_probs[:, :, i].squeeze(), probs, next_char, attentions,
                                         dec_output, dec_hidden, p_gens))

            new_elements.sort(key=lambda elem: elem[0][0] / length_penalty(len(elem[1]), alpha), reverse=True)
            beam = new_elements[:beam_size]
        _, probabilities, _, attentions, _, _, p_gens = beam[0]
        probabilities = torch.stack(probabilities)
        attentions = torch.stack(attentions)
        p_gens = torch.stack(p_gens)
        return probabilities, attentions, p_gens
