# coding=utf-8

import torch
from torch.nn import Module

from models import decoder
from models.decoder import DecoderAndPointer
from models.defaults import default_eda, default_attention, default_embedding, default_gru
from models.encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EDA_C(Module):
    def __init__(self, vocabulary_size, sos_token, eos_token, pad_token,
                 max_string_length=default_eda['string_max_length'], attention_size=default_attention['size'],
                 embedding_size=default_embedding['size'], hidden_size=default_gru['hidden_size'],
                 num_layers=default_gru['num_layers'], dropout=default_gru['dropout'], fixed_encoder=None):
        super().__init__()
        self.max_string_length = max_string_length
        self.attention_size = attention_size
        self.vocabulary_size = vocabulary_size
        if fixed_encoder:
            # Fix encoder's weights
            for p in fixed_encoder.parameters():
                p.requires_grad_(False)
            self.encoder = fixed_encoder
        else:
            self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size, num_layers, dropout)
        # self.decoder = Decoder(vocabulary_size)
        self.decoder = DecoderAndPointer(vocabulary_size, embedding_size, hidden_size, num_layers, dropout,
                                         attention_size, pad_token, shift_focus=True)
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, x, y=None):
        assert y is not None if self.training else y is None
        # Unpack input
        x, lengths = x
        batch_size = x.size(0)

        # encoder pass
        enc_annotations, annotations_len, enc_hidden = self.encoder(x, lengths)

        # decoder first hidden is last encoder hidden (using both forward and backward pass)
        fw_to_bw = enc_hidden.size(0) // 2
        dec_hidden = 0.5 * (enc_hidden[:fw_to_bw] + enc_hidden[fw_to_bw:])  # [num_dir x B x H]
        # dec_hidden = decoder.init_first_hidden(batch_size).float()

        # decoder first output (needed for attention) init to zeros
        dec_output = self.decoder.init_first_output(batch_size).float()

        # init first decoder input <sos>
        last_char_index = decoder.init_first_input_index(batch_size, self.sos_token).double()

        p_gen = decoder.init_first_p_gen(batch_size)

        first_prob = torch.zeros((batch_size, 1, self.vocabulary_size)).to(device)
        first_prob[:, 0, self.sos_token] = 1
        probabilities = [first_prob]
        if self.training:
            max_len = y.size(1)
        else:
            max_len = self.max_string_length
            sentence_end = [False for _ in range(batch_size)]
            attentions = [torch.zeros((batch_size, x.size(1))).to(device)]
            p_gens = [p_gen[0]]
        for pos in range(1, max_len):
            # Decoder
            # _, dec_output, dec_hidden, prob = self.decoder(last_char_index, dec_output, dec_hidden, enc_annotations)

            # DecoderAndPointer
            att, dec_output, dec_hidden, prob, _, p_gen = self.decoder(last_char_index, dec_output, dec_hidden,
                                                                       enc_annotations, x, p_gen)
            probabilities.append(prob)

            # find next char [Bx1x1]
            if self.training:
                target = [ref[pos] for ref in y]
                last_char_index = torch.tensor(target).double().view(batch_size, 1, 1).to(device)
            else:
                attentions.append(att)
                p_gens.append(p_gen.squeeze(2))

                last_char_index = prob.data.argmax(2).unsqueeze(2).double()

                # Update list of ended sentences
                last_char_list = last_char_index.view(-1).tolist()
                sentence_end_iter = [i for i, ch in enumerate(last_char_list) if ch == self.eos_token]
                for i in sentence_end_iter:
                    sentence_end[i] = True

                if all(sentence_end):
                    break
        if self.training:
            return torch.stack(probabilities, dim=1).squeeze()
        else:
            return torch.stack(probabilities, dim=1).squeeze(), torch.stack(attentions), torch.stack(p_gens)
