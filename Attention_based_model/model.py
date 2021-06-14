#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lang import SOS_NUM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.5, is_gru=False, with_rnn_dropout=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.is_gru = is_gru
        self.with_rnn_dropout = with_rnn_dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        if with_rnn_dropout:
            self.rnn_dropout = nn.Dropout(dropout_p)
        if is_gru:
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, word_inputs):
        # running over the whole input seq at once
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)  # second dim is batch size (1 here)
        embedded = self.dropout(embedded)
        # cf. h_n of shape (num_layers * num_directions, batch, hidden_size)
        if self.is_gru:
            output, final_hidden = self.rnn(embedded)
            bridge_hidden = final_hidden[1].unsqueeze(0)
            return output, bridge_hidden, bridge_hidden  # to keep same output signature
        else:
            output, (final_hidden, final_cell) = self.rnn(embedded)
            if self.with_rnn_dropout:
                output = self.rnn_dropout(output)
            bridge_hidden = final_hidden[1].unsqueeze(0)
            bridge_cell = final_cell[1].unsqueeze(0)
            return output, bridge_hidden, bridge_cell


class Attn(nn.Module):
    def __init__(self, hidden_size, output_emb_size, do_ney=False):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.output_emb_size = output_emb_size
        self.do_ney = do_ney
        if do_ney:
            self.ney_layer = nn.Linear(output_emb_size, hidden_size, bias=False)
        # here 2 * hidden_size with bidirectional encoder
        self.key_layer = nn.Linear( 2 * hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, current_word_emb):
        keys = self.key_layer(encoder_outputs)
        query = self.query_layer(hidden)
        if self.do_ney:
            ney = self.ney_layer(current_word_emb)
            scores = self.energy_layer(torch.tanh(query + keys + ney))
        else:
            scores = self.energy_layer(torch.tanh(query + keys))
        # the whole squeeze business seems for batches but here it's a little odd
        energies = scores.squeeze()
        return F.softmax(energies, dim=0).unsqueeze(0).unsqueeze(0)


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, emb_size, n_layers=1, dropout_p=0.5,
                 is_gru=False, generate_first=True, force_current=True, with_rnn_dropout=False,
                 do_ney=False, word_bias=False):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.is_gru = is_gru
        self.generate_first = generate_first
        self.force_current = force_current
        self.with_rnn_dropout = with_rnn_dropout
        self.do_ney = do_ney
        self.word_bias = word_bias

        # Define layers
        # output_size is the output vocab size
        self.embedding = nn.Embedding(output_size, emb_size)
        self.dropout = nn.Dropout(dropout_p)
        if with_rnn_dropout:
            self.rnn_dropout = nn.Dropout(dropout_p)

        self.attn = Attn(hidden_size, emb_size, do_ney)

        if is_gru:
            self.rnn = nn.GRU(emb_size + hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        else:
            self.rnn = nn.LSTM(emb_size + hidden_size * 2, hidden_size, n_layers,
                               dropout=dropout_p) # hidden_size + 2 * hidden_size + emb_size
        # Two layers needed
        self.out1 = nn.Linear(hidden_size * 3 + emb_size, hidden_size * 2)
        self.out2 = nn.Linear(hidden_size * 2, output_size)  # Alex was worried 64 would be too small
        # * 2 with bidir. encoder  /!\ not anymore with backward last state
        self.bridge = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bridge_cell = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input_sentence, current_word, previous_word_emb, previous_state, last_cell, encoder_outputs, bridge_hidden, bridge_cell):
        # Running forward for a single decoder time step, but using all encoder outputs

        # Initializing with encoder last state
        if previous_state is None:
            previous_state = torch.tanh(self.bridge(bridge_hidden))
        if last_cell is None:
            last_cell = torch.tanh(self.bridge_cell(bridge_cell))

        # Get the embedding of the current input word (last output word)
        if previous_word_emb is None:
            previous_word_emb = self.embedding(torch.tensor([[SOS_NUM]], device=device)).view(1, 1, -1)
            previous_word_emb = self.dropout(previous_word_emb)

        # previous_word_emb = self.embedding(previous_word).view(1, 1, -1)
        # previous_word_emb = self.dropout(previous_word_emb)

        # Calculate attention weights and apply to encoder outputs
        current_word_emb = self.embedding(current_word).view(1, 1, -1)
        current_word_emb = self.dropout(current_word_emb)
        attn_weights = self.attn(previous_state[-1], encoder_outputs, current_word_emb)

        # Word bias
        if self.word_bias:
            f = lambda x: float(len(x)) if x != "EOS" else 1.0
            vf = np.vectorize(f)
            lengths = vf(np.array(input_sentence))
            means = np.mean(lengths)
            norm_avg_lengths = lengths / means
            lengths_tensor = torch.tensor(norm_avg_lengths, requires_grad=False).float().to(device)
            lengths_tensor = lengths_tensor.unsqueeze(0).unsqueeze(0)
            # element wise multiplication
            attn_weights = attn_weights * lengths_tensor

        # e.g. output of bmm: 1x64 (ou 1x128 with bidirectional)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))

        if self.generate_first:
            # g(y_{i-1}, s_{i-1}, c_i)
            previous_state = previous_state.squeeze(0)  # B x N
            context = context.squeeze(0)  # B x N
            if self.with_rnn_dropout:
                pre_out = torch.cat([previous_word_emb.squeeze(0), self.rnn_dropout(previous_state), context], dim=1)
            else:
                pre_out = torch.cat([previous_word_emb.squeeze(0), previous_state, context], dim=1)
            # Additional layer + tanh
            # (could be replace by maxout layer as in Badhanau)
            output = torch.tanh(self.out1(pre_out))
            output = F.log_softmax(self.out2(output), dim=1)

            # s_i = f(s_{i-}, y_{i}, c_i)
            if not self.force_current:
                # get current generated word (and embed it)
                opv, topi = output.topk(1)
                current_word_emb = self.embedding(topi).view(1, 1, -1)
                current_word_emb = self.dropout(current_word_emb)

            previous_state = previous_state.unsqueeze(0) # ugly after squeeze
            context = context.unsqueeze(0)  # ugly after squeeze
            rnn_input = torch.cat((current_word_emb, context), 2)
            if self.is_gru:
                _, state = self.rnn(rnn_input, previous_state)
            else:
                _, (state, cell) = self.rnn(rnn_input, (previous_state, last_cell))
            context = context.squeeze(0)  # B x N

        else:
            # s_i = f(s_{i-1}, y_{i-1}, c_i)
            rnn_input = torch.cat((previous_word_emb, context), 2)
            if self.is_gru:
                _, state = self.rnn(rnn_input, previous_state)
            else:
                _, (state, cell) = self.rnn(rnn_input, (previous_state, last_cell))

            # g(y_{i-1}, s_i, c_i)
            state = state.squeeze(0)  # B x N
            context = context.squeeze(0)  # B x N
            if self.with_rnn_dropout:
                pre_out = torch.cat([previous_word_emb.squeeze(0), self.rnn_dropout(state), context], dim=1)
            else:
                pre_out = torch.cat([previous_word_emb.squeeze(0), state, context], dim=1)
            # Additional layer + tanh
            # (could be replace by maxout layer as in Badhanau)
            output = torch.tanh(self.out1(pre_out))
            output = F.log_softmax(self.out2(output), dim=1)
            state = state.unsqueeze(0)  # seq_len (1) x B x N

        # Return final output, hidden state, and attention weights (for visualization)
        if self.is_gru:
            return output, context, state, state, attn_weights, current_word_emb  # to keep output signature identical to LSTM
        else:
            return output, context, state, cell, attn_weights, current_word_emb



