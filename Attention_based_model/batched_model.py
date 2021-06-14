#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from lang import SOS_NUM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()

def make_model(src_vocab, src_vocab_len, tgt_vocab_len, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1, generate_first=True, word_bias=False):
    """Helper: Construct a model from hyperparameters."""

    attention = BahdanauAttention(hidden_size, word_bias=word_bias, src_vocab=src_vocab)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout, generate_first=generate_first),
        # TODO: pretrained embeddings for src
        # embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))
        nn.Embedding(src_vocab_len, emb_size), dropout,
        nn.Embedding(tgt_vocab_len, emb_size), dropout,
        Generator(hidden_size, tgt_vocab_len))

    return model.cuda() if USE_CUDA else model

# Helper for word length bias
def f(x, vocab):
    w = vocab.itos[x]
    if w == "</s>":
        return 1.0
    elif w == "<pad>":
        return 0.0
    else:
        return float(len(w))
vlength = np.vectorize(f)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, src_dropout, trg_embed, trg_dropout, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.src_dropout = nn.Dropout(src_dropout)
        self.trg_dropout = nn.Dropout(trg_dropout)

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_dropout(self.src_embed(src)), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src, src_mask, trg, trg_mask, decoder_hidden=None):
        return self.decoder(self.trg_dropout(self.trg_embed(trg)), encoder_hidden, encoder_final, src, src_mask, trg_mask, hidden=decoder_hidden)


class Generator(nn.Module):
    """Define linear / tanh + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        # NOTE: doubled intermediary hidden dim and added bias
        self.proj = nn.Linear(2 * hidden_size, vocab_size, bias=True)

    def forward(self, x):
        # NOTE: added a non-linearity
        return F.log_softmax(self.proj(torch.tanh(x)), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        # NOTE: could add RNN dropout and LSTM option
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # NOTE: changed bridge (backward final only)
        # fwd_final = final[0:final.size(0):2]
        # bwd_final = final[1:final.size(0):2]
        # final = torch.cat([fwd_final, bwd_final],
        #                   dim=2)  # [num_layers, batch, 2*dim]
        final = final[1:final.size(0):2]

        return output, final


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None,
                 word_bias=False, src_vocab=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        self.word_bias = word_bias
        self.src_vocab = src_vocab

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, encoder_hidden=None, src=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)  # [B, 1, N], could also do transpose?

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # Word bias
        if self.word_bias:
            source = src.unsqueeze(1).numpy()  # [B, 1, N]
            lengths = vlength(source, self.src_vocab)
            # means = np.mean(lengths)
            # norm_avg_lengths = lengths / means
            # word_lengths = torch.tensor(norm_avg_lengths, requires_grad=False).float().to(device)
            # weighted_alphas = alphas * word_lengths
            word_lengths = torch.tensor(lengths, requires_grad=False).float().to(device)
            norm = alphas * word_lengths
            norm = norm.sum(2)
            normalized_word_lengths = word_lengths / norm.unsqueeze(2)
            weighted_alphas = alphas * normalized_word_lengths

            # The context vector is the weighted sum of the values.
            context = torch.bmm(weighted_alphas, encoder_hidden)  # [B, 1, N] * [B, N, 2D] -> [B, 1, 2D]
        else:
            # The context vector is the weighted sum of the values.
            context = torch.bmm(alphas, encoder_hidden)  # [B, 1, N] * [B, N, 2D] -> [B, 1, 2D]

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        # QUESTION: OUPS, shouldn't I return the weighted alphas if using word bias??
        if self.word_bias:
            return context, weighted_alphas
        else:
            return context, alphas


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention,
                 num_layers=1, dropout=0.5, bridge=True,
                 generate_first=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        # self.dropout = dropout
        self.generate_first = generate_first

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        # NOTE: using only bwd encoder state (hidden_size instead of 2*hidden_size for first arg)
        self.bridge = nn.Linear(hidden_size, hidden_size,
                                bias=True) if bridge else None

        # NOTE: changed projection size to 2*hidden_size and added bias
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size, 2 * hidden_size, bias=True)

    def forward_step(self, prev_embed, curr_embed, encoder_hidden, src, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)

        y_{i-1}: prev_embed
        s_{i-1}: hidden
        s_{i}: output
        c_i: context
        h: proj_key
        """

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        # a(s_{i-1}, h)
        context, attn_probs = self.attention(query=query, proj_key=proj_key, encoder_hidden=encoder_hidden, src=src, mask=src_mask)

        if self.generate_first:
            # output g(y_{i-1}, s_{i-1}, c_i)
            pre_output = torch.cat([prev_embed, hidden.transpose(0, 1), context], dim=2)
            pre_output = self.pre_output_layer(pre_output)

            # update rnn hidden state
            # s_i = f(s_{i-1}, y_{i}, c_i)
            rnn_input = torch.cat([curr_embed, context], dim=2)
            output, hidden = self.rnn(rnn_input, hidden)

        else:
            # update rnn hidden state
            # s_i = f(s_{i-1}, y_{i-1}, c_i)
            rnn_input = torch.cat([prev_embed, context], dim=2)
            output, hidden = self.rnn(rnn_input, hidden)

            # output g(y_{i-1}, s_i, c_i)
            pre_output = torch.cat([prev_embed, output, context], dim=2)
            pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output, attn_probs

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src, src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        # /!\ trg_mask.size(-1) = trg_embed.size(1) - 1 because we removed <s>
        # but that's a little funky. That allows us to implement generate_first option
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)  # [B, N, D * 2] -> [B, N, D]

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        attentions = []

        # unroll the decoder RNN for max_len steps
        # (size of the  longest target in the batch)
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # [B, 1, D]
            if self.generate_first:
                curr_embed = trg_embed[:, i + 1].unsqueeze(1)  # [B, 1, D]
            else:
                curr_embed = None
            output, hidden, pre_output, attn_probs = self.forward_step(prev_embed, curr_embed, encoder_hidden, src, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
            attentions.append(attn_probs)

        decoder_states = torch.cat(decoder_states, dim=1)  # [B, N, D]
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        attention_vectors = torch.cat(attentions, dim=1)
        return decoder_states, hidden, pre_output_vectors, attention_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))
