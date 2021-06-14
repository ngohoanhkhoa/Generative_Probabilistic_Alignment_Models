#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batched version of Translation with a Sequence to Sequence Network and Attention
*************************************************************
Author: Pierre Godard, modified from Joost Bastings (<https://github.com/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb>)
and Sean Robertson (<https://github.com/spro/practical-pytorch>)
+ borrowing segmentation helpers from and <https://github.com/mzboito/word_discovery>
"""

from __future__ import unicode_literals, print_function, division
from io import open
import sys
import logging
from tensorboardX import SummaryWriter
import argparse
import os
import glob
import random
import time
import datetime
import evaluation
import segment
import math
import numpy as np

import torch
import torch.nn as nn
# from torch import optim
import torchtext

from util import timeSince, save_plot, save_and_plot_attention, plot_attention, count_params
# from batched_model import BatchedEncoderRNN, BatchedBahdanauAttnDecoderRNN, device, rebatch, Batch
from batched_model import device, make_model, USE_CUDA
from batched_lang import EOS_TOKEN, SOS_TOKEN, Batch, rebatch, prepare_data_torchtext, log_data_info, sentence_from_tensor


# torch.set_num_threads(12)

# Training
# ========

# Initialization
# --------------

clip = 5.0

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # TODO: essayer autre chose? loi normale
            m.bias.data.fill_(0.0)
    if type(m) == nn.Embedding:
        # doing it like the LIG system
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        # torch.nn.init.xavier_uniform_(m.weight)


# Loss
# ----

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

def calculate_aux_loss(matrix, src_length, trg_length):
    """ Calculate auxiliary loss to bias the number of units
    on the target side.

    matrix input of dimension [B, TN, SN] (batch size, target max length, source max length]
    """

    # TODO: try to make this more efficient
    # NOTE: Normally attentions are already masked so we can only make sure we have proper I and J
    total_aux = torch.zeros(1, device=device)
    batch_size = matrix.size(0)
    for k in range(batch_size):
        J = src_length[k]
        I = trg_length[k]
        S = 0
        for i in range(I - 2):  # SOS on target so I - 2 instead of I - 1
            alpha_i = matrix[k][i]
            alpha_i_plus_1 = matrix[k][i + 1]
            S += torch.dot(alpha_i, alpha_i_plus_1)
        # QUESTION: I think -1.0 is a mistake and shouldn't be there
        total_aux += torch.abs(I.float() - 1.0 - J.float() - S)
        # total_aux += I.float() - J.float() - S
    return total_aux

def batch_calculate_aux_loss(matrix, src_length, trg_length, trg_mask):
    """ Optimize calculation of auxiliary loss to bias the number of units
    on the target side.

    matrix input of dimension [B, TN, SN] (batch size, target max length, source max length]
    """

    # mask invalid attention weights
    # NB: 22-01-2019, remove last line (EOS) on the target side
    # (I don't know how to avoid a for loop here w/ fancy indexing or if it's possible)
    mask_trg_mask = torch.zeros_like(trg_mask)
    for i in range(trg_length.size(0)):
        mask_trg_mask[i, trg_length[i] - 2] = 1
    trg_mask.masked_fill_(mask_trg_mask == 1, 0.0)
    matrix.data.masked_fill_(trg_mask.unsqueeze(2) == 0, 0.0)

    matrix_transpose = matrix.transpose(1, 2)  # [B, TN, TS] -> [B, TS, TN]
    product = torch.bmm(matrix, matrix_transpose)  # [B, TN, TN]
    # cf. "To take a batch diagonal, pass in dim1=-2, dim2=-1."
    # https://pytorch.org/docs/0.4.1/torch.html?highlight=diagonal#torch.diagonal
    dots = torch.diagonal(product, offset=-1, dim1=-2, dim2=-1)
    S = dots.sum(dim=1)

    # NOTE: Re-adding substraction of 1.0 (SOS token present in trg_length but not in the attention matrices)
    # standard AUX
    aux_losses = trg_length.float() - 1.0 - src_length.float() - S

    # variant: adding a mult. factor (avg. ratio nb_mb_wd / nb_fr_wd on first 100 sent)
    # avg_ratio = 0.789176
    # aux_losses = trg_length.float() - 1.0 - avg_ratio * src_length.float() - S

    # approximate absolute value in a differentiable manner
    abs_values = torch.sqrt(aux_losses * aux_losses + 0.001)
    total_aux = torch.sum(abs_values)
    return total_aux


class AuxLossCompute:
    """Compute sum of NLL and auxiliary loss."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, matrix, src_length, trg_length, trg_mask, lambda_aux):
        x = self.generator(x)
        nll_loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        nll_loss = nll_loss / norm
        # old_aux_loss = calculate_aux_loss(matrix, src_length, trg_length)
        aux_loss = batch_calculate_aux_loss(matrix, src_length, trg_length, trg_mask)
        aux_loss = aux_loss / norm

        # NOTE: TEST, hard coded
        loss = nll_loss + lambda_aux * aux_loss
        # loss = (1.0 - lambda_aux) * nll_loss + lambda_aux * aux_loss

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        # QUESTION: note sure why SimpleLossCompute re-multiplies by norm...
        return loss.data.item() * norm, nll_loss.data.item() * norm , aux_loss.data.item() * norm


# Run training
# ------------

def run_epoch(data_iter, model, loss_compute, print_every=50, writer=None, batch_counter=0, curr_epoch=0, nb_epochs=1, aux_wait=0):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_nll_loss = 0
    total_aux_loss = 0
    print_tokens = 0

    if isinstance(loss_compute, AuxLossCompute):
        with_aux = True
        lambda_aux = max(curr_epoch - aux_wait, 0) / nb_epochs
    else:
        with_aux = False

    for i, batch in enumerate(data_iter, 1):

        out, _, pre_output, attention_vectors = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        if with_aux:
            loss, nll_loss, aux_loss = loss_compute(pre_output, batch.trg_y, batch.nseqs,
                                                    attention_vectors, batch.src_lengths, batch.trg_lengths, batch.trg_mask, lambda_aux)
            total_nll_loss += nll_loss
            total_aux_loss += aux_loss
        else:
            loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)

        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            logging.info("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.nseqs, print_tokens / elapsed))
            if with_aux:
                logging.info("Epoch Step: %d NLL loss: %f" % (i, nll_loss / batch.nseqs))
                logging.info("Epoch Step: %d AUX loss: %f" % (i, aux_loss / batch.nseqs))

            if writer is not None:
                writer.add_scalar("batched/loss", loss / batch.nseqs, batch_counter + i)
                if with_aux:
                    writer.add_scalar("batched/nll_loss", nll_loss / batch.nseqs, batch_counter + i)
                    writer.add_scalar("batched/aux_loss", aux_loss / batch.nseqs, batch_counter + i)
            start = time.time()
            print_tokens = 0

    # NOTE: with aux loss this isn't really a perplexity anymore
    return math.exp(total_loss / float(total_tokens)), i


def train(model, train_iter, decode_iter, SRC, TRG, pad_index, num_epochs=1, lr=0.001,
          print_every=100, with_aux_loss=False, aux_wait=0, writer=None):

    if USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.apply(init_weights)

    train_perplexities = []

    batch_cumul = 0
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}")
        model.train()
        if with_aux_loss:
            # QUESTION: why do we instantiate these objects for each epoch (as opposed to the whole training, outside the for loop) ?
            Loss = AuxLossCompute(model.generator, criterion, optim)
        else:
            Loss = SimpleLossCompute(model.generator, criterion, optim)
            
        train_perplexity, batch_number = run_epoch((rebatch(pad_index, b) for b in train_iter),
                                                   model,
                                                   Loss,
                                                   print_every=print_every,
                                                   writer=writer, batch_counter=batch_cumul, curr_epoch=epoch, nb_epochs=num_epochs, aux_wait=aux_wait)
        batch_cumul += batch_number
        logging.info("Train perplexity: %f" % train_perplexity)
        train_perplexities.append(train_perplexity)
        if writer is not None:
            writer.add_scalar('batched/train_perplexity', train_perplexity, epoch)
        

#        # log attention matrix
#        if writer is not None:
#            model.eval()
#            with torch.no_grad():
#                for i, batch in enumerate((rebatch(pad_index, b) for b in decode_iter)):
#                    out, _, pre_output, attention_vectors = model.forward(batch.src,
#                                                                          batch.trg,
#                                                                          batch.src_mask,
#                                                                          batch.trg_mask,
#                                                                          batch.src_lengths,
#                                                                          batch.trg_lengths)
#                    attentions = attention_vectors.squeeze().cpu()
#                    # cutting <s> and </s> to be compatible with save_and_plot_attention
#                    src = sentence_from_tensor(SRC, batch.src)
#                    trg = sentence_from_tensor(TRG, batch.trg)[1:]
#                    fig = plot_attention(src, trg, attentions)
#                    writer.add_figure('batched/attentions', fig, epoch)
#                    break


        # NOTE: to print exemples for each epoch
        # model.eval()
        # with torch.no_grad():
        #     saved_option = model.decoder.generate_first
        #     model.decoder.generate_first = False
        #     print_examples((rebatch(pad_index, x) for x in decode_iter),
        #                    model, n=2, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)
        #     model.decoder.generate_first = saved_option

    return train_perplexities


# Decoding
# ========

def force_decode_corpus(decode_iter, model, src_vocab, trg_vocab, base_dir, run_name, xp_name):
    """Assumes batch size of 1."""

    model.eval()
    with torch.no_grad():
        alignment_set = []
        for i, batch in enumerate(decode_iter):
#            alignment_sent = []
            out, _, pre_output, attention_vectors = model.forward(batch.src, batch.trg,
                                                  batch.src_mask, batch.trg_mask,
                                                  batch.src_lengths,
                                                  batch.trg_lengths)
            attentions = attention_vectors.squeeze().cpu()
            # cutting <s> and </s> to be compatible with save_and_plot_attention
            src = " ".join(sentence_from_tensor(src_vocab, batch.src)[:-1])
            trg = " ".join(sentence_from_tensor(trg_vocab, batch.trg)[1:-1])
            att = attentions.numpy()
            att_matrix = att[:-1, :-1]
            
#            print(src)
#            print(trg)
#            for s_index, s in enumerate(att[:-1, :-1]):
#                print(s_index+1, end=' ')
#                print(s)
#                
#            for s_index, s in enumerate(att[:-1, :-1]):
#                alignment_sent.append(str(s_index+1)+'-'+str(np.argmax(s)+1))
            alignment_set.append([src, trg, att_matrix])
            
        

        name = run_name + '/' + xp_name + '/attention'
        with open(base_dir + name + '.result', encoding='utf-8', mode='w') as file_handler:
            for sent_alignment in alignment_set:
                file_handler.write('Source: ' + sent_alignment[1] + '\n')
                file_handler.write('Target: ' + sent_alignment[0] + '\n')
                for s_index, s in enumerate(sent_alignment[2]):
                    file_handler.write(str(s_index+1) + ' ')
                    for v in s:
                        file_handler.write(str(v) + ' ')
                    file_handler.write('\n')
                file_handler.write('----------\n')
                        
        #save_and_plot_attention(src, trg, attentions, base_dir, name)


def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output, _ = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]

def print_examples(example_iter, model, n=2, max_len=100, sos_index=1, src_eos_index=None, trg_eos_index=None, src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None

    for i, batch in enumerate(example_iter):

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg

        result, _ = greedy_decode(
            model, batch.src, batch.src_mask, batch.src_lengths,
            max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i + 1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()

        count += 1
        if count == n:
            break


def main(arguments):

    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--train', default=False, action='store_true')
    #parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('-r', '--run_name', help="A name for the run", type=str)
    parser.add_argument('-f', '--fold_name', default='', help="A fold number", type=str)
    parser.add_argument('-e', '--nb_epochs', help="Number of training epochs", type=int)

    parser.add_argument('--batch_size', default=1, help="Batch size", type=int)
    parser.add_argument('--dropout', default=0.5, help="Dropout rate", type=float)
    parser.add_argument('--learning_rate', default=0.001, help="Dropout rate", type=float)
    parser.add_argument('--hidden', default=64, help="Hidden size", type=int)
    parser.add_argument('--out_emb_size', default=64, help="Output embeddings size", type=int)
    # NOTE: interface GRU/LSTM? Deprecated for now (only GRU)
    parser.add_argument('--is_gru', default=False, action='store_true')
    parser.add_argument('--update_first', default=False, action='store_true')
    # NOTE: predict_current is deprecated
    parser.add_argument('--predict_current', default=False, action='store_true')
    # NOTE: with_rnn_dropout is deprecated
    parser.add_argument('--with_rnn_dropout', default=False, action='store_true')
    # NOTE: do_ney is deprecated
    parser.add_argument('--do_ney', default=False, action='store_true')
    parser.add_argument('--word_bias', default=False, action='store_true')
    parser.add_argument('--aux_loss', default=False, action='store_true')
    parser.add_argument('--aux_wait', default=0, help="Wait nb epochs before adding aux loss", type=int)
    parser.add_argument('--nolog_tb', default=False, action='store_true')
    # segmentation
    parser.add_argument('--segment', default=False, action='store_true')
    parser.add_argument('--segment_path', help='Path for attention matrices', type=str)
    parser.add_argument('--segmented', help='Path for segmented output', type=str)
    # score
    parser.add_argument('--score', default=False, action='store_true')
    parser.add_argument('--gold', help='Path for gold standard', type=str)
    # xp (temp)
    parser.add_argument('--data_train_file', help="Data file directory", type=str)
    parser.add_argument('--data_test_file', help="Data file directory", type=str)
    parser.add_argument('--output_dir', help="Output file directory", type=str)

    args = parser.parse_args(arguments)

    is_gru = 'True' if args.is_gru else 'False'
    
    # xp_name = 'FRMB_e' + str(args.nb_epochs) + '_' + args.src + '_' + args.trg + '_batch' + str(args.batch_size) + '_drop' + str(args.dropout) + '_rnndrop' + str(args.with_rnn_dropout) + '_hidden' + str(args.hidden) + '_out' + str(args.out_emb_size) + '_gru' + is_gru + '_upfirst' + str(args.update_first) + '_predcur' + str(args.predict_current) + '_ney' + str(args.do_ney) + '_bias' + str(args.word_bias) + '_aux' + str(args.aux_loss) + '_' + args.fold_name
    xp_name = 'alignment' + str(args.nb_epochs) + '_' + \
    '_batch' + str(args.batch_size) + \
    '_drop' + str(args.dropout) + \
    '_lr' + str(args.learning_rate) + \
    '_hidden' + str(args.hidden) + \
    '_out' + str(args.out_emb_size) + \
    '_gru' + is_gru + \
    '_upfirst' + str(args.update_first) + \
    '_bias' + str(args.word_bias) + \
    '_aux' + str(args.aux_loss) + \
    '_wait' + str(args.aux_wait) + \
    '_' + args.fold_name + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    # PATHS
    output_dir = args.output_dir


    directory = os.path.dirname(output_dir + args.run_name + '/' + xp_name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    data_train_file = args.data_train_file
    data_test_file = args.data_test_file
    

    # TENSORBOARD
    if args.nolog_tb:
        writer = None
    else:
        writer = SummaryWriter(output_dir + 'tensorboard/' + args.run_name + '_' + xp_name)

    # LOGGER
    log_file = output_dir + args.run_name + '/' + xp_name + '.log'
    print(log_file)
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s: %(message)s",
        datefmt = '%H:%M:%S'
    )
    # set WARNING level for Matplotlib
    plot_logger = logging.getLogger('matplotlib')
    plot_logger.setLevel(logging.WARNING)

    # TRAIN AND DECODE
    if args.train:
        # architecture
        batch_size = args.batch_size
        dropout_proba = args.dropout
        hidden_size = args.hidden
        emb_size = args.out_emb_size
        n_layers = 1

        # batched version [WIP]
        train_set, test_set, SRC, TRG, PAD_INDEX = prepare_data_torchtext(data_train_file, data_test_file)
        # fields can be accessed with e.g.: train_set.fields['Source'].vocab.itos[6]

        # build BucketIterators
        train_iter = torchtext.data.BucketIterator(train_set, batch_size=batch_size,
                                                   train=True,
                                                   sort_within_batch=True,
                                                   sort_key=lambda x: (len(x.Source), len(x.Target)),
                                                   repeat=False,
                                                   device=device)
        decode_iter = torchtext.data.Iterator(test_set, batch_size=1,
                                              train=False,
                                              sort=False, repeat=False,
                                              device=device)

        model = make_model(SRC.vocab, len(SRC.vocab), len(TRG.vocab),
                           emb_size=emb_size, hidden_size=hidden_size,
                           num_layers=n_layers, dropout=dropout_proba, 
                           generate_first=not args.update_first, word_bias=args.word_bias)
        logging.info(model)
        total_model, trainable_model = count_params(model)
        logging.info(f"Model has {total_model} params ({trainable_model} trainable).\n")
        # also log some info on the data
        log_data_info(train_set, SRC, TRG)

        train_perplexities = train(model, train_iter, decode_iter, SRC, TRG, PAD_INDEX,
                                   num_epochs=args.nb_epochs, lr=args.learning_rate,
                                   print_every=10,
                                   with_aux_loss=args.aux_loss, aux_wait=args.aux_wait, writer=writer)
        
        logging.info(f"Train perplexities: {train_perplexities}")

        # force decode and write attention matrices
        force_decode_corpus((rebatch(PAD_INDEX, b) for b in decode_iter), model, SRC, TRG, output_dir, args.run_name, xp_name)


    # SEGMENTATION
#    if args.segment:
#        if args.train:
#            sentences_paths = glob.glob(output_dir + args.run_name + '/' + xp_name + '/attention' + "*.txt")
#            output_path = output_dir + args.run_name + '/' + xp_name + '.segmented'
#        else:
#            sentences_paths = glob.glob(args.segment_path + "*.txt")
#            output_path = args.segmented
#
#        # clean output file
#        if os.path.exists(output_path):
#            os.remove(output_path)
#        for index in range(0, len(sentences_paths)):
#            file_path = segment.get_path(index, sentences_paths)
#            # removing EOS
#            final_str = segment.segment(file_path).replace(" </s>","").replace("</s>","")
#            # segment
#            segment.write_segmentation(final_str, output_path)

#
#    # SCORING
#    if args.score:
#        if args.train and args.segment:
#            gold = dat_dir + f'mboshi/mboshi_traindev{toy_str}_' + args.trg + '.word'
#            segmented = output_path
#        else:
#            gold = args.gold
#            segmented = args.segmented
#        x, wp, wr, wf, bp, br, bf, lp, lr, lf = evaluation.get_prf_metrics(gold,
#                                                                           segmented)
#        results = f"X {(x * 100):.2f} WP {(wp * 100):.2f} WR {( wr * 100):.2f} WF {(wf * 100):.2f}  BP {(bp * 100):.2f} BR {( br * 100):.2f} BF {(bf * 100):.2f} LP {(lp * 100):.2f} LR {( lr * 100):.2f} LF {(lf * 100):.2f}"
#        logging.info(results)
#        with open(segmented + '.results', mode='w', encoding='utf-8') as r:
#            r.write(results)
#
#        # results in tensorboard
#        if writer is not None:
#            writer.add_scalar('results/x', x, 1)
#            writer.add_scalar('results/wp', wp, 1)
#            writer.add_scalar('results/wr', wr, 1)
#            writer.add_scalar('results/wf', wf, 1)
#            writer.add_scalar('results/bp', bp, 1)
#            writer.add_scalar('results/br', br, 1)
#            writer.add_scalar('results/bf', bf, 1)
#            writer.add_scalar('results/lp', lp, 1)
#            writer.add_scalar('results/lr', lr, 1)
#            writer.add_scalar('results/lf', lf, 1)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
