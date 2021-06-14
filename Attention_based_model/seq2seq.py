#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Pierre Godard, modified from Sean Robertson <https://github.com/spro/practical-pytorch>`
also borrowing heavily from <https://github.com/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb>
and <https://github.com/mzboito/word_discovery>
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

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data

from model import EncoderRNN, BahdanauAttnDecoderRNN, device
from util import timeSince, save_plot, save_and_plot_attention, plot_attention, count_params
from lang import EOS_NUM, prepare_data


# torch.set_num_threads(12)

# Training
# ========

# Preparing Training Data
# -----------------------

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_NUM)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def sentence_from_tensor(lang, tensor):
    local = list(tensor.view(-1).cpu().numpy())
    words = [lang.index2word[idx] for idx in local]
    return words

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor

# Training the Model
# ------------------

teacher_forcing_ratio = 1.1
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

def calculate_aux_loss(matrix):
    S = 0
    J = matrix.size(1)
    I = matrix.size(0)
    # TODO:  check indices
    for i in range(I - 1):
        alpha_i = matrix[i]
        alpha_i_plus_1 = matrix[i + 1]
        # dot = torch.dot(alpha_i, alpha_i_plus_1)
        # S += dot
        S += torch.dot(alpha_i, alpha_i_plus_1)
    return torch.abs(I - 1.0 - J - S)
    # return I - 1.0 - J - S

def train(input_sentence, input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, with_aux_loss):

    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    nll_loss = 0
    # aux_loss = 0
    aux_loss = torch.zeros(1, device=device)

    # get size of target sentences
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs, bridge_hidden, bridge_cell = encoder(input_tensor)

    # prepare input and output variables
    previous_embedding = None
    # use last hidden state from encoder to start decoder
    decoder_hidden = None
    decoder_cell = None

    # wether to use teacher forcing
    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # shunt scheduled sampling for now (should already be the case)
    use_teacher_forcing = True

    # for tensorboard, to be able to write an attention matrix sample
    decoder_attentions = torch.zeros(target_length, input_length, device=device)

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            current_input = target_tensor[di]  # for Ney-style
            decoder_output, decoder_context, decoder_hidden, decoder_cell, decoder_attention, previous_embedding = \
                decoder(input_sentence, current_input, previous_embedding, decoder_hidden, decoder_cell, encoder_outputs, bridge_hidden, bridge_cell)
            nll_loss += criterion(decoder_output, target_tensor[di])
            # current target is next input
            # previous_input = current_input  # Teacher forcing
            decoder_attentions[di] = decoder_attention.data

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # current_input = target_tensor[di]  # TODO: will not work in inference
            decoder_output, decoder_context, decoder_hidden, decoder_cell, decoder_attention = \
                decoder(current_input, previous_input, decoder_hidden, decoder_cell, encoder_outputs, bridge_hidden, bridge_cell)
            nll_loss += criterion(decoder_output, target_tensor[di])
            # get most likely word index from output
            topv, topi = decoder_output.topk(1)
            previous_input = topi.squeeze().detach()  # detach from history as input

            if previous_input.item() == EOS_NUM:
                break

    if with_aux_loss:
        aux_loss = calculate_aux_loss(decoder_attentions)
    # else:
    #     aux_loss = torch.zeros(1, requires_grad=False)

    # backprop
    loss = nll_loss + aux_loss
    loss.backward()  # TODO: backward seult tous les x forward
    # QUESTION: should I do the clipping?
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return nll_loss.item() / target_length, aux_loss.item() / target_length, decoder_attentions

# TODO: interface learning rate
def train_epochs(pairs, in_lang, out_lang, encoder, decoder, n_epochs,
                 print_every=1000, plot_every=100, learning_rate=0.001,
                 with_aux_loss=False, writer=None):
    start = time.time()
    plot_losses = []
    epoch_nll_loss_total = 0
    epoch_aux_loss_total = 0
    print_nll_loss_total = 0
    print_aux_loss_total = 0
    plot_nll_loss_total = 0
    plot_aux_loss_total = 0

    # different param init
    encoder.apply(init_weights)
    decoder.apply(init_weights)

    # optim
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # nb iters per epoch
    n_sents = len(pairs)
    training_pairs = [tensorsFromPair(pairs[i], in_lang, out_lang)
                      for i in range(n_sents)]

    criterion = nn.NLLLoss()
    current_iter = 0
    n_iters = n_epochs * n_sents
    logging.info(f"Running for {n_epochs} epoch(s) ({n_iters} iterations)")
    for e in range(1, n_epochs +1):
        random.shuffle(training_pairs)
        logging.info(f"Starting epoch {e}...")
        for i_sent in range(1, n_sents + 1):
            current_iter += 1
            training_pair = training_pairs[i_sent - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            # to log attention matrices and for word bias (list of source words)
            input_sentence = sentence_from_tensor(in_lang, input_tensor)

            nll_loss, aux_loss, attentions = train(input_sentence, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, with_aux_loss)
            print_nll_loss_total += nll_loss
            plot_nll_loss_total += nll_loss
            epoch_nll_loss_total += nll_loss
            if with_aux_loss:
                print_aux_loss_total += aux_loss
                plot_aux_loss_total += aux_loss
                epoch_aux_loss_total += aux_loss

            if current_iter % print_every == 0:
                print_nll_loss_avg = print_nll_loss_total / print_every
                print_nll_loss_total = 0
                logging.info(f"{timeSince(start, current_iter / n_iters)} ({current_iter} {current_iter / n_iters * 100:.0f}%) NLL:{print_nll_loss_avg:.4f}")
                if with_aux_loss:
                    print_aux_loss_avg = print_aux_loss_total / print_every
                    print_aux_loss_total = 0
                    logging.info(f"{timeSince(start, current_iter / n_iters)} ({current_iter} {current_iter / n_iters * 100:.0f}%) AUX:{print_aux_loss_avg:.4f}")

            if current_iter % plot_every == 0:
                plot_nll_loss_avg = plot_nll_loss_total / plot_every
                plot_losses.append(plot_nll_loss_avg)
                plot_nll_loss_total = 0
                if with_aux_loss:
                    plot_aux_loss_avg = plot_aux_loss_total / plot_every
                    plot_losses.append(plot_aux_loss_avg)
                    plot_aux_loss_total = 0

                # tensorboard
                if writer is not None:
                    writer.add_scalar('loss/nll_loss_avg', plot_nll_loss_avg, current_iter)
                    if with_aux_loss:
                        writer.add_scalar('loss/aux_loss_avg', plot_aux_loss_avg, current_iter)

            # log attention example
            # ("wa amituunga obia itsωω s elenge")
            ex_fr = "il a flanqué des coups de poing à son ami en pleine figure"
            cur_fr = ' '.join(input_sentence[:-1]).strip()
            if ex_fr == cur_fr:
                if writer is not None:
                    target = sentence_from_tensor(out_lang, target_tensor)
                    fig = plot_attention(input_sentence, target, attentions)
                    writer.add_figure('attentions', fig, e)

        if writer is not None:
            # log loss averaged on the epoch
            writer.add_scalar('loss/epoch_nll_loss_avg', epoch_nll_loss_total / n_sents, e)
            if with_aux_loss:
                writer.add_scalar('loss/epoch_aux_loss_avg', epoch_aux_loss_total / n_sents, e)
            epoch_nll_loss_total = 0
            epoch_aux_loss_total = 0


    return plot_losses



# Forced decoding
# ===============

def force_decode(encoder, decoder, source_sent, target_sent, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, source_sent)
        input_sentence = sentence_from_tensor(input_lang, input_tensor)
        target_tensor = tensorFromSentence(output_lang, target_sent)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs, bridge_hidden, bridge_cell = encoder(input_tensor)

        previous_embedding = None

        decoder_hidden = None
        decoder_cell = None

        decoder_attentions = torch.zeros(target_length, input_length)

        for di in range(target_length):
            current_input = target_tensor[di]  # for Ney-style
            decoder_output, decoder_context, decoder_hidden, decoder_cell, decoder_attention, previous_embedding = \
                decoder(input_sentence, current_input, previous_embedding, decoder_hidden, decoder_cell, encoder_outputs, bridge_hidden, bridge_cell)
            decoder_attentions[di] = decoder_attention.data
            # previous_input = current_input

        return decoder_attentions

def force_decode_corpus(pairs, input_lang, output_lang, encoder, decoder, base_dir, run_name, xp_name):
    n = len(pairs)
    for i in range(n):
        pair = pairs[i]
        attentions = force_decode(encoder, decoder, pair[0], pair[1], input_lang, output_lang)
        name = run_name + '/' + xp_name + '/attention.' + str(i)
        save_and_plot_attention(pair[0], pair[1], attentions, base_dir, name)


def main(arguments):

    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('-r', '--run_name', help="A name for the run", type=str)
    parser.add_argument('-f', '--fold_name', default=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), help="A fold number", type=str)
    parser.add_argument('-e', '--nb_epochs', help="Number of training epochs", type=int)
    parser.add_argument('-s', '--src', default='word', help="Source granularity", type=str)
    parser.add_argument('-t', '--trg', default='letter', help="Target representation", type=str)
    parser.add_argument('--batch_size', default=1, help="Batch size", type=int)
    parser.add_argument('--dropout', default=0.5, help="Dropout rate", type=float)
    parser.add_argument('--hidden', default=64, help="Hidden size", type=int)
    parser.add_argument('--out_emb_size', default=64, help="Output embeddings size", type=int)
    parser.add_argument('--is_gru', default=False, action='store_true')
    parser.add_argument('--update_first', default=False, action='store_true')
    parser.add_argument('--predict_current', default=False, action='store_true')
    parser.add_argument('--with_rnn_dropout', default=False, action='store_true')
    parser.add_argument('--do_ney', default=False, action='store_true')
    parser.add_argument('--word_bias', default=False, action='store_true')
    parser.add_argument('--aux_loss', default=False, action='store_true')
    parser.add_argument('--nolog_tb', default=False, action='store_true')
    # segmentation
    parser.add_argument('--segment', default=False, action='store_true')
    parser.add_argument('--segment_path', help='Path for attention matrices', type=str)
    parser.add_argument('--segmented', help='Path for segmented output', type=str)
    # score
    parser.add_argument('--score', default=False, action='store_true')
    parser.add_argument('--gold', help='Path for gold standard', type=str)
    # xp (temp)
    parser.add_argument('--mboshi_source', default=False, action='store_true')

    args = parser.parse_args(arguments)

    is_gru = 'True' if args.is_gru else 'False'
    xp_name = 'FRMB_e' + str(args.nb_epochs) + '_' + args.src + '_' + args.trg + '_batch' + str(args.batch_size) + '_drop' + str(args.dropout) + '_rnndrop' + str(args.with_rnn_dropout) + '_hidden' + str(args.hidden) + '_out' + str(args.out_emb_size) + '_gru' + is_gru + '_upfirst' + str(args.update_first) + '_predcur' + str(args.predict_current) + '_ney' + str(args.do_ney) + '_bias' + str(args.word_bias) + '_aux' + str(args.aux_loss) + '_' + args.fold_name
    # output_dir = '/home/pierre_godard_home_gmail_com/dat/'
    # dat_dir = '/home/pierre_godard_home_gmail_com/dev/attseg2/data/'
    output_dir = '/vol/work/godard/dat/THESIS/attseg2/'
    dat_dir = '/vol/work/godard/dat/THESIS/corpora/mboshi-french/'

    # output_dir = '/workdir/godard/dat/'
    # dat_dir = '/people/godard/dev/attseg2/data/'

    directory = os.path.dirname(output_dir + args.run_name + '/' + xp_name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.toy:
        toy_str = '_toy'
    else:
        toy_str = ''
    if args.mboshi_source:
        source_file = dat_dir + f'mboshi/mboshi_traindev{toy_str}_' + args.trg + '.word'
    else:
        source_file = dat_dir + f'french/french_traindev{toy_str}.' + args.src
    target_file = dat_dir + f'mboshi/mboshi_traindev{toy_str}_' + args.trg + '.char'

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
    # catch uncaught exceptions
    # def my_handler(_type, value, tb):
    #     logger.exception("Uncaught exception: {0}".format(str(value)))

    # Install exception handler
    # sys.excepthook = my_handler

    # TRAIN AND DECODE
    if args.train:
        # architecture
        batch_size = args.batch_size
        dropout_proba = args.dropout
        hidden_size = args.hidden
        emb_size = args.out_emb_size
        n_layers = 1

        # batch size 1
        try:
            assert batch_size == 1
        except AssertionError:
            sys.exit("\nThis version of seq2seq.py only supports batches of size 1.")

        input_lang, output_lang, pairs = prepare_data(source_file, target_file)
        logging.info(random.choice(pairs))

        # instantiate encoder
        encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout_p=dropout_proba,
                             is_gru=args.is_gru,
                             with_rnn_dropout=args.with_rnn_dropout).to(device)
        logging.info(encoder)
        total_encoder, trainable_encoder = count_params(encoder)
        logging.info(f"Encoder has {total_encoder} params ({trainable_encoder} trainable).\n")

        # instantiate decoder
        attn_decoder = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, emb_size, n_layers,
                                              dropout_p=dropout_proba,
                                              is_gru=args.is_gru,
                                              generate_first=not args.update_first,
                                              force_current=not args.predict_current,
                                              with_rnn_dropout=args.with_rnn_dropout,
                                              do_ney=args.do_ney,
                                              word_bias=args.word_bias).to(device)
        logging.info(attn_decoder)
        total_decoder, trainable_decoder = count_params(attn_decoder)
        logging.info(f"Decoder with attention has {total_decoder} params ({trainable_decoder} trainable).\n")

        # train
        encoder.train()
        attn_decoder.train()
        plot_losses = train_epochs(pairs, input_lang, output_lang, encoder, attn_decoder, args.nb_epochs, print_every=100,  with_aux_loss=args.aux_loss, writer=writer)
        save_plot(plot_losses, output_dir, args.run_name, xp_name)

        # force decode and write attention matrices
        encoder.eval()
        attn_decoder.eval()
        force_decode_corpus(pairs, input_lang, output_lang, encoder, attn_decoder, output_dir, args.run_name, xp_name)

    # SEGMENTATION
    if args.segment:
        if args.train:
            sentences_paths = glob.glob(output_dir + args.run_name + '/' + xp_name + '/attention' + "*.txt")
            output_path = output_dir + args.run_name + '/' + xp_name + '.segmented'
        else:
            sentences_paths = glob.glob(args.segment_path + "*.txt")
            output_path = args.segmented

        # clean output file
        if os.path.exists(output_path):
            os.remove(output_path)
        for index in range(0, len(sentences_paths)):
            file_path = segment.get_path(index, sentences_paths)
            final_str = segment.segment(file_path).replace(" </s>","").replace("</s>","") #removing EOS
            # segment
            segment.write_segmentation(final_str, output_path)

    # SCORING
    if args.score:
        if args.train and args.segment:
            gold = dat_dir + f'mboshi/mboshi_traindev{toy_str}_' + args.trg + '.word'
            segmented = output_path
        else:
            gold = args.gold
            segmented = args.segmented
        x, wp, wr, wf, bp, br, bf, lp, lr, lf = evaluation.get_prf_metrics(gold,
                                                                           segmented)
        results = f"X {(x * 100):.2f} WP {(wp * 100):.2f} WR {( wr * 100):.2f} WF {(wf * 100):.2f}  BP {(bp * 100):.2f} BR {( br * 100):.2f} BF {(bf * 100):.2f} LP {(lp * 100):.2f} LR {( lr * 100):.2f} LF {(lf * 100):.2f}"
        logging.info(results)
        with open(segmented + '.results', mode='w', encoding='utf-8') as r:
            r.write(results)

        # results in tensorboard
        if writer is not None:
            writer.add_scalar('results/x', x, 1)
            writer.add_scalar('results/wp', wp, 1)
            writer.add_scalar('results/wr', wr, 1)
            writer.add_scalar('results/wf', wf, 1)
            writer.add_scalar('results/bp', bp, 1)
            writer.add_scalar('results/br', br, 1)
            writer.add_scalar('results/bf', bf, 1)
            writer.add_scalar('results/lp', lp, 1)
            writer.add_scalar('results/lr', lr, 1)
            writer.add_scalar('results/lf', lf, 1)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
