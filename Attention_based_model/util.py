#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
from io import open
import os
import sys
import numpy as np
from scipy.stats.stats import pearsonr
#import seaborn as sns



# Timing
# ======

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Timing
# ======

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# Visualizing Attention
# =====================

def plot_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    # hack
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return fig

def save_and_plot_attention(input_sentence, output_words, attentions, base_dir, name):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['</s>'], rotation=90)
    # hack
    ax.set_yticklabels([''] + output_words.split(' ') + ['</s>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(base_dir + name + '.png', dpi=fig.dpi)
    plt.gcf().clear()
    plt.close('all')
    # write plain text attention
    att = attentions.numpy()
    src_tokens = input_sentence.split() + ['</s>']
    trg_tokens = output_words.split() + ['</s>']
    with open(base_dir + name + '.txt', encoding='utf-8', mode='w') as attn_file:
        attn_file.write('\t' + '\t'.join(src_tokens) + '\n')
        for i in range(len(trg_tokens)):
            coeffs = '\t'.join(map(str, list(att[i])))
            attn_file.write('{}\t{}\n'.format(trg_tokens[i], coeffs))


def read_attention_file(report_name):
    """Reads a report file and outputs a numpy array.

    :report_name: the full path of a report file representing an attention matrix.
    :returns: a numpy array, src and trg labels

    """
    with open(report_name, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        b = None
        src = lines[0]
        for line in lines[1:]:
            line = line.strip().split('\t')
            if b is None:
                trg = line[0]
                b = np.array([line[1:]])
            else:
                trg += '\t' + line[0]
                b = np.append(b, [line[1:]], axis=0)
    a = b.astype(np.float)
    return a, src, trg

def write_attention(a, src, trg, new_report_name):
    """Writes a file corresponding to an attention matrix.

    :a: numpy array
    :src: string of source labels
    :trg: string of target labels
    :new_report_name: name of the output file
    :returns: nothing.
    """
    with io.open(new_report_name, encoding='utf-8', mode='w') as f:
        f.write(trg)
        s_list = src.strip().split('\t')
        for i in range(len(a)):
            line = s_list[i] + '\t' + '\t'.join(map(str, list(a[i,:])))
            f.write(line + '\n')

def plot_attention_from_file(attention_file):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # get attention, input and output words
    attentions, src, trg = read_attention_file(attention_file)
    # print(src, trg)

    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + src.split(), rotation=90)
    # hack
    ax.set_yticklabels([''] + trg.split())

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(os.path.join(os.path.dirname(attention_file), os.path.basename(attention_file) + '.png'), dpi=fig.dpi)
    plt.gcf().clear()
    plt.close('all')


# Plotting results
# ----------------

def save_plot(points, base_dir, run_name, xp_name):
    # fig = plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(base_dir + run_name + '/' + xp_name + '/loss_curve.png', dpi=fig.dpi)
    plt.gcf().clear()
    plt.close('all')


# To save graphs to tensorboard
# graph
# dummy_input = torch.zeros([5, 1], dtype=torch.long).to(device)
# with SummaryWriter(comment='encoder') as w:
#     with torch.no_grad():
#         w.add_graph(encoder1, dummy_input, verbose=True)

# graph
# with SummaryWriter(comment='decoder') as w:
#     with torch.no_grad():
#         w.add_graph(attn_decoder1, (torch.zeros([1, 1], dtype=torch.long),
#                                None,
#                                None,
#                                torch.randn(5,1,128),
#                                torch.randn(1,1,64),
#                                torch.randn(1,1,64)),
#                     verbose=False)

# Entropy related utils
# ---------------------

def calculate_entropy(attention):
    """Reads a numpy array corresponding to an attention matrix
    and outputs the average entropy.

    :att: a numpy array (src indexing lines and trg indexing columns)
    :returns: a float number corresponding to the average entropy

    """
    h_list = []
    for r in attention:
        h = stats.entropy(r)
        # we normalize by the entropy of the uniform distribution
        # on a sentence of that length so we have a percentage of
        # the max possible noise
        h /= math.log(r.shape[0])
        h_list += [h]
    avg_h = np.mean(np.array(h_list))
    return avg_h

def draw_hist(data, name):
    """ Save histogram from a list of data points.

    :data: a list of floats
    :name: the name for the image
    :returns: nothing
    """
    sns_plot = sns.distplot(data, bins=40)
    # sns_plot = sns.distplot(data, norm_hist=False, kde=False, bins=40)
    sns_plot.set_title(".".join(name.split('.')[1:-3]).split('/')[2])
    sns_plot.set_xlim([0.0,1.0])
    sns_plot.set_xlabel("entropy")
    sns_plot.set_ylabel("density")
    fig = sns_plot.get_figure()
    fig.savefig(f"{name}.png")
    plt.gcf().clear()

def entropy_histogram(scope, output_dir=None):
    """Builds a histogram representing the entropy of the
    matrices corresponding to the scope.

    :scope: a glob-like scope
    :output_dir: destination of the png histogram
    :returns: the global average entropy

    """
    d_list = glob.glob(scope + "**/reports/", recursive=True)
    assert len(d_list) == 1
    d = d_list[0]
    # /vol/work/godard/dat/att-seg/output/TRUE_E50_B1_D64_T10.0_Wfalse_Pfalse/reports/
    entropies = []
    for f in glob.glob(d + "report.*"):
        att, _, _ = read_attention(f)
        h = calculate_entropy(att)
        entropies += [h]
    if output_dir is not None:
        draw_hist(entropies, output_dir + os.path.basename(f))
    return np.mean(np.array(entropies))

def draw_training_curves(scope, output_dir):
    """Draw dev and training loss.

    :scope: a glob-like scope
    :output_dir: destination of the png curves
    :returns: None

    """
    import xnmt
    print('hello')
    if output_dir is not None:
        for f in glob.glob(scope + "**/*.log.yaml", recursive=True):
            with open(f, 'r') as s:
                doc = yaml.load(s)
            dev_loss = []
            train_loss = []
            mle_loss = []
            aux_loss = []
            for item in doc:
                if 'epoch' in item and 'score' in item:
                    dev_loss += [item['score'].loss]
                if 'data' in item and 'loss' in item:
                    if str(item['epoch']).endswith('.0'):
                        train_loss += [item['loss']]
                if 'loss_name' in item and 'loss' in item:
                    if item['loss_name'] == 'mle':
                        mle_loss += [item['loss']]
                    if item['loss_name'] == 'aux_loss':
                        aux_loss += [item['loss']]

            name = output_dir + os.path.basename(f)
            plt.title(".".join(name.split('.')[1:-3]).split('/')[2])
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.plot(dev_loss, label='dev')
            plt.plot(train_loss, label='train')
            plt.legend()
            plt.savefig(f"{name}.png")
            plt.gcf().clear()

            if len(mle_loss) > 0:
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.plot(mle_loss, label='mle')
                plt.plot(aux_loss, label='aux')
                plt.legend()
                plt.savefig(f"{name}_mle_aux.png")
                plt.gcf().clear()


# Word length bias stats
# ----------------------
def l(w):
    if w == "</s>":
        return 1.0
    else:
        return float(len(w))
vl = np.vectorize(l)

def print_word_length_att_correl(runs):
    """ Computes average correlation between word length and attention
    for each experiment in the list of runs provided.

    """
    print(runs)
    for run in runs:
        correl_runs = []
        p_value_runs = []
        print(f"\nCollecting run {run}")
        for dl in os.listdir(run):
            d = os.path.join(run, dl)
            if os.path.isdir(d):
                # correls = 0
                # p_values = 0
                # count_att = 0
                src_lengths = []
                s = []
                for att_file in glob.glob(d + "/attention*.txt"):
                    a, src, trg = read_attention_file(att_file)
                    src_lengths.extend(vl(src.strip().split('\t')))
                    s.extend(np.sum(a, axis=0))
                    # correls += correl
                    # p_values += p_value
                    # count_att += 1
                    assert len(src_lengths) == len(s)
                correl, p_value = pearsonr(src_lengths, s)
                # print(count_att)
                correl_runs.append(correl)
                p_value_runs.append(p_value)
                print('.', end='', flush=True)

        print(correl_runs)
        print(p_value_runs)


def draw_len_dist(filename):
    # print(filename)
    lengths = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        lengths.extend([len(line.strip().split())])
    # print(lengths[0:5])
    bins = np.arange(0, 15 + 1, 1)
    sns_plot = sns.distplot(lengths, hist=True, kde=True, kde_kws={"bw": 0.3}, rug=False, bins=bins-0.5)
    # sns_plot.set_title(filename)
    sns_plot.set_xlim([0.0,15.0])
    plt.xticks(bins)
    plt.yticks([0,0.05,0.1,0.15,0.2,0.25])
    sns_plot.set_xlabel("sentence length", fontsize=15)
    sns_plot.set_ylabel("density", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    # format = 'eps', dpi = 1000
    # plt.savefig(f"{filename}.eps", format='eps', dpi=1000)
    # fig.savefig(f"{filename}.png", dpi=1000)
    fig.savefig(f"{filename}.png", dpi=300)
    plt.gcf().clear()



# ----------------------

def main(arguments):
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=str, nargs="*", help='Run paths, e.g. "./run_01/ ./run_02/"')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--correl', action='store_true', help='Correlation between word length and attention quantity.')
    parser.add_argument('--segmented', type=str, help='Segmentated file.')
    parser.add_argument('--lendist', action='store_true', help='Plot length distribution for a segmentation.')
    args = parser.parse_args(arguments)

    if args.correl:
        print_word_length_att_correl(args.runs)
    elif args.lendist:
        draw_len_dist(args.segmented)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
