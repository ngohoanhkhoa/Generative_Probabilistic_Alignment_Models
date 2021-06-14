#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import numpy as np
import sys
import logging
import argparse
import pickle
import glob
import re
import os
import tabulate

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def strip_list(lst):
    new_list = []
    for line in lst:
        newline = strip_accents(line.strip())
        new_list += [newline]
    return new_list

def get_token_type_avgs(segmented, nb_truncate=0, strip_accents=False):
    """ Calculate number of tokens and types, and average token length
    from an input file.
    """

    with open(segmented, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        nb_lines = len(lines)
        if strip_accents:
            lines = strip_list(lines)
        if nb_truncate != 0:
            nb_test_lines = min(nb_truncate,len(lines))
        else:
            nb_test_lines = len(lines)

        words = []
        for line in lines[:nb_test_lines]:
            words += line.strip().split()
        chars = []
        for word in words:
            chars += list(word)
        tok = len(words)
        typ = len(set(words))
        avg_tok_len = float(len(chars))/float(len(words))
        avg_sent_len = float(len(words) / nb_lines)
    # return "{:,}".format(tok), "{:,}".format(typ), "{0:.2f}".format(avg)
    return tok, typ, avg_tok_len, avg_sent_len

def f_measure(precision, recall):
    """ Standard f-measure from precision and recall.
    """
    if precision + recall == 0.0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

def make_vec(line):
    """ Create a vector indicating if a word boundary (whitespace) exists
    after each character of the input line.

    Boundary before first character and after last character are ignored.
    """
    b = []
    for j in range(0, len(line)-1):
        if line[j+1] == " ":
            b.append(True)
        else:
            if line[j] != " ":
                b.append(False)
    if len(line) == 1:
        # special edge case with a line of 1 character
        # (would otherwise return an empty vector)
        return np.array([False])
    return np.array(b)

def count_token_hits(gold, seg):
    """ Count the number of words correctly segmented in seg
    w.r.t. the gold reference gold.
    """

    hits = 0
    left_bound = True
    for i in range(len(seg)):
        if seg[i] and gold[i]:
            if left_bound:
                hits += 1
            left_bound = True
        elif (seg[i] and not gold[i]) or (not seg[i] and gold[i]):
            left_bound = False
    if left_bound:
        hits += 1
    return hits

def get_prf_metrics(gold, segmented, truncation=0, strip_accents=False):
    """ Calculates P/R/F metrics for tokens, boundaries and types.
    Calculate sentence exact-match X as well.

    Truncation indicates the number of lines to consider. 0 means all lines.
    """

    g = open(gold, 'r', encoding='utf-8')
    s = open(segmented, 'r', encoding='utf-8')
    glines = g.readlines()
    slines = s.readlines()
    g.close()
    s.close()
    if strip_accents:
        glines = strip_list(glines)
        slines = strip_list(slines)
    try:
        assert len(glines) == len(slines)
    except AssertionError:
        gn = len(glines)
        sn = len(slines)
        sys.exit(f"Reference ({gn}) and segmented ({sn}) files don't have the same number of lines.")

    # nlines = len(glines)
    exact_match = 0
    gboundaries = 0
    sboundaries = 0
    gwordcount = 0
    swordcount = 0
    bhits = 0
    thits = 0
    glexicon = []
    slexicon = []
    if truncation != 0:
        nb_test_lines = min(truncation,len(glines))
    else:
        nb_test_lines = len(glines)
    for i in range(nb_test_lines):
        gline = glines[i]
        sline = slines[i]
        try:
            assert ("".join(gline.split()) == "".join(sline.split()))
        except AssertionError:
            logging.error(gline)
            logging.error(sline)
            sys.exit("Segmentation and reference are not compatible on line " + str(i+1) + ".")
        gwords = gline.strip().split()
        gline = ' '.join(gwords)  # ensures single whitespaces
        swords = sline.strip().split()
        sline = ' '.join(swords)  # ensures single whitespaces
        # exact match
        if gwords == swords:
            exact_match += 1
        # lexicon
        glexicon.extend(gwords)
        slexicon.extend(swords)
        # boundaries
        gboundaries += len(gwords) - 1
        gwordcount += len(gwords)
        sboundaries += len(swords) - 1
        swordcount += len(swords)
        gbound_vec = make_vec(gline)
        sbound_vec = make_vec(sline)
        boundhits = gbound_vec & sbound_vec
        bhits += len(boundhits[boundhits[:]==True])
        # words (tokens)
        thit = count_token_hits(gbound_vec, sbound_vec)
        thits += thit

    # Results
    glexicon = set(glexicon)
    slexicon = set(slexicon)
    x = exact_match / nb_test_lines
    lp = len(slexicon & glexicon) / len(slexicon)
    lr = len(slexicon & glexicon) / len(glexicon)
    lf = f_measure(lp, lr)
    if sboundaries == 0:
        bp = 0
    else:
        bp = bhits / sboundaries
    br = bhits / gboundaries
    bf = f_measure(bp, br)
    wp = thits / swordcount
    wr = thits / gwordcount
    wf = f_measure(wp, wr)
    return x, wp, wr, wf, bp, br, bf, lp, lr, lf

def process_results(runs, nb_truncate, strip_accents, is_verbose=True, output_format='plain', pickle_file=None):
    """ Process results for a run.
    """

    count = 0
    table = [['run', 'target', 'size', 'rep', 'gran', 'epoch', 'batch', 'drop', 'lrate', 'hidden', 'out', 'gru', 'upfirst', 'bias', 'aux', 'wait', 'timeid', 'X', 'WP', 'WR', 'WF', 'BP', 'BR', 'BF', 'LP', 'LR', 'LF', 'token', 'type', 'avg_tok', 'avg_sent']]

    # run name, e.g. "./run_01/"
    nmissed = 0
    print(runs)
    for run in runs:
        print(f"\nCollecting run {run}")
        for segfile in glob.glob(run + "**/*.segmented", recursive=True):
            # print(segfile)
            if 'EN' in segfile:
                target = 'english'
            elif 'MB' in segfile:
                target = 'mboshi'
            else:
                sys.exit('Unable to identify source language in path.')
            # file name example: FRMB_e800_word_letter_batch64_drop0.5_lr0.001_hidden64_out64_gruTrue_upfirstFalse_biasFalse_auxFalse_wait0_2018-12-30_16:30:30.segmented
            match = re.search(r'.*_e(.*?)_(.*?)_(.*?)_batch(.*?)_drop(.*?)_lr(.*?)_hidden(.*?)_out(.*?)_gru(.*?)_upfirst(.*?)_bias(.*?)_aux(.*?)(?:_wait(.*?))?_([^.]*)',
                              os.path.basename(segfile))
            if match is not None:
                # print(os.path.basename(segfile))
                epoch = match.group(1)
                gran = match.group(2)
                rep = match.group(3)
                size = 'traindev'  # NOTE: would have to interface that if needed
                batch = match.group(4)
                drop = match.group(5)
                lrate = match.group(6)
                hidden = match.group(7)
                out = match.group(8)
                gru = match.group(9)
                upfirst = match.group(10)
                bias = match.group(11)
                aux = match.group(12)
                wait = match.group(13)
                timeid = match.group(14)

                # print(epoch,gran,rep,size,batch,drop,lrate,hidden,out,gru,upfirst,bias,aux,wait,timeid)

                # Evaluate the segmentation
                # NB: target instead of source here because of the neural architecture/segmentation
                gold_base = '/vol/work/godard/dat/THESIS/corpora/' + target + '-french/' + target + '/'
                # print(gold_base)
                print('.', end='', flush=True)
                if target == 'mboshi':
                    rep_str = '_' + rep
                else:
                    rep_str = ''
                gold_file_name = gold_base + target + '_' + size + rep_str + '.word'
                s = open(segfile, 'r', encoding='utf-8')
                sn = len(s.readlines())
                g = open(gold_file_name, 'r', encoding='utf-8')
                gn = len(g.readlines())
                if gn == sn:
                    print('.', end='', flush=True)
                    tok, typ, avg_tok_len, avg_sent_len = get_token_type_avgs(segfile, nb_truncate, strip_accents)
                    x, wp, wr, wf, bp, br, bf, lp, lr, lf = get_prf_metrics(gold_file_name, segfile, nb_truncate, strip_accents)

                    # Build the table
                    if gran == '':
                        gran = 'NA'
                    run_str = run.split('/')[-1]
                    table_line = [run_str, target, size, rep, gran,
                                  epoch, batch, drop, lrate, hidden, out,
                                  gru, upfirst, bias, aux, wait, timeid,
                                  "{0:.2f}".format(float(x) * 100),
                                  "{0:.2f}".format(float(wp) * 100),
                                  "{0:.2f}".format(float(wr) * 100),
                                  "{0:.2f}".format(float(wf) * 100),
                                  "{0:.2f}".format(float(bp) * 100),
                                  "{0:.2f}".format(float(br) * 100),
                                  "{0:.2f}".format(float(bf) * 100),
                                  "{0:.2f}".format(float(lp) * 100),
                                  "{0:.2f}".format(float(lr) * 100),
                                  "{0:.2f}".format(float(lf) * 100),
                                  "{:,}".format(tok),
                                  "{:,}".format(typ),
                                  "{0:.2f}".format(float(avg_tok_len)),
                                  "{0:.2f}".format(float(avg_sent_len))
                                  ]
                    table += [table_line]
                    count += 1
                else:
                    print("\nmissed " + segfile)
                    nmissed += 1
                # break

    headers = np.array(table[0])
    sorted_table = np.array(sorted(table[1:], key=lambda x: float(x[20]), reverse=True))

    if len(sorted_table) > 0:
        final_table = np.vstack((headers, sorted_table))
    else:
        sys.exit("No results could be computed.")

    if is_verbose:
        print('\n(' + str(count) + ' file(s) in table.)')
        ttok, ttyp, tavg_tok_len, tavg_sent_len = get_token_type_avgs(gold_file_name, nb_truncate, strip_accents)
        print(f"\nTrue token number: {ttok:,} / True type number: {ttyp:,} / True average token len: {tavg_tok_len:.2f} / True average sentence len: {tavg_sent_len:.2f}")

    print("\n\n")
    if output_format == 'plain':
        print(tabulate.tabulate(final_table,tablefmt="plain"))
    elif output_format == 'org':
        print(tabulate.tabulate(final_table,tablefmt="orgtbl"))
    elif output_format == 'latex':
        print(tabulate.tabulate(final_table,tablefmt="latex_booktabs"))
    else:
        sys.exit("Unknown output format.")

    print(f"\n({nmissed} file(s) were missed because of a different numbers of lines in reference and segmentation.)")

    # Store results
    if pickle_file:
        with open(pickle_file, mode="wb") as o:
            pickle.dump(final_table[1:], o)

def main():
    parser = argparse.ArgumentParser(usage='to format results')
    parser.add_argument('--runs', type=str, nargs="*", help='Run paths, e.g. "./run_01/ ./run_02/"')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--trunc', type=int, default = 0, help='Truncate to a certain nb. of lines for test.')
    parser.add_argument('--strip', action='store_true', help='Strip accents for eval.')
    parser.add_argument('--fmt', type=str, help='Output format.')
    # parser.add_argument('-a', '--avg_results', action='store_true', default=False)
    parser.add_argument('-f', '--pickle_file', default=False, type=str, help='i/o pickle file')
    parser.add_argument('--score', action='store_true', help='Scoring mode.')
    parser.add_argument('--gold', type=str, help='Reference segmentation.')
    parser.add_argument('--segmented', type=str, help='Segmentation to evaluate against gold.')

    args = parser.parse_args()
    if args.score:
        x, wp, wr, wf, bp, br, bf, lp, lr, lf = get_prf_metrics(args.gold, args.segmented)
        results = f"X {(x*100):.2f} WP {(wp*100):.2f} WR {(wr*100):.2f} WF {(wf*100):.2f}  BP {(bp*100):.2f} BR {(br*100):.2f} BF {(bf*100):.2f} LP {(lp*100):.2f} LR {(lr*100):.2f} LF {(lf*100):.2f}"
        print(results)
    else:
        process_results(args.runs, args.trunc, args.strip, args.verbose, args.fmt, args.pickle_file)

    # if args.avg_results and args.pickle_file:
    #     print_avg_results(args.pickle_file)

if __name__ == '__main__':
    sys.exit(main())
