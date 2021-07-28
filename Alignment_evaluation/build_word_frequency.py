
import numpy
import argparse, io
from collections import OrderedDict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('corpus')
args = parser.parse_args()

corpus = open(args.corpus, 'r', encoding='utf-8')
freq_file = open(args.corpus+'.dict.freq','w+')

word_freqs = OrderedDict()

for line in corpus:
    words_in = line.strip().split(' ')
    for w in words_in:
        if w not in word_freqs:
            word_freqs[w] = 0
        word_freqs[w] += 1

for word in word_freqs:
    freq_file.write(word +' '+ str(word_freqs[word]) + '\n')
