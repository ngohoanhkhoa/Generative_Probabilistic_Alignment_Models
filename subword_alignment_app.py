import streamlit as st

import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

palettes = list(mcolors.TABLEAU_COLORS.values())
figScale = 1.5
# ===============================================================================
# ===============================================================================
st.title('Optimizing Word Alignments with Better Subword Tokenization')
st.markdown('''by Anh Khoa Ngo Ho''')
st.markdown('''*Proceedings of Machine Translation Summit XVIII: Research Track, Association for Machine Translation in the Americas*''')
st.markdown('''Word alignments identify translational correspondences between words in a parallel sentence pair and are used, for example, to train statistical machine translation and learn bilingual dictionaries or to perform the quality estimation. Subword tokenization has become a standard preprocessing step for a large number of applications and notably for state-of-the-art open vocabulary machine translation systems. In this paper and we thoroughly study how this preprocessing step interacts with the word alignment task and propose several tokenization strategies to obtain well-segmented parallel corpora. Using these new techniques and we were able to improve baseline word-based alignment models for six language pairs (English-French, English-German, English-Romanian, English-Czech, English-Japanese, English-Vietnamese).''')
st.markdown(
    '''The full code is found in [Github](https://github.com/ngohoanhkhoa/Generative_Probabilistic_Alignment_Models). All descriptions and other informations are shown in [Paper](https://aclanthology.org/2021.mtsummit-research.21/) and [Theses.fr](https://www.theses.fr/2021UPASG014). Note that all alignment results are computed by [Fastalign](https://github.com/clab/fast_align) (with default parameters) and subword tokenization is made by BPE method implemented in [SentencePiece](https://github.com/google/sentencepiece).''')

corpora = ('English->French', 'French->English',
           'English->German', 'German->English',
           'English->Romanian', 'Romanian->English',
           'English->Czech', 'Czech->English',
           'English->Japanese', 'Japanese->English',
           'English->Vietnamese', 'Vietnamese->English')

corpus = st.selectbox(label="Corpus:", options=corpora)

if corpus == 'English->French':
    corpus_ = 'en-fr'
    st.markdown('''Number of sentence pairs: 447''')
    st.markdown(
        '''Source: the 2003 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
    file_directory = 'subword/subword_stats_en_fr'
    vocabulary_char_size_src = 111
    vocabulary_char_size_tgt = 115
    vocabulary_word_size_src = 106322
    vocabulary_word_size_tgt = 112734
elif corpus == 'French->English':
    corpus_ = 'fr-en'
    st.markdown('''Number of sentence pairs: 447''')
    st.markdown(
        '''Source: the 2003 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
    file_directory = 'subword/subword_stats_fr_en'
    vocabulary_char_size_src = 115
    vocabulary_char_size_tgt = 111
    vocabulary_word_size_src = 112734
    vocabulary_word_size_tgt = 106322

elif corpus == 'English->German':
    corpus_ = 'en-ge'
    st.markdown('''Number of sentence pairs: 509''')
    st.markdown(
        '''Source: [Europarl](http://www-i6.informatik.rwth-aachen.de/goldAlignment/)''')
    file_directory = 'subword/subword_stats_en_ge'
    vocabulary_char_size_src = 218
    vocabulary_char_size_tgt = 235
    vocabulary_word_size_src = 96898
    vocabulary_word_size_tgt = 311582
elif corpus == 'German->English':
    corpus_ = 'ge-en'
    st.markdown('''Number of sentence pairs: 509''')
    st.markdown(
        '''Source: [Europarl](http://www-i6.informatik.rwth-aachen.de/goldAlignment/)''')
    file_directory = 'subword/subword_stats_ge_en'
    vocabulary_char_size_src = 235
    vocabulary_char_size_tgt = 218
    vocabulary_word_size_src = 311582
    vocabulary_word_size_tgt = 96898

elif corpus == 'English->Romanian':
    corpus_ = 'en-ro'
    st.markdown('''Number of sentence pairs: 246''')
    st.markdown(
        '''Source: the 2015 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
    file_directory = 'subword/subword_stats_en_ro'
    vocabulary_char_size_src = 124
    vocabulary_char_size_tgt = 131
    vocabulary_word_size_src = 74279
    vocabulary_word_size_tgt = 115567
elif corpus == 'Romanian->English':
    corpus_ = 'ro-en'
    st.markdown('''Number of sentence pairs: 246''')
    st.markdown(
        '''Source: the 2015 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
    file_directory = 'subword/subword_stats_ro_en'
    vocabulary_char_size_src = 131
    vocabulary_char_size_tgt = 124
    vocabulary_word_size_src = 115567
    vocabulary_word_size_tgt = 74279

elif corpus == 'English->Czech':
    corpus_ = 'en-cz'
    st.markdown('''Number of sentence pairs: 2 501''')
    st.markdown(
        '''Source: [Marecek, 2016], [url](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804)''')
    file_directory = 'subword/subword_stats_en_cz'
    vocabulary_char_size_src = 246
    vocabulary_char_size_tgt = 157
    vocabulary_word_size_src = 62877
    vocabulary_word_size_tgt = 147188
elif corpus == 'Czech->English':
    corpus_ = 'cz-en'
    st.markdown('''Number of sentence pairs: 2 501''')
    st.markdown(
        '''Source: [Marecek, 2016], [url](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804)''')
    file_directory = 'subword/subword_stats_cz_en'
    vocabulary_char_size_src = 157
    vocabulary_char_size_tgt = 246
    vocabulary_word_size_src = 147188
    vocabulary_word_size_tgt = 62877

elif corpus == 'English->Japanese':
    corpus_ = 'en-ja'
    st.markdown('''Number of sentence pairs: 1 235''')
    st.markdown(
        '''Source: [KFTT](http://www.phontron.com/kftt/#alignments)''')
    file_directory = 'subword/subword_stats_en_ja'
    vocabulary_char_size_src = 2920
    vocabulary_char_size_tgt = 5766
    vocabulary_word_size_src = 156107
    vocabulary_word_size_tgt = 126246
elif corpus == 'Japanese->English':
    corpus_ = 'ja-en'
    st.markdown('''Number of sentence pairs: 1 235''')
    st.markdown(
        '''Source: [KFTT](http://www.phontron.com/kftt/#alignments)''')
    file_directory = 'subword/subword_stats_ja_en'
    vocabulary_char_size_src = 5766
    vocabulary_char_size_tgt = 2920
    vocabulary_word_size_src = 126246
    vocabulary_word_size_tgt = 156107

elif corpus == 'English->Vietnamese':
    corpus_ = 'en-vi'
    st.markdown('''Number of sentence pairs: 3 447''')
    st.markdown(
        '''Source: [EVBCorpus](https://code.google.com/archive/p/evbcorpus/)''')
    file_directory = 'subword/subword_stats_en_vi'
    vocabulary_char_size_src = 133
    vocabulary_char_size_tgt = 171
    vocabulary_word_size_src = 42544
    vocabulary_word_size_tgt = 19853
elif corpus == 'Vietnamese->English':
    corpus_ = 'vi-en'
    st.markdown('''Number of sentence pairs: 3 447''')
    st.markdown(
        '''Source: [EVBCorpus](https://code.google.com/archive/p/evbcorpus/)''')
    file_directory = 'subword/subword_stats_vi_en'
    vocabulary_char_size_src = 171
    vocabulary_char_size_tgt = 133
    vocabulary_word_size_src = 19853
    vocabulary_word_size_tgt = 42544

# =============================
subword_stats = json.load(open(file_directory, 'r'))


def get_vocab_pair(stats):
    vocs_src = []
    vocs_tgt = []

    for vocab_pair in stats:
        if vocab_pair != 'reference':
            voc_src = vocab_pair.split('-')[0]
            voc_tgt = vocab_pair.split('-')[1]

            if voc_src not in vocs_src + ['char', 'word']:
                vocs_src.append(voc_src)
            if voc_tgt not in vocs_tgt + ['char', 'word']:
                vocs_tgt.append(voc_tgt)

    vocs_src = [int(v) for v in vocs_src]
    vocs_tgt = [int(v) for v in vocs_tgt]

    vocs_src.sort()
    vocs_tgt.sort()

    vocs_src = ['char'] + [str(v) for v in vocs_src] + ['word']
    vocs_tgt = ['char'] + [str(v) for v in vocs_tgt] + ['word']

    return vocs_src, vocs_tgt


def show_matrix(stats, query="aer", query_name="AER"):
    vocs_src, vocs_tgt = get_vocab_pair(stats)

    len_src = len(vocs_src)
    len_tgt = len(vocs_tgt)

    matrix = np.zeros((len_src, len_tgt))

    for idx_src, voc_src in enumerate(vocs_src):
        for idx_tgt, voc_tgt in enumerate(vocs_tgt):
            voc_pair = voc_src+'-'+voc_tgt
            matrix[idx_src, idx_tgt] = stats[voc_pair][query]

    fig = px.imshow(matrix,
                    labels=dict(x="Target vocabulary size",
                                y="Source vocabulary size", color=query_name),
                    x=vocs_tgt,
                    y=vocs_src)

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=5, r=5, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)


def count_length(len_list):
    len_dict = {}
    for len in len_list:
        if len not in len_dict:
            len_dict[len] = 1
        else:
            len_dict[len] += 1

    return len_dict


def show_length(stats, vocab_size, source=True):
    if source:
        len_word_col = 'len_word_source'
        len_char_col = 'len_char_source'
    else:
        len_word_col = 'len_word_target'
        len_char_col = 'len_char_target'

    len_freq_list = []
    lens_word = count_length(stats['word-word'][len_word_col])
    for len in lens_word:
        len_freq_list.append({'Tokenization level': 'word',
                              'Sequence length': len, 'Number of sequences': lens_word[len]})
    lens_char = count_length(stats['char-char'][len_char_col])
    for len in lens_char:
        len_freq_list.append({'Tokenization level': 'character',
                              'Sequence length': len, 'Number of sequences': lens_char[len]})
    if source:
        lens_subword = count_length(stats[vocab_size+'-char'][len_word_col])
    else:
        lens_subword = count_length(stats['char-'+vocab_size][len_word_col])

    for len in lens_subword:
        len_freq_list.append({'Tokenization level': 'subword',
                              'Sequence length': len, 'Number of sequences': lens_subword[len]})

    len_freq_df = pd.DataFrame(len_freq_list)

    fig = px.histogram(len_freq_df, x="Sequence length", y="Number of sequences", color="Tokenization level",
                       marginal="violin")

    fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


def show_len_differences(stats, method='Mean', score='AER', absolute=True):
    lens_diff = []
    for vocab_pair in stats:
        if vocab_pair != 'reference':
            len_diff = np.array(stats[vocab_pair]['len_word_source']) - \
                                np.array(stats[vocab_pair]['len_word_target'])
            if absolute:
                len_diff = np.abs(len_diff)

            len_diff_mean = np.mean(len_diff)
            len_diff_sum = np.sum(len_diff)
            len_diff_max = np.max(len_diff)
            len_diff_min = np.min(len_diff)

            lens_diff.append({'Vocabulary pair': vocab_pair,
                              'Length difference - Mean': len_diff_mean,
                              'Length difference - Sum': len_diff_sum,
                              'Length difference - Max': len_diff_max,
                              'Length difference - Min': len_diff_min,
                              'AER': stats[vocab_pair]['aer'],
                              'F-score': stats[vocab_pair]['f1'],
                              'Precision': stats[vocab_pair]['precision'],
                              'Recall': stats[vocab_pair]['recall']})

    lens_diff_df = pd.DataFrame(lens_diff)

    fig = px.scatter(lens_diff_df, x="Length difference - " + method, y=score, color='Vocabulary pair',
                     hover_data=['Vocabulary pair',
                                 'Length difference - Mean',
                                 'Length difference - Sum',
                                 'Length difference - Max',
                                 'Length difference - Min',
                                 'AER', 'F-score', 'Precision', 'Recall'])

    st.plotly_chart(fig, use_container_width=True)


def show_len_rate(stats, method='Subword fertility', score='AER'):
    if score == 'AER':
        score_ = 'aer'
    elif score == 'F-score':
        score_ = 'f1'
    elif score == 'Precision':
        score_ = 'precision'
    else:
        score_ = 'recall'

    vocs_src, vocs_tgt = get_vocab_pair(stats)

    vocs_src_len_rate = []
    for v in vocs_src:
        if method == 'Subword fertility':
            value = np.sum(stats[v+'-word']['len_word_source']) / \
                           np.sum(stats['reference']['len_word_source'])
        elif method == 'Absolute compression rate':
            value = np.sum(stats[v+'-word']['len_word_source']) / \
                           np.sum(stats['reference']['len_char_source'])
        elif method == 'Relative compression rate':
            value_1 = np.sum(stats[v+'-word']['len_word_source']) / \
                           np.sum(stats['reference']['len_char_source'])
            if v == 'char':
                v_ = vocabulary_char_size_src
            elif v == 'word':
                v_ = vocabulary_word_size_src
            else:
                v_ = int(v)

            value_2 = vocabulary_char_size_src/int(v_)
            value = value_1 * value_2
        else:
            value = 0

        vocs_src_len_rate.append(v + '->' + str(np.round(value, 2)))

    vocs_tgt_len_rate = []
    for v in vocs_tgt:
        if method == 'Subword fertility':
            value = np.sum(stats['word-'+v]['len_word_target']) / \
                           np.sum(stats['reference']['len_word_target'])
        elif method == 'Absolute compression rate':
            value = np.sum(stats['word-'+v]['len_word_target']) / \
                           np.sum(stats['reference']['len_char_target'])
        elif method == 'Relative compression rate':
            value_1 = np.sum(stats['word-'+v]['len_word_target']) / \
                           np.sum(stats['reference']['len_char_target'])

            if v == 'char':
                v_ = vocabulary_char_size_src
            elif v == 'word':
                v_ = vocabulary_word_size_src
            else:
                v_ = int(v)

            value_2 = vocabulary_char_size_tgt/int(v_)
            value = value_1 * value_2
        else:
            value = 0

        vocs_tgt_len_rate.append(v + '->' + str(np.round(value, 2)))

    len_src = len(vocs_src)
    len_tgt = len(vocs_tgt)

    matrix = np.zeros((len_src, len_tgt))

    for idx_src, voc_src in enumerate(vocs_src):
        for idx_tgt, voc_tgt in enumerate(vocs_tgt):
            voc_pair = voc_src+'-'+voc_tgt
            matrix[idx_src, idx_tgt] = stats[voc_pair][score_]

    fig = px.imshow(matrix,
                    labels=dict(x="Target voc. size -> " + method,
                                y="Source voc. size -> " + method, color=score),
                    x=vocs_tgt_len_rate,
                    y=vocs_src_len_rate)

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=5, r=5, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)


def show_length_ngram(stats, vocab_size, source=True):
    if source:
        n_gram_col = 'n_gram_src'
    else:
        n_gram_col = 'n_gram_tgt'

    ngrams_list = []
    ngrams_word = stats['word-word'][n_gram_col]
    for ngram in ngrams_word:
        ngrams_list.append({'Tokenization level': 'word',
                            'Token length': int(ngram), 'Number of unique tokens': ngrams_word[ngram]})

    ngrams_char = stats['char-char'][n_gram_col]

    for ngram in ngrams_char:
        ngrams_list.append({'Tokenization level': 'character',
                            'Token length': int(ngram),
                            'Number of unique tokens': ngrams_char[ngram]})
    if source:
        ngrams = stats[vocab_size+'-char'][n_gram_col]
    else:
        ngrams = stats['char-'+vocab_size][n_gram_col]

    for ngram in ngrams:
        ngrams_list.append({'Tokenization level': 'subword',
                            'Token length': int(ngram), 'Number of unique tokens': ngrams[ngram]})

    ngram_df = pd.DataFrame(ngrams_list)

    fig = px.histogram(ngram_df, x="Token length", y="Number of unique tokens",
                       color="Tokenization level", marginal="violin", nbins=20)

    fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


def show_performance_ngram(stats, score='AER', threshold=0.5):
    if score == 'AER':
        score_ = 'aer'
    elif score == 'F-score':
        score_ = 'f1'
    elif score == 'Precision':
        score_ = 'precision'
    else:
        score_ = 'recall'

    ngram_src_list = []
    ngram_tgt_list = []
    for vocab_pair in stats:
        if vocab_pair != 'reference':
            ngrams_src = stats[vocab_pair]['n_gram_src']
            ngrams_tgt = stats[vocab_pair]['n_gram_tgt']
            for ngram_src in ngrams_src:
                if ngram_src not in ngram_src_list:
                    ngram_src_list.append(ngram_src)
            for ngram_tgt in ngrams_tgt:
                if ngram_tgt not in ngram_tgt_list:
                    ngram_tgt_list.append(ngram_tgt)

    ngram_src_max = np.max([int(v) for v in ngram_src_list])
    ngram_tgt_max = np.max([int(v) for v in ngram_tgt_list])

    matrix_performance_sum = np.zeros((ngram_src_max, ngram_tgt_max))
    matrix_performance_num = np.zeros((ngram_src_max, ngram_tgt_max))

    for vocab_pair in stats:
        if vocab_pair != 'reference':
            ngrams_src = stats[vocab_pair]['n_gram_src']
            ngrams_tgt = stats[vocab_pair]['n_gram_tgt']
            for ngram_src in ngrams_src:
                for ngram_tgt in ngrams_tgt:
                    ngram_src_rate = ngrams_src[ngram_src] / \
                        np.sum(list(ngrams_src.values()))
                    ngram_tgt_rate = ngrams_tgt[ngram_tgt] / \
                        np.sum(list(ngrams_tgt.values()))
                    if ngram_src_rate >= threshold and ngram_tgt_rate >= threshold:
                        matrix_performance_sum[int(
                            ngram_src)-1, int(ngram_tgt)-1] += stats[vocab_pair][score_]
                        matrix_performance_num[int(
                            ngram_src)-1, int(ngram_tgt)-1] += 1

    row_idx_max = 0
    for row_idx, row in enumerate(matrix_performance_num[:, 0]):
        if row != 0 and row_idx > row_idx_max:
            row_idx_max = row_idx

    col_idx_max = 0
    for col_idx, col in enumerate(matrix_performance_num[0, :]):
        if col != 0 and col_idx > col_idx_max:
            col_idx_max = col_idx

    matrix_performance = matrix_performance_sum[:row_idx_max+1,
                                                :col_idx_max+1]/matrix_performance_num[:row_idx_max+1, :col_idx_max+1]

    y_name = [str(v+1) for v in range(np.shape(matrix_performance)[0])]
    x_name = [str(v+1) for v in range(np.shape(matrix_performance)[1])]

    fig = px.imshow(matrix_performance,
                    labels=dict(x="Target token length",
                                y="Source token length", color=score),
                    x=x_name, y=y_name)

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=5, r=5, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)


with st.expander("Aligned/unaligned words"):
    st.header("Aligned/unaligned words")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source words")
        st.markdown("**Number of words**: %s words" %
                    (subword_stats['reference']['num_word_source']))
        st.markdown("**Number of aligned words**: %s words" %
                    (subword_stats['reference']['num_source2notnull_ref']))

        type_source_word = st.radio(
            "Type", ('Aligned source words', 'Unaligned source words'))

        if type_source_word == 'Aligned source words':
            show_matrix(subword_stats, query="num_source2notnull_pred",
                        query_name="Number of words")

            st.markdown("""---""")
            st.markdown("**Correct aligned words**")
            show_matrix(subword_stats, query="num_source2null_pred_tn",
                        query_name="Number of words")
        else:
            show_matrix(subword_stats, query="num_source2null_pred",
                        query_name="Number of words")
            st.markdown("""---""")
            st.markdown("**Correct unaligned words**")
            show_matrix(subword_stats, query="num_source2null_pred_tp",
                        query_name="Number of words")

    with col2:
        st.subheader("Target words")
        st.markdown("**Number of words**: %s words" %
                    (subword_stats['reference']['num_word_target']))
        st.markdown("**Number of aligned words**: %s words" %
                    (subword_stats['reference']['num_target2notnull_ref']))

        type_target_word = st.radio(
            "Type", ('Aligned target words', 'Unaligned target words'))

        if type_target_word == 'Aligned target words':
            show_matrix(subword_stats, query="num_target2notnull_pred",
                        query_name="Number of words")
            st.markdown("""---""")
            st.markdown("**Correct aligned words**")
            show_matrix(subword_stats, query="num_target2null_pred_tn",
                        query_name="Number of words")
        else:
            show_matrix(subword_stats, query="num_target2null_pred",
                        query_name="Number of words")
            st.markdown("""---""")
            st.markdown("**Correct unaligned words**")
            show_matrix(subword_stats, query="num_target2null_pred_tp",
                        query_name="Number of words")

with st.expander("Alignment links"):
    st.header("Alignment links")
    col_link_1, col_link_2 = st.columns(2)
    with col_link_1:
        st.write("**Number of All possible links**: "
                 + str(subword_stats['reference']["total_num_link"]))
        st.write("**Number of Sure links**: "
                 + str(subword_stats['reference']["num_sure"]))
        st.write("**Number of Fuzzy links**: "
                 + str(subword_stats['reference']["num_fuzzy"]))
        st.write("**Number of Non-existing links**: "
                 + str(subword_stats['reference']["num_no_ref"]) + " (including null links)")
        st.write("**Number of Null links**: "
                 + str(subword_stats['reference']["num_no_ref_null"]))
    with col_link_2:
        fig2, ax2 = plt.subplots()

        x = ["Sure links", "Fuzzy links", "Null links"]
        y = [subword_stats['reference']["num_sure"], subword_stats['reference']
             ["num_fuzzy"],  subword_stats['reference']["num_no_ref_null"]]
        sns.barplot(x=x, y=y, ax=ax2, palette=palettes)

        ax2.set(ylim=(0, np.max(y)*figScale))
        ax2.set(xlabel="All possible links I*J: " + str(subword_stats['reference']["total_num_link"])
                + "\n Non-existing links: " + str(subword_stats['reference']["num_no_ref"]) + ", including Null links", ylabel='Number of links')
        for index, value in enumerate(y):
            ax2.text(index, value, str(value), color='black',
                     ha="center", verticalalignment="bottom")

        fig2.tight_layout(pad=3.)
        fig2.tight_layout()

        st.write(fig2)

    st.subheader("Number of links")
    type_link = st.selectbox(label="Type:", options=(
        "Alignment links", "Non-existing links", "Null links"))
    if type_link == "Alignment links":
        show_matrix(subword_stats, query="num_align_pred",
                    query_name="Number of links")
    elif type_link == "Non-existing links":
        show_matrix(subword_stats, query="num_no_pred",
                    query_name="Number of links")
    else:
        show_matrix(subword_stats, query="num_no_pred_null",
                    query_name="Number of links")

    st.subheader("Number of correct links")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Number of Correct alignment links**")
        show_matrix(subword_stats, query="num_true_align_tp",
                    query_name="Number of links")
    with col4:
        st.write("**Number of Correct Non-existing links**")
        show_matrix(subword_stats, query="num_true_no_tn",
                    query_name="Number of links")

    st.subheader("Scores")
    col5, col6 = st.columns(2)
    with col5:
        st.write("**Alignment Error Rate**")
        show_matrix(subword_stats, query="aer",
                    query_name="AER")

        st.write("**Precision**")
        show_matrix(subword_stats, query="precision",
                    query_name="Precision")

    with col6:
        st.write("**F-score**")
        show_matrix(subword_stats, query="f1",
                    query_name="F-score")

        st.write("**Recall**")
        show_matrix(subword_stats, query="recall",
                    query_name="Recall")

with st.expander("Fertility"):
    st.header("Fertility")
    col_fer_1, col_fer_2 = st.columns(2)
    with col_fer_1:
        st.write("Number of One-to-one links: "
                 + str(subword_stats['reference']["num_align_ref_one2one"])+"-"+str(subword_stats['reference']["num_align_ref_one2one"]))
        st.write("Number of One-to-many links: " + str(subword_stats['reference']["num_align_ref_one2many_source"])+"-"+str(
            subword_stats['reference']["num_align_ref_one2many_target"]))
        st.write("Number of Many-to-one links: " + str(subword_stats['reference']["num_align_ref_many2one_source"])+"-"+str(
            subword_stats['reference']["num_align_ref_many2one_target"]))
        st.write("Number of Many-to-many links: " + str(subword_stats['reference']["num_align_ref_many2many_source"])+"-"+str(
            subword_stats['reference']["num_align_ref_many2many_target"]))
    with col_fer_2:
        fig3, ax3 = plt.subplots()

        x = ["One2One", "One2Many", "Many2One", "Many2Many", "%"]
        y = [subword_stats['reference']["num_align_ref_one2one"], subword_stats['reference']["num_align_ref_one2many_target"],
             subword_stats['reference']["num_align_ref_many2one_source"], subword_stats['reference']["num_align_ref_many2many"]]
        y_text = [str(subword_stats['reference']["num_align_ref_one2one"])+"-"+str(subword_stats['reference']["num_align_ref_one2one"]),
                  str(subword_stats['reference']
                      ["num_align_ref_one2many_source"])
                  + "-"+str(subword_stats['reference']
                            ["num_align_ref_one2many_target"]),
                  str(subword_stats['reference']
                      ["num_align_ref_many2one_source"])
                  + "-"+str(subword_stats['reference']
                            ["num_align_ref_many2one_target"]),
                  str(subword_stats['reference']
                      ["num_align_ref_many2many_source"])
                  + "-"+str(subword_stats['reference']
                            ["num_align_ref_many2many_target"])
                  ]
        y_percent = np.array(y)*100/subword_stats['reference']["num_align_ref"]

        one2one_percent = (
            100 * subword_stats['reference']["num_align_ref_one2one"]/subword_stats['reference']["num_align_ref"])
        one2many_percent = one2one_percent + \
            (100 * subword_stats['reference']["num_align_ref_one2many_target"]
             / subword_stats['reference']["num_align_ref"])
        many2one_percent = one2many_percent + \
            (100 * subword_stats['reference']["num_align_ref_many2one_source"]
             / subword_stats['reference']["num_align_ref"])
        many2many_percent = many2one_percent + \
            (100 * subword_stats['reference']["num_align_ref_many2many"]
             / subword_stats['reference']["num_align_ref"])

        ax3.set(ylim=(0, np.max(y)*figScale))
        ax3.set(xlabel="Alignment links: Source-Target",
                ylabel='Number of links')
        ax3.bar(x=x, height=[subword_stats['reference']["num_align_ref_one2one"],
                             0, 0, 0, 0], color=palettes[0])
        ax3.bar(x=x, height=[
                0, subword_stats['reference']["num_align_ref_one2many_target"], 0, 0, 0], color=palettes[1])
        ax3.bar(x=x, height=[
                0, 0, subword_stats['reference']["num_align_ref_many2one_source"], 0, 0], color=palettes[2])
        ax3.bar(x=x, height=[
                0, 0, 0, subword_stats['reference']["num_align_ref_many2many"], 0], color=palettes[3])

        for index, (height, value) in enumerate(zip(y, y_text)):
            ax3.text(index, height, value, color='black',
                     ha="center", verticalalignment="bottom")

        ax3X = ax3.twinx()
        ax3X.set(ylabel='Percentage')
        ax3X.set(ylim=(0, 110))
        ax3X.bar(x=x, height=[0, 0, 0, 0,
                              many2many_percent], color=palettes[3])
        ax3X.bar(x=x, height=[0, 0, 0, 0, many2one_percent], color=palettes[2])
        ax3X.bar(x=x, height=[0, 0, 0, 0, one2many_percent], color=palettes[1])
        ax3X.bar(x=x, height=[0, 0, 0, 0, one2one_percent], color=palettes[0])

        y_bar = [one2one_percent, one2many_percent,
                 many2one_percent, many2many_percent]

        for index, value in zip(y_bar, y_percent):
            if value != 0:
                ax3X.text(4, index, str(np.round(value, 1)) + "%",
                          color='black', ha="center", verticalalignment="bottom")

        ax3X.legend(("Many2Many", "Many2One", "One2Many", "One2One"), loc='upper center', bbox_to_anchor=(0.5, 1.15),
                    fancybox=True, ncol=4)
        st.write(fig3)

    st.subheader("Number of links")
    type_fer_link = st.selectbox(label="Type:", options=(
        "One-to-one links", "One-to-many links", "Many-to-one links", "Many-to-many links"))
    col_fer_3, col_fer_4 = st.columns(2)
    if type_fer_link == "One-to-one links":
        with col_fer_3:
            st.markdown("**One-to-one links**")
            show_matrix(subword_stats, query="num_align_pred_one2one",
                        query_name="Number of links")

        with col_fer_4:
            st.markdown("**Correct One-to-one links**")
            show_matrix(subword_stats, query="num_true_align_tp_one2one_pred",
                        query_name="Number of links")

    elif type_fer_link == "One-to-many links":
        with col_fer_3:
            st.markdown("**One-to-many links**")
            show_matrix(subword_stats, query="num_align_pred_one2many_target",
                        query_name="Number of links")
        with col_fer_4:
            st.markdown("**Correct One-to-many links**")
            show_matrix(subword_stats, query="num_true_align_tp_one2many_pred",
                        query_name="Number of links")

    elif type_fer_link == "Many-to-one links":
        with col_fer_3:
            st.markdown("**Many-to-one links**")
            show_matrix(subword_stats, query="num_align_pred_many2one_source",
                        query_name="Number of links")
        with col_fer_4:
            st.markdown("**Correct Many-to-one links**")
            show_matrix(subword_stats, query="num_true_align_tp_many2one_pred",
                        query_name="Number of links")
    else:
        with col_fer_3:
            st.markdown("**Many-to-many links**")
            show_matrix(subword_stats, query="num_align_pred_many2many",
                        query_name="Number of links")
        with col_fer_4:
            st.markdown("**Correct Many-to-many links**")
            show_matrix(subword_stats, query="num_true_align_tp_many2many_pred",
                        query_name="Number of links")

with st.expander("Sequence lengths"):
    st.header("Sequence lengths")
    vocs_src, vocs_tgt = get_vocab_pair(subword_stats)
    col_len_1, col_len_2 = st.columns(2)
    with col_len_1:
        st.subheader("Source")
        vocab_src_size = st.selectbox(
            label="Subword vocabulary size:", options=[v for v in vocs_src if v not in ['char', 'word']], key='vocab_src_size')

        show_length(subword_stats, vocab_src_size, source=True)
    with col_len_2:
        st.subheader("Target")
        vocab_tgt_size = st.selectbox(
            label="Subword vocabulary size:", options=[v for v in vocs_tgt if v not in ['char', 'word']], key='vocab_tgt_size')
        show_length(subword_stats, vocab_tgt_size, source=False)

    st.subheader(
        "Length differences between source and target sequences")

    absolute = st.radio("Value",
                        ('Absolute length difference', 'Non-absolute length difference'))

    col_len_3, col_len_4 = st.columns(2)
    with col_len_3:
        combination_method = st.selectbox(
            label="Combination of length differences for all sequences ", options=('Mean', 'Sum', 'Max', 'Min'))
    with col_len_4:
        score = st.selectbox(
            label="Score \n", options=('AER', 'F-score', 'Precision', 'Recall'))

    if absolute == 'Absolute length difference':
        show_len_differences(
            subword_stats, combination_method, score, absolute=True)
    else:
        show_len_differences(
            subword_stats, combination_method, score, absolute=False)

    st.subheader("Length rate")

    col_len_5, col_len_6 = st.columns(2)
    with col_len_5:
        len_rate_method = st.selectbox(
            label="Method:", options=('Subword fertility', 'Absolute compression rate', 'Relative compression rate'))
    with col_len_6:
        len_rate_score = st.selectbox(
            label="Score \n", options=('AER', 'F-score', 'Precision', 'Recall'), key='length_rate_score')
    if len_rate_method == 'Subword fertility':
        st.markdown(
            "**The subword fertility**: [Rust et al., 2021](https://arxiv.org/abs/2012.15613)")
    elif len_rate_method == 'Subword fertility':
        st.markdown(
            "**Absolute compression rate**: [Maronikolakis et al., 2021](https://arxiv.org/abs/2109.05772)")
    else:
        st.markdown(
            "**Relative compression rate**: [Maronikolakis et al., 2021](https://arxiv.org/abs/2109.05772)")
    show_len_rate(subword_stats, method=len_rate_method, score=len_rate_score)

with st.expander("Token lengths"):
    st.header("Token lengths")

    vocs_src, vocs_tgt = get_vocab_pair(subword_stats)
    col_ngram_1, col_ngram_2 = st.columns(2)
    with col_ngram_1:
        st.subheader("Source")
        vocab_src_size_n_gram = st.selectbox(
            label="Subword vocabulary size:", options=[v for v in vocs_src if v not in ['char', 'word']], key='vocab_src_size_n_gram')

        show_length_ngram(subword_stats, vocab_src_size, source=True)
    with col_ngram_2:
        st.subheader("Target")
        vocab_tgt_size_n_gram = st.selectbox(
            label="Subword vocabulary size:", options=[v for v in vocs_tgt if v not in ['char', 'word']], key='vocab_tgt_size_n_gram')

        show_length_ngram(subword_stats, vocab_tgt_size, source=False)

    st.subheader("Token length coverage")
    st.markdown(
        'We collect all unique tokens in a corpus, compute frequencies of the unique tokens of length n and average scores of the vocabulary pairs that the frequencies(for source and target language) are higher than a threshold. [Novotný et al., 2021](https://arxiv.org/abs/2102.02585)')

    col_ngram_5, col_ngram_6 = st.columns(2)
    with col_ngram_5:
        threshold = st.slider('Threshold of frequency', 0.0, 1.0, 0.05)

    with col_ngram_6:
        ngram_score = st.selectbox(
            label="Score \n", options=('AER', 'F-score', 'Precision', 'Recall'), key='ngram_score_score')

    show_performance_ngram(
        subword_stats, score=ngram_score, threshold=threshold)
