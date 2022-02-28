import streamlit as st

import json
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors



def show_graphs_reference(result, figScale=1.2):

    palettes = list(mcolors.TABLEAU_COLORS.values())
    #-----------------------------------------------------------------
    fig1, ax1 = plt.subplots()

    x = ["Source I="+ str(result["num_word_source"]), "Target J="+ str(result["num_word_target"])]
    y = [result["num_word_source"], result["num_word_target"] ]
    y_aligned = [result["num_source2notnull_ref"], result["num_target2notnull_ref"] ]
    y_notAligned = [result["num_source2null_ref"], result["num_target2null_ref"] ]

    ax1.bar(x=x, height=np.array(y_aligned)+np.array(y_notAligned), color=palettes[0])
    ax1.bar(x=x, height=y_aligned, color=palettes[1])
    ax1.set(xlabel="Aligned/Unaligned words", ylabel='Number of words')
    ax1.set(ylim=(0, np.max(y)*figScale))
    for index, (v1, v2) in enumerate(zip(y_aligned,y_notAligned)):
        if v1 != 0:
            ax1.text(index, v1, str(v1), color='black', ha="center", verticalalignment="bottom")
        if v2 != 0:
            ax1.text(index, v1+v2, str(v2), color='black', ha="center", verticalalignment="bottom")

    ax1.legend(("Unaligned", "Aligned"),loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, ncol=2)

    fig1.tight_layout(pad=3.)
    fig1.tight_layout()

    #-----------------------------------------------------------------
    fig2, ax2 = plt.subplots()

    x = ["Sure links", "Fuzzy links", "Null links"]
    y = [result["num_sure"], result["num_fuzzy"],  result["num_no_ref_null"]]
    sns.barplot(x=x, y=y, ax=ax2, palette=palettes)

    ax2.set(ylim=(0, np.max(y)*figScale))
    ax2.set(xlabel="All possible links I*J: " + str(result["total_num_link"]) + \
                "\n Non-existing links: " + str(result["num_no_ref"]) + ", including Null links", ylabel='Number of links')
    for index, value in enumerate(y):
        ax2.text(index, value, str(value), color='black', ha="center", verticalalignment="bottom")

    fig2.tight_layout(pad=3.)
    fig2.tight_layout()
    #-----------------------------------------------------------------
    fig3, ax3 = plt.subplots()

    x = ["One2One", "One2Many", "Many2One", "Many2Many", "%"]
    y = [result["num_align_ref_one2one"], result["num_align_ref_one2many_target"],
        result["num_align_ref_many2one_source"], result["num_align_ref_many2many"]]
    y_text = [str(result["num_align_ref_one2one"])+"-"+str(result["num_align_ref_one2one"]),
            str(result["num_align_ref_one2many_source"])+"-"+str(result["num_align_ref_one2many_target"]),
            str(result["num_align_ref_many2one_source"])+"-"+str(result["num_align_ref_many2one_target"]),
            str(result["num_align_ref_many2many_source"])+"-"+str(result["num_align_ref_many2many_target"])
            ]
    y_percent = np.array(y)*100/result["num_align_ref"]

    one2one_percent = (100 * result["num_align_ref_one2one"]/result["num_align_ref"])
    one2many_percent = one2one_percent + (100 * result["num_align_ref_one2many_target"]/result["num_align_ref"])
    many2one_percent = one2many_percent + (100 * result["num_align_ref_many2one_source"]/result["num_align_ref"])
    many2many_percent = many2one_percent + (100 * result["num_align_ref_many2many"]/result["num_align_ref"])

    ax3.set(ylim=(0, np.max(y)*figScale))
    ax3.set(xlabel="Alignment links: Source-Target", ylabel='Number of links')
    ax3.bar(x=x, height=[result["num_align_ref_one2one"], 0, 0, 0, 0], color=palettes[0])
    ax3.bar(x=x, height=[0, result["num_align_ref_one2many_target"], 0, 0, 0], color=palettes[1])
    ax3.bar(x=x, height=[0, 0, result["num_align_ref_many2one_source"], 0, 0], color=palettes[2])
    ax3.bar(x=x, height=[0, 0, 0, result["num_align_ref_many2many"], 0], color=palettes[3])

    for index, (height, value) in enumerate(zip(y,y_text)):
        ax3.text(index, height, value, color='black', ha="center", verticalalignment="bottom")

    ax3X = ax3.twinx()
    ax3X.set(ylabel='Percentage')
    ax3X.set(ylim=(0, 110))
    ax3X.bar(x=x, height=[0, 0, 0, 0, many2many_percent], color=palettes[3])
    ax3X.bar(x=x, height=[0, 0, 0, 0, many2one_percent], color=palettes[2])
    ax3X.bar(x=x, height=[0, 0, 0, 0, one2many_percent], color=palettes[1])
    ax3X.bar(x=x, height=[0, 0, 0, 0, one2one_percent], color=palettes[0])

    y_bar = [one2one_percent, one2many_percent, many2one_percent, many2many_percent]

    for index, value in zip(y_bar, y_percent):
        if value != 0:
            ax3X.text(4, index, str(np.round(value, 1)) +"%", color='black', ha="center", verticalalignment="bottom")

    ax3X.legend(("Many2Many", "Many2One", "One2Many", "One2One"),loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, ncol=4)
    #-----------------------------------------------------------------

    fig3.tight_layout(pad=3.)
    fig3.tight_layout()

    st.markdown("Number of Source words I: "+ str(result["num_word_source"]))
    st.markdown("Number of Target words J: "+ str(result["num_word_target"]))

    with st.expander("Aligned/unaligned words"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source words**")
            st.write("\# Aligned words: "+ str(result["num_source2notnull_ref"]))
            st.write("\# Unaligned words: "+ str(result["num_source2null_ref"]))

            st.markdown("**Target words**")
            st.write("\# Aligned words: "+ str(result["num_target2notnull_ref"]))
            st.write("\# Unaligned words: "+ str(result["num_target2null_ref"]))
        with col2:
            st.write(fig1)

    with st.expander("Alignment links"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("\# All possible links I*J: "+ str(result["total_num_link"]))
            st.write("\# Sure links: "+ str(result["num_sure"]))
            st.write("\# Fuzzy links: "+ str(result["num_fuzzy"]))
            st.write("\# Non-existing links: "+ str(result["num_no_ref"]) + " (including null links)")
            st.write("\# Null links: "+ str(result["num_no_ref_null"]))
        with col2:
            st.write(fig2)

    with st.expander("Fertility"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("\# One-to-one links: "+ str(result["num_align_ref_one2one"])+"-"+str(result["num_align_ref_one2one"]))
            st.write("\# One-to-many links: "+ str(result["num_align_ref_one2many_source"])+"-"+str(result["num_align_ref_one2many_target"]))
            st.write("\# Many-to-one links: "+ str(result["num_align_ref_many2one_source"])+"-"+str(result["num_align_ref_many2one_target"]))
            st.write("\# Many-to-many links: "+ str(result["num_align_ref_many2many_source"])+"-"+str(result["num_align_ref_many2many_target"]))
        with col2:
            st.write(fig3)


#===============================================================================
#===============================================================================
st.title('Word Alignment Statistics')
st.markdown('''by Anh Khoa Ngo Ho''')
st.markdown('''Word alignment statistics for six corpora: English-French, English-German, English-Romanian, English-Czech, English-Japanese, English-Vietnamese.''')
st.markdown('''The full code is found in [Github](https://github.com/ngohoanhkhoa/Generative_Probabilistic_Alignment_Models). All descriptions and other informations are shown in [Theses.fr](https://www.theses.fr/2021UPASG014).''')

corpora = ('English-French', 'English-German', 'English-Romanian', 'English-Czech', 'English-Japanese', 'English-Vietnamese')
corpus = st.selectbox(label="Corpus:", options=corpora)

if corpus == 'English-French':
    corpus_ = 'en-fr'
    st.markdown('''Number of sentence pairs: 447''')
    st.markdown('''Source: the 2003 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
elif corpus == 'English-German':
    corpus_ = 'en-ge'
    st.markdown('''Number of sentence pairs: 509''')
    st.markdown('''Source: [Europarl](http://www-i6.informatik.rwth-aachen.de/goldAlignment/)''')
elif corpus == 'English-Romanian':
    corpus_ = 'en-ro'
    st.markdown('''Number of sentence pairs: 246''')
    st.markdown('''Source: the 2015 word alignment challenge [Mihalcea and Pedersen, 2003], [url](https://web.eecs.umich.edu/mihalcea/wpt05/)''')
elif corpus == 'English-Czech':
    corpus_ = 'en-cz'
    st.markdown('''Number of sentence pairs: 2 501''')
    st.markdown('''Source: [Marecek, 2016], [url](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804)''')
elif corpus == 'English-Japanese':
    corpus_ = 'en-ja'
    st.markdown('''Number of sentence pairs: 1 235''')
    st.markdown('''Source: [KFTT](http://www.phontron.com/kftt/#alignments)''')
elif corpus == 'English-Vietnamese':
    corpus_ = 'en-vi'
    st.markdown('''Number of sentence pairs: 3 447''')
    st.markdown('''Source: [EVBCorpus](https://code.google.com/archive/p/evbcorpus/)''')


st.header('Statistics')
reference_stats = json.load(open("reference_stats", 'r'))
show_graphs_reference(reference_stats[corpus_])
