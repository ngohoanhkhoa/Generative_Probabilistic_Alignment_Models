** Analysis for Reference links:
python alignment_analysis.py -r reference_file -s source_corpus -t target_corpus

** Analysis for Prediction links:
python alignment_analysis.py -r reference_file -s source_corpus -t target_corpus -p prediction_file

** Add analysis for rare/unk words:
	Get frequency for words: python build_word_frequency.py train_corpus_file
	Run: --show_frequency True -fm maximum_frequency_for_rare_word -fs frequency_for_source -ft frequency_for_target

** Add analysis for POS:
	Get POS for corpus using Space, this file is created by replace each word by its POS
	Run: --show_part_of_speech True -ps pos_for_source -pt pos_for_target