python en-ro-var-balancing-cost.py --alpha 0 1 5 10 50 100 --beta 1 5 10 50 100 --gamma 0 0.5 1 --trainFolder en-ro-balancing-cost-train --logFolder en-ro-balancing-cost-log

python en-ro-var-supervise-tunning.py --alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --trainFolder en-ro-supervise-train --logFolder en-ro-supervise-log


git pull && python train.py --data en-ro-toy \
--model model_variational_based.model_variational_bpe_ibm1_balancing_cost \
--model_name test \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_seq_length 150 \
--batch_size 5 \

git pull && python train.py --data en-de \
--model model_hmm_based.model_hmm_bpe_bilstm_ \
--model_name test_new --max_jump_width 200 --jump_width_update_freq -1 \
--source_vocabulary_size 32001 --target_vocabulary_size 32001 \
--shuffle_each_epoch True --max_seq_length 100 --batch_size 2

git pull && python train.py --data en-de --model model_variational_based.model_variational_bpe --model_name test --source_vocabulary_size 32001 --target_vocabulary_size 32001 --shuffle_each_epoch True --max_seq_length 100 --batch_size 2 --max_jump_width 200

python train.py \
--data de-en \
--model model_variational_based.model_variational_bpe \
--model_name ibm1_var_test \
--max_jump_width 150  \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--jump_width_update_freq 0 \
--max_gradient_norm 1. \
--hidden_units 64 \
--embedding_size 64 \
--learning_rate 0.001 \
--max_seq_length 100 \
--keep_prob 1 \
--display_freq 1 \
--alpha_target_expectation 1 \
--alpha_alignment_expectation 1 \
--alpha_KL_divergence 0 \
--model_parameter_loaded_from_checkpoint /vol/work2/2017-NeuralAlignments/exp-ngoho/models/de-en_model_variational_based.model_variational_bpe_ibm1_var_1/de-en_model_variational_based.model_variational_bpe_ibm1_var_1.ckpt-27000

python train.py \
--data de-en \
--model model_variational_based.model_variational_bpe \
--model_name test \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_seq_length 108 \
--min_seq_length 2 \
--alpha_reconstruction_expectation 1 \
--alpha_alignment_expectation 1 \
--alpha_KL_divergence_freq 10 \


python train.py \
--data de-en \
--model model_variational_based.model_variational_bpe_share_params \
--model_name test \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_seq_length 108 \
--min_seq_length 2 \
--alpha_reconstruction_expectation 1 \
--alpha_alignment_expectation 1 \
--alpha_KL_divergence_freq 1 \

python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_share_params_noise_z \
--model_name test \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 0.001 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 100 \
--alpha_alignment_expectation 100 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--batch_size 50 \
--prob_drop_word_noise 0 \
--max_jump_width_noise 0 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_share_params_test.ckpt-35000


python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_noise_z \
--model_name test \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 0.001 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 50 \
--alpha_alignment_expectation 50 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--max_epochs 30 \
--save_freq 1000 \
--prob_drop_word_noise 0.1 \
--max_jump_width_noise 3 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_share_params_test.ckpt-35000



python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_share_params_noise_z_data_hmm \
--model_name 1 \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 50. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 0.001 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 5 \
--alpha_alignment_expectation 50 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--batch_size 50 \
--prob_drop_word_noise 0.1 \
--max_jump_width_noise 3 \
--target_train_data_mono_use 1 \
--source_train_data_mono_use 1 \
--display_freq 1 \
--source_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-fr/europarl-v7.en-fr.cln.low.en.lenSent50.bpe \
--target_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-ro/news.2019.ro.shuffled.deduped.low.lenSent50.bpe.lenSent149 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_share_params_noise_z_data_2.ckpt-85000



git pull && python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_supervise_hmm \
--model_name 1 \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 1e-5 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 5 \
--alpha_alignment_expectation 100 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--batch_size 10 \
--alpha_p0 0.1 \
--p0 0.2 \
--display_freq 1 \
--supervised_mask_threshold 0.5 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_supervise_test_1.ckpt-55000




git pull && python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_share_params_noise_z_data \
--model_name final_1 \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 1e-5 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 5 \
--alpha_alignment_expectation 100 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--batch_size 50 \
--alpha_p0 0.1 \
--p0 0.3 \
--valid_freq 5 \
--display_freq 5 \
--prob_drop_word_noise 0.1 \
--max_jump_width_noise 3 \
--target_train_data_mono_use 1 \
--source_train_data_mono_use 1 \
--train_data_bi_use 1 \
--supervised_mask_threshold 0.5 \
--source_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-fr/europarl-v7.en-fr.cln.low.en.lenSent50.bpe \
--target_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-ro/news.2019.ro.shuffled.deduped.low.lenSent50.bpe.lenSent149 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_share_params_test_1.ckpt-165000 \
--model_parameter_loaded_from_checkpoint_HMM /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_hmm_null_1.ckpt-45000 \


/vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_share_params_hmm_test_2.ckpt-80000


git pull && python train.py \
--data en-fr \
--model model_variational_based.model_variational_bpe_supervise_hmm \
--model_name final_2 \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 1e-5 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 5 \
--alpha_alignment_expectation 100 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--batch_size 50 \
--alpha_p0 0.5 \
--p0 0.0 \
--valid_freq 5 \
--display_freq 5 \
--prob_drop_word_noise 0.1 \
--max_jump_width_noise 3 \
--target_train_data_mono_use 1 \
--source_train_data_mono_use 1 \
--supervised_mask_threshold 0.5 \
--source_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-fr/europarl-v7.en-fr.cln.low.en.lenSent50.bpe \
--target_train_data_mono /vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/en-ro/news.2019.ro.shuffled.deduped.low.lenSent50.bpe.lenSent149 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version8/en-fr_model_variational_based.model_variational_bpe_share_params_hmm_test_2/en-fr_model_variational_based.model_variational_bpe_share_params_hmm_test_2.ckpt-120000


git pull && python train.py \
--data en-ro \
--model model_variational_based.model_variational_bpe_hmm \
--model_name null_2 \
--max_jump_width 150 \
--source_vocabulary_size 32001 \
--target_vocabulary_size 32001 \
--max_gradient_norm 100. \
--embedding_size 128 \
--hidden_units 64 \
--embedding_sample_size 64 \
--learning_rate 0.001 \
--max_seq_length 150 \
--min_seq_length 1 \
--alpha_reconstruction_expectation 10 \
--alpha_alignment_expectation 50 \
--alpha_KL_divergence 0.5 \
--alpha_KL_divergence_freq 25 \
--keep_prob 1.0 \
--model_parameter_loaded_from_checkpoint_IBM1 /vol/work2/2017-NeuralAlignments/exp-ngoho/models-gpu/version9/en-ro_model_variational_based.model_variational_bpe_hmm_null_1.ckpt-45000



python train-gpu-sh.py \
--data en-fr-h \
--model model_hmm_based.model_hmm_transition_target_character_bilstm_emission_target_character_bilstm \
--model_name final \
--max_jump_width 5 \
--source_vocabulary_size 50000 \
--target_vocabulary_size 50000 \
--emission_update_freq -50 \
--jump_width_update_freq -50 \
--max_gradient_norm 5. \
--hidden_units 128 \
--embedding_size 128 \
--learning_rate 1e-3 \
--max_seq_length 50 \