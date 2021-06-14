import tensorflow as tf

# Avoid thread explosion
#os.environ["OMP_NUM_THREADS"] = "8"
#os.environ["MKL_NUM_THREADS"] = "8"

# For GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

data_original_dir = '/vol/work2/2017-NeuralAlignments/exp-ngoho/data_original/'
FLAGS = tf.app.flags.FLAGS

# MODEL INFO
tf.app.flags.DEFINE_string('note',None, 'Note about the training')
tf.app.flags.DEFINE_string('model', None, 'Model for training')
tf.app.flags.DEFINE_string('model_name', None, 'File name used for model')
tf.app.flags.DEFINE_string('data', None, 'Model for training')
tf.app.flags.DEFINE_string('model_dir','/vol/work2/2017-NeuralAlignments/exp-ngoho/models/', 'Path to save model')
tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint',None, 'Load Model parameters from file, if None get the last checkpoint')

# ALIGNMENT
tf.app.flags.DEFINE_integer('evaluate_alignment_start_from', 1 , 'For evaluation, alignment starts from (0 or 1)')
tf.app.flags.DEFINE_integer('emission_update_freq', 0, 'Emission fixed parameter update freq, # epoch')
tf.app.flags.DEFINE_integer('jump_width_update_freq', 0, 'Jump distance update rate, # epoch')
tf.app.flags.DEFINE_float('p0',0.01, 'P0')
tf.app.flags.DEFINE_integer('max_jump_width', 80, 'Max distance')

tf.app.flags.DEFINE_string('initialize_jump_set', 'heuristic', 'Initialize Jump distance set (heuristic, uniform, random)')
tf.app.flags.DEFINE_string('initial_transition_pi', 'heuristic', 'Set value for initial transition pi (heuristic, uniform, random)')
tf.app.flags.DEFINE_float('transition_heuristic_prob', 0.6, 'Set value for initial transition at Jump Width = 1 (heuristic)')
tf.app.flags.DEFINE_float('alpha_transition', 1.0, 'Set value')
tf.app.flags.DEFINE_float('alpha_p0', 1.0, 'Set value')

# EXTENSION
# Context
tf.app.flags.DEFINE_integer('target_window_size', 0, 'Size of window for target sentence - context')
tf.app.flags.DEFINE_integer('source_window_size', 0, 'Size of window for source sentence - history')
# Sub-vocabulary
tf.app.flags.DEFINE_integer('source_sub_vocabulary_size', 5000, 'Word source sub vocabulary size')
# Character model
tf.app.flags.DEFINE_integer('character_source_vocabulary_size', -1, 'Character source vocabulary size (= -1 for all character)')
tf.app.flags.DEFINE_integer('character_target_vocabulary_size', -1, 'Character target vocabulary size (= -1 for all character)')
tf.app.flags.DEFINE_integer('max_word_source_length', 40, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('max_word_target_length', 40, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('character_hidden_units', 64, 'Character hidden units')
tf.app.flags.DEFINE_integer('character_embedding_size', 64, 'Character embedding units')
tf.app.flags.DEFINE_boolean('use_source_character', True, 'Use source character')
# Word model
tf.app.flags.DEFINE_integer('source_vocabulary_size', 50000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('target_vocabulary_size', 50000, 'Target vocabulary size')
tf.app.flags.DEFINE_integer('hidden_units', 64, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('embedding_size', 64, 'Embedding dimensions of encoder and decoder inputs')
# Seq2seq model
tf.app.flags.DEFINE_integer('max_seq_length', 50, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('min_seq_length', 1, 'Minimum sequence length')
# TRAINING PARAMETERS
tf.app.flags.DEFINE_integer('max_epochs', 100, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 100, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length',True, 'Sort pre-fetched minibatches by their target sequence lengths')

tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_integer('display_freq',10, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 100, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_integer('save_freq', 500, 'Save model checkpoint every this iteration')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_integer('intra_op_parallelism_threads', 4, 'Number of cpu will be used')
tf.app.flags.DEFINE_integer('inter_op_parallelism_threads', 4, 'Number of cpu will be used')
tf.app.flags.DEFINE_string('mode', 'train', 'Mode (train, evaluate)')


if FLAGS.data == 'en-fr':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-fr/testing.low.en', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-fr/testing.low.fr', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-fr/testing.en-fr.align', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'fr-en':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-fr/testing.low.fr', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-fr/testing.low.en', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-fr/testing.fr-en.align', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.fr.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-fr/europarl-v7.en-fr.cln.low.en.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'en-ro':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-ro/corp.test.ro-en.cln.en.low', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-ro/corp.test.ro-en.cln.ro.low', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-ro/test.en-ro.ali.startFrom1', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'ro-en':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-ro/corp.test.ro-en.cln.ro.low', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-ro/corp.test.ro-en.cln.en.low', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-ro/test.ro-en.ali.startFrom1', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.ro.utf8.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-ro/train.merg.en-ro.cln.en.utf8.low.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'en-cz':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-cz/training.en-cz.en.tok.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-cz/training.en-cz.cz.tok.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-cz/training.en-cz.en.tok.low.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-cz/training.en-cz.cz.tok.low.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-cz/testing.en-cz.en.low', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-cz/testing.en-cz.cz.low', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-cz/testing.en-cz.alignment.fixed', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-cz/training.en-cz.en.tok.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-cz/training.en-cz.cz.tok.low.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'cz-en':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-cz/training.en-cz.cz.tok.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-cz/training.en-cz.en.tok.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-cz/training.en-cz.cz.tok.low.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-cz/training.en-cz.en.tok.low.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-cz/testing.en-cz.cz.low', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-cz/testing.en-cz.en.low', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-cz/testing.cz-en.alignment.fixed', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-cz/training.en-cz.cz.tok.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-cz/training.en-cz.en.tok.low.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'en-de':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-de/corp.train.de-en.en.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-de/corp.train.de-en.de.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-de/corp.train.de-en.low.cln.en.final.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-de/corp.train.de-en.low.cln.de.final.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-de/corp.test.de-en.en.low.ngoho', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-de/corp.test.de-en.de.low.ngoho', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-de/alignmentDeEn.fixed.ali.startFrom1.en-de.ngoho', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-de/corp.train.de-en.en.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-de/corp.train.de-en.de.low.character.pkl', 'Path to target character vocabulary')
if FLAGS.data == 'de-en':
    tf.app.flags.DEFINE_string('source_vocabulary', data_original_dir+'en-de/corp.train.de-en.de.low.pkl', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', data_original_dir+'en-de/corp.train.de-en.en.low.pkl', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', data_original_dir+'en-de/corp.train.de-en.low.cln.de.final.lenSent50', 'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', data_original_dir+'en-de/corp.train.de-en.low.cln.en.final.lenSent50', 'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', data_original_dir+'en-de/corp.test.de-en.de.low.ngoho', 'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', data_original_dir+'en-de/corp.test.de-en.en.low.ngoho', 'Path to target validation data')
    tf.app.flags.DEFINE_string('reference_valid_data', data_original_dir+'en-de/alignmentDeEn.fixed.ali.startFrom1.de-en.ngoho', 'Path to alignment validation data')
    tf.app.flags.DEFINE_string('source_character_vocabulary', data_original_dir+'en-de/corp.train.de-en.de.low.character.pkl', 'Path to source character vocabulary')
    tf.app.flags.DEFINE_string('target_character_vocabulary', data_original_dir+'en-de/corp.train.de-en.en.low.character.pkl', 'Path to target character vocabulary')

# Pretrained model
if FLAGS.data == 'en-fr':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-fr_hmm_target_character.ckpt-54000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/fr-en_hmm_target_character_1.ckpt-39000' , '')
if FLAGS.data == 'fr-en':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/fr-en_hmm_target_character_1.ckpt-39000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-fr_hmm_target_character.ckpt-54000' , '')
if FLAGS.data == 'en-ro':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-ro_hmm_target_ctx_character_3.ckpt-21000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/ro-en_hmm_target_ctx_character_1.ckpt-9000' , '')
if FLAGS.data == 'ro-en':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/ro-en_hmm_target_ctx_character_1.ckpt-9000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-ro_hmm_target_ctx_character_3.ckpt-21000' , '')
if FLAGS.data == 'en-cz':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-cz_hmm_target_ctx_character_3.ckpt-6000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/cz-en_hmm_target_ctx_character_2.ckpt-3000' , '')
if FLAGS.data == 'cz-en':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/cz-en_hmm_target_ctx_character_2.ckpt-3000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-cz_hmm_target_ctx_character_3.ckpt-6000' , '')
if FLAGS.data == 'en-de':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-de_hmm_target_character.ckpt-57000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/de-en_hmm_target_character.ckpt-90000' , '')
if FLAGS.data == 'de-en':
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_source', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/de-en_hmm_target_character.ckpt-90000' , '')
    tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint_target', '/vol/work2/2017-NeuralAlignments/exp-ngoho/pretrained_model/en-de_hmm_target_character.ckpt-57000' , '')

if 'character_subvoc_scan' in FLAGS.model:
    from framework.alignment_character_subvoc import Alignment
elif 'character' in FLAGS.model:
    from framework.alignment_character import Alignment
else:
    from framework.alignment import Alignment
    
def main(_):
    
    alignment = Alignment(FLAGS)
    alignment.run()
    
if __name__ == '__main__':
    tf.app.run()

#==============================================================================
# 
#==============================================================================