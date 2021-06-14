#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
import numpy as np
import tensorflow as tf
import framework.tools as tools

#==============================================================================
# 
#==============================================================================

class BaseModel(object):

    def __init__(self, FLAGS, session, log):
        self.start_time = time.time()
        
        # Model
        self.FLAGS = FLAGS
        self.sess = session
        self.log = log
        
        self.model_path = os.path.join(self.FLAGS.model_dir, self.FLAGS.data+'_'+self.FLAGS.model)
        self.file_name = self.FLAGS.data+'_'+self.FLAGS.model
        self.checkpoint_path = os.path.join(self.model_path,
                                            self.file_name + '.ckpt')
        if self.FLAGS.model_name is not None:
            self.model_path = os.path.join(self.FLAGS.model_dir, self.FLAGS.data+'_'+self.FLAGS.model+'_'+self.FLAGS.model_name)
            self.checkpoint_path = os.path.join(self.model_path,
                                            self.file_name+'_'+self.FLAGS.model_name + '.ckpt')
        self.restored = False
        
        # dtype should be the same, tf_float_dtype = tf.float32 could have problem with cnn
        self.int_dtype = np.int64
        self.float_dtype = np.float64
        self.tf_float_dtype = tf.float32
        self.tf_int_dtype = tf.int32
        
        # Data
        self.train_set = None
        self.valid_set = None
        
        # Training model
        self.batch_size = FLAGS.batch_size
        self.max_seq_length = FLAGS.max_seq_length
        self.optimizer = FLAGS.optimizer
        self.learning_rate = FLAGS.learning_rate
        self.max_gradient_norm = FLAGS.max_gradient_norm
        
        self.emission_update_freq = self.FLAGS.emission_update_freq
        self.jump_width_update_freq = self.FLAGS.jump_width_update_freq
        
        # Tensorflow parameter
        self.keep_prob = tf.placeholder(self.tf_float_dtype, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        
        # Alignment model
        self.evaluate_alignment_start_from = FLAGS.evaluate_alignment_start_from
        
        self.sure_batch = None
        self.possible_batch = None
        self.get_reference()
        self.best_AER = 100.
        

        # Build model
        self.build_model()
        
#==============================================================================
# Store/Restore model
#==============================================================================

    def save(self, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        # temporary code
        #del tf.get_collection_ref('LAYER_NAME_UIDS')[0]
        save_path = saver.save(self.sess, save_path=path, global_step=global_step)
        self.log.info('Model saved at %s', save_path)
        
    def save_model(self):
        # Save the model checkpoint
        if self.FLAGS.save_freq != 0 and self.global_step.eval() % self.FLAGS.save_freq == 0:
            self.log.info('SAVE MODEL: %s' , self.checkpoint_path)
            self.save(self.checkpoint_path, global_step=self.global_step)
        
    def restore(self, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(self.sess, save_path=path)
        self.log.info('Model restored from %s', path)
        self.restored = True
            
    def get_file_path(self, name):
        path = os.path.join(self.model_path,
                            self.file_name + '_' + str(name) + '.' + str(self.global_step.eval() ) )
        return path
    
    def print_log(self, cost, start_time):
        if self.global_step.eval() % self.FLAGS.display_freq == 0:
            self.log.info('Epoch %d , Step %d , Cost: %.5f in %ds at %s', 
                          self.global_epoch_step.eval(), 
                          self.global_step.eval(),
                          cost,
                          time.time() - self.start_time,
                          tools.get_time_now())
            self.start_time = time.time()

#==============================================================================
# 
#==============================================================================

    def get_reference(self):
        target_file = open(self.FLAGS.reference_valid_data)
        target_lines = target_file.readlines()
        target_lines = [str(line[:-1]) for line in target_lines]
        target_lines = np.reshape(target_lines, (np.int(len(target_lines)/2), 2))
        
        sure_batch = []
        possible_batch = []
        
        sure = target_lines[:,0]
        possible = target_lines[:,1]
        
        for i in sure:
            sure_batch.append(i.split())
            
        for i in possible:
            possible_batch.append(i.split())
            
        self.sure_batch = sure_batch
        self.possible_batch = possible_batch
        
    def calculate_AER(self, alignment_set):
        
        sure_alignment = 0.
        possible_alignment = 0.
        count_alignment = 0.
        count_sure = 0.
        for sure, possible, alignment in zip(self.sure_batch, self.possible_batch, alignment_set ):
            for w in alignment:
                if w in sure:
                    sure_alignment+=1.
                if w in possible:
                    possible_alignment+=1.
                    
            count_alignment += float(len(alignment))
            count_sure += float(len(sure))
            
        return 1. - (sure_alignment*2 + possible_alignment)/ (count_alignment + count_sure)
    
    def viterbi(self, observations, N, pi, T, E_):
        T = np.transpose(T) #(j,i)
          
        L = observations.shape[0]
        q_star = np.zeros(L)
    
        delta = np.zeros((L, N))
        psi = np.zeros((L, N))
        
        """ Initialization """
        delta[0, :] = pi * E_[:, 0]
    
        """ Forward Updates """
        for t in range(1, L):
            #temp (j,i)
            temp = T * delta[t-1, :]
            psi[t, :] = np.argmax(temp, axis=1)
            delta[t, :] = np.max(temp, axis=1) * E_[:, t]
            
        """ Termination """
        q_star[L-1] = np.argmax(delta[L-1, :])
        p_star = np.max(delta[L-1, :])
    
        """ Backward state sequence """
        for t in range(L-2, -1, -1):
            q_star[t] = psi[t+1, int(q_star[t+1])]
            
        return [int(x) for x in q_star], p_star