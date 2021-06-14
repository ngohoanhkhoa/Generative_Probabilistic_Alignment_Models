#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from model.model_variational_based.model_variational_bpe_ibm1 import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        
        self.get_reference_BPE_idx()
        
        self.build_optimizer()
        self.build_initializer()
        self.build_word_model_placeholders()
        self.initialize_transition_parameter_nn()
        
        self.build_variational_encoder()
        self.build_reconstruction()
        self.build_alignment()
        
        self.build_update()
        
        
    def train(self):
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        self.test_set = tools.BiTextBPEIterator(source=self.FLAGS.source_test_data,
                                                target=self.FLAGS.target_test_data,
                                                batch_size=1,
                                                n_words_source=self.FLAGS.source_vocabulary_size,
                                                n_words_target=self.FLAGS.target_vocabulary_size)
        
        source_idx_batch = []
        target_idx_batch = []
        
        source_file = open(self.FLAGS.source_idx_test_data)
        source_lines = source_file.readlines()
        
        target_file = open(self.FLAGS.target_idx_test_data)
        target_lines = target_file.readlines()
        
        for i in source_lines:
            source_idx_batch.append(i.split())
    
        for i in target_lines:
            target_idx_batch.append(i.split())
            
        self.source_idx_test_batch = source_idx_batch
        self.target_idx_test_batch = target_idx_batch

        
        self.test_set = tools.BiTextBPEIterator(source=self.FLAGS.source_test_data,
                                                target=self.FLAGS.target_test_data,
                                                batch_size=1,
                                                n_words_source=self.FLAGS.source_vocabulary_size,
                                                n_words_target=self.FLAGS.target_vocabulary_size)
        
        
        self.evaluate()
        for epoch_idx in range(self.FLAGS.max_epochs):
            self.log.info('------------------')
            if self.global_epoch_step.eval() >= self.FLAGS.max_epochs:
                self.log.info('Training is complete. Reach the max epoch number: %d ', self.global_epoch_step.eval())
                break
   
            for train_seq in self.train_set:
                start_time = time.time()
                batch = self.prepare_batch(*train_seq)
                update_info = self.train_batch(*batch)
                self.print_log(update_info, start_time)
                
                self.evaluate()
                self.save_model()

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            # Increase the epoch index of the model
            self.log.info('Epoch %d done at %s', 
                          self.global_epoch_step.eval(), 
                          tools.get_time_now())
            self.global_epoch_step_op.eval()
            
            self.save(self.checkpoint_path, global_step=self.global_step)


        
    def evaluate(self):
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            self.log.info('------------------')
            self.evaluate_alignment(self.valid_set, 'VALIDATION', self.source_idx_batch, self.target_idx_batch, 
                                    self.sure_batch, self.possible_batch)
            self.evaluate_reconstruction(self.valid_set, 'VALIDATION')
            self.evaluate_kl(self.valid_set, 'VALIDATION')
            self.evaluate_alignment(self.test_set, 'TESTING   ', self.source_idx_test_batch, self.target_idx_test_batch, 
                                    self.sure_test_batch, self.possible_test_batch)
            self.evaluate_reconstruction(self.test_set, 'TESTING   ')
            self.evaluate_kl(self.test_set, 'TESTING   ')
            self.log.info('------------------')
            
    def evaluate_reconstruction(self, valid_set, type_set):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            start_time = time.time()
            
            reconstruction_set = []
            accuracy_set = []
            
            for valid_seq in valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.decode_reconstruction(*batch)
                
                for idx in range(1):
                    target_length = batch[4][idx]
                    target = batch[2][idx]
                    
                    num_word_correct = 0
                    for index_word, (predited, correct) in enumerate(zip(eval_info[0][idx], target)):
                        if index_word < target_length:
                            if predited == correct:
                                num_word_correct+=1
                                
                    accuracy_set.append(num_word_correct/target_length)
                    
                reconstruction_set.append(np.prod(eval_info[1]))
                    
            ACC = np.mean(accuracy_set)
            likelihood_score = np.mean(reconstruction_set)
            
            self.log.info('%s: Epoch %d , Step %d , Likelihood: %.10f , ACC: %.5f in %ds at %s',
                          type_set,
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          likelihood_score, 
                          ACC, 
                          time.time() - start_time, 
                          tools.get_time_now())

            self.start_time = time.time()
            
    def evaluate_alignment(self, valid_set, type_set, source_idx_batch, target_idx_batch, sure_batch, possible_batch):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            start_time = time.time()
            
            alignment_set = []
            likelihood_set = []
            emission_set = []
            
            for valid_seq in valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.eval(*batch)
                        
                #--------------------------------------------------------------
                
                for idx in range(1):
                    source_length = batch[1][idx]
                    target_length = batch[4][idx]
                    alignment = []
                    for index_source, index_target in enumerate(eval_info[0][idx]):
                        if index_source < source_length and index_target < target_length:
                            # Check alignment reference: Start from 0 or 1 ?
                            alignment.append(str(index_source+self.evaluate_alignment_start_from) \
                            + "-" + str(index_target+self.evaluate_alignment_start_from))
                            #alignment.append(str(index_source) + "-" + str(index_target))
                            
                    alignment_set.append(alignment)

                likelihood_set.append(eval_info[1])
                emission_set.append(eval_info[2])
                
            alignment_set = self.converse_BPE_to_word(alignment_set, source_idx_batch, target_idx_batch)

            AER = self.calculate_AER(alignment_set, sure_batch, possible_batch)
            likelihood_score = np.mean(likelihood_set)
            
            self.log.info('%s: Epoch %d , Step %d , Likelihood: %.10f , AER: %.5f in %ds at %s',
                          type_set,
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          likelihood_score, 
                          AER, 
                          time.time() - start_time, 
                          tools.get_time_now())
                        
            self.start_time = time.time()
            
    def evaluate_kl(self, valid_set, type_set):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            
            alignment_cost_batch = []
            reconstruction_cost_batch = []
            kl_cost_batch = []
            
            for valid_seq in valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.get_kl(*batch)
                
                alignment_cost_batch.append(eval_info[0])
                reconstruction_cost_batch.append(eval_info[1])
                kl_cost_batch.append(eval_info[2])
                    
            alignment_cost = np.mean(alignment_cost_batch)
            reconstruction_cost = np.mean(reconstruction_cost_batch)
            kl_cost = np.mean(kl_cost_batch)
            
            self.log.info('%s: Epoch %d , Step %d , Alignment: %.5f, Reconstruction: %.5f, KL: %5f',
                          type_set,
                          self.global_epoch_step.eval(),
                          self.global_step.eval() , 
                          alignment_cost, 
                          reconstruction_cost,kl_cost)
            
    def converse_BPE_to_word(self, alignment_set, source_idx_batch, target_idx_batch):
        word_alignments = []
        for subword_alignment, ref_s, ref_t in zip(alignment_set, source_idx_batch, target_idx_batch):
            word_alignment = []
            for subword in subword_alignment:
                subword_src = subword.split('-')[0]
                subword_tgt = subword.split('-')[1]
                word_src = 0
                word_tgt = 0
                for s in ref_s:
                    word_src_ref = s.split('-')[0]
                    subword_src_ref = s.split('-')[1]
                    if subword_src == subword_src_ref:
                        word_src = word_src_ref
                for t in ref_t:
                    word_tgt_ref = t.split('-')[0]
                    subword_tgt_ref = t.split('-')[1]
                    if subword_tgt == subword_tgt_ref:
                        word_tgt = word_tgt_ref
                if word_src + '-'+ word_tgt not in word_alignment:
                    word_alignment.append(word_src + '-'+word_tgt)
            word_alignments.append(word_alignment)
            
        return word_alignments
    
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
        
        
        target_file = open(self.FLAGS.reference_test_data)
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
            
        self.sure_test_batch = sure_batch
        self.possible_test_batch = possible_batch
    
    def calculate_AER(self, alignment_set, sure_batch, possible_batch):
        
        sure_alignment = 0.
        possible_alignment = 0.
        count_alignment = 0.
        count_sure = 0.
        for sure, possible, alignment in zip(sure_batch, possible_batch, alignment_set ):
            for w in alignment:
                if w in sure:
                    sure_alignment+=1.
                if w in possible:
                    possible_alignment+=1.
                    
            count_alignment += float(len(alignment))
            count_sure += float(len(sure))
            
        return 1. - (sure_alignment*2 + possible_alignment)/ (count_alignment + count_sure)