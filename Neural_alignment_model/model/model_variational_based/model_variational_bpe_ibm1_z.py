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
        
    def eval(self,
             sources,
             source_lengths,
             targets,
             targets_null,
             target_lengths):
        
        target_length = target_lengths[0]
        source_length = source_lengths[0]
        source = sources[0][:source_length]
        
        emission_finals = []
        for n in range(self.FLAGS.latent_variable_number):
            input_feed = {}
            input_feed[self.keep_prob.name] = 1.0
            input_feed[self.sources.name] = sources
            input_feed[self.targets.name] = targets
            input_feed[self.source_lengths.name] = source_lengths
            input_feed[self.target_lengths.name] = target_lengths
            
            input_feed[self.targets_null.name] = targets_null
            
            output_feed = [self.emission]
            #----------------------------------------------------------------------
            outputs = self.sess.run(output_feed, input_feed)
            
            emission_finals.append(np.array(outputs[0][0]))
        
        emission_final = np.mean(emission_finals, axis=0)
        
        initial_transition, transition = self.get_transition_evaluation_nn(target_length)
        
        
        state_seq, likelihood_seq = self.viterbi(np.array(source),
                                        target_length*2,
                                        initial_transition,
                                        transition, 
                                        emission_finals)
        
        return [state_seq], [likelihood_seq], [emission_final]
    
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
        
        self.log.info('Alpha reconstruction expectation: %s', self.FLAGS.alpha_reconstruction_expectation)
        self.log.info('Alpha alignment expectation: %s', self.FLAGS.alpha_alignment_expectation)
        self.log.info('Alpha KL divergence: %s', self.FLAGS.alpha_KL_divergence)
        self.log.info('Alpha KL divergence freq: %s', self.FLAGS.alpha_KL_divergence_freq)
        
        self.log.info('Number of latent variable: %s', self.FLAGS.latent_variable_number)
        
    def train(self):
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        self.evaluate_alignment()
        
    def viterbi(self, observations, N, pi, T, E_s):
        deltas = []
        for E_ in E_s:
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
                
            deltas.append(delta)
            
        delta_ = np.mean(deltas,axis=0)
            
        """ Termination """
        q_star[L-1] = np.argmax(delta_[L-1, :])
        p_star = np.max(delta_[L-1, :])
    
        """ Backward state sequence """
        for t in range(L-2, -1, -1):
            q_star[t] = psi[t+1, int(q_star[t+1])]
            
        return [int(x) for x in q_star], p_star