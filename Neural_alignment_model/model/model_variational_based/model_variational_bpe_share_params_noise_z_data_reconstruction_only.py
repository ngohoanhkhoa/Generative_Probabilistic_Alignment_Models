#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from tensorflow.python import pywrap_tensorflow

from model.model_variational_based.model_variational_bpe_share_params_noise_z_data import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        super(Model, self).__init__(FLAGS, session, log)
        
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        # Cost for e -> y -> e, f
        e_y_reconstruction_expectation = tf.reduce_mean(self.e_y_reconstruction_expectation)
        f_y_alignment_expectation = tf.reduce_mean(self.f_y_alignment_expectation)
        y_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.y_KL_divergence, axis=1))
        
        self.e_y_cost_reconstruction_expectation =  (-e_y_reconstruction_expectation)
        self.f_y_cost_alignment_expectation =  (-f_y_alignment_expectation)
        self.y_cost_KL_divergence = y_KL_divergence * self.annealing_KL_divergence
        self.y_cost_KL_divergence_eval = y_KL_divergence
        
        self.y_cost = (self.FLAGS.alpha_alignment_expectation * self.f_y_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.y_cost_KL_divergence)
                
        
        # Cost for f -> x -> f, e
        f_x_reconstruction_expectation = tf.reduce_mean(self.f_x_reconstruction_expectation)
        e_x_alignment_expectation = tf.reduce_mean(self.e_x_alignment_expectation)
        x_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.x_KL_divergence, axis=1))
        
        self.f_x_cost_reconstruction_expectation =  (-f_x_reconstruction_expectation)
        self.e_x_cost_alignment_expectation =  (-e_x_alignment_expectation)
        self.x_cost_KL_divergence = x_KL_divergence * self.annealing_KL_divergence
        self.x_cost_KL_divergence_eval = x_KL_divergence
               
        self.x_cost = (self.FLAGS.alpha_alignment_expectation * self.e_x_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.x_cost_KL_divergence)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.y_cost + self.x_cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
        
        y_gradients_reconstruction = tf.gradients(self.e_y_cost_reconstruction_expectation, trainable_params)
        y_clip_gradients_reconstruction, _ = tf.clip_by_global_norm(y_gradients_reconstruction, self.max_gradient_norm)
        self.updates_reconstruction_y = self.opt.apply_gradients(zip(y_clip_gradients_reconstruction,trainable_params),
                                                         global_step=self.global_step_reconstruction_y)
        
        x_gradients_reconstruction = tf.gradients(self.f_x_cost_reconstruction_expectation, trainable_params)
        x_clip_gradients_reconstruction, _ = tf.clip_by_global_norm(x_gradients_reconstruction, self.max_gradient_norm)
        self.updates_reconstruction_x = self.opt.apply_gradients(zip(x_clip_gradients_reconstruction,trainable_params),
                                                         global_step=self.global_step_reconstruction_x)