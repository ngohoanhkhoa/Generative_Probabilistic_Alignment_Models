
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io

try:
    import cPickle as pkl
except:
    import pickle as pkl
    
from model_IBM1 import Model

#==============================================================================
"""
Model 1: Evaluate IBM1 model from Vocabulary table
"""
#==============================================================================

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        super(Model, self).__init__(FLAGS,session, log)
        
#==============================================================================
# 
#==============================================================================
        
    def train(self):
        self.log.info('----------------')
        self.log.info('----------------')
        #----------------------------------------------------------------------
        if self.FLAGS.IBM1_parameter_loaded_from_file is not None:
            IBM_parameter_file = io.open(self.FLAGS.IBM1_parameter_loaded_from_file, mode='rb')
            self.emission_vocabulary_table_IBM1 = pkl.load(IBM_parameter_file)
            self.log.info('Train IBM1: LOAD IBM1 PARAMETERS: %s', self.FLAGS.IBM1_parameter_loaded_from_file)
            self.evaluate(0)
        else:
            self.log.info('IBM1 Parameter file is not found')