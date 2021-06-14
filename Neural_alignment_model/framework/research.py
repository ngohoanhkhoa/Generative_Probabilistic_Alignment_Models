import os
import framework.tools as tools
import io
import numpy as np

class Research(object):
    
    def __init__(self, model, FLAGS):
        self.model = model
        self.FLAGS = FLAGS

        self.train_set = self.model.train_set        
        
        self.inverse_source_dict = self.model.train_set.inverse_source_dict
        self.inverse_target_dict = self.model.train_set.inverse_target_dict
        
        self.jump_distance_file = None
        self.transition_file = None
        self.emission_file = None
        self.alignment_file = None
        
    def make_file_path(self, name):
        path = os.path.join(self.FLAGS.model_dir,
                            self.FLAGS.model_name, 
                            str(name) + '.'+ str(self.model.global_step.eval() ))
        return path
            
    def make_distance_file(self):
        self.jump_distance_file = io.open(self.make_file_path('distance'), 'a',encoding='utf-8')
        
    def make_transition_file(self):
        self.transition_file = io.open(self.make_file_path('transition'), 'w',encoding='utf-8')
        
    def make_emission_file(self):
        self.emission_file = io.open(self.make_file_path('emission'), 'w',encoding='utf-8')
        
    def make_alignment_file(self):
        self.alignment_file = io.open(self.make_file_path('alignment'), 'w',encoding='utf-8')
            
    
    def write_distance_file(self, negative_set_value):
        path = os.path.join(self.FLAGS.model_dir,
                            self.FLAGS.model_name, 
                            'distance.0')
        jump_distance_file = io.open(path, 'a',encoding='utf-8')

        for i in negative_set_value:
            jump_distance_file.write(u'{} '.format(i))
        jump_distance_file.write(u'\n')
        jump_distance_file.close()
        
    def write_p0_file(self, p0):
        path = os.path.join(self.FLAGS.model_dir,
                            self.FLAGS.model_name, 
                            'p0.0')
        jump_p0_file = io.open(path, 'a',encoding='utf-8')

        jump_p0_file.write(u'{} '.format(p0))
        jump_p0_file.close()
        
                   
    def write_transition_file(self, transition):
        transition_file = io.open(self.make_file_path('transition'), 'a',encoding='utf-8')
        for i in transition:
            for j in i :
                transition_file.write(u'{} '.format(j))
            transition_file.write(u'\n')
        transition_file.write(u'-------\n')
        transition_file.close()
            
    def write_emission_file(self, source_sentence, target_sentence, emission):
        emission_file = io.open(self.make_file_path('emission'), 'a',encoding='utf-8')
        
        source_sent_ = []
        for widx in source_sentence:
            if widx == 0:
                break
            source_sent_.append(self.inverse_source_dict.get(widx, tools.UNK))
            
        target_sent_ = []
        for widx in target_sentence:
            if widx == 0:
                break
            target_sent_.append(self.inverse_target_dict.get(widx, tools.UNK))
            
        for i, t in enumerate(target_sent_):
            for j, s in enumerate(source_sent_):
                emission_file.write(u'{} '.format(t))
                emission_file.write(u'{} '.format(s))
                emission_file.write(u'{} '.format(emission[i,j])) 
                emission_file.write(u'\n')
        emission_file.close()
                
    def write_emission_file_complementary_sum_sampling(self, source_sentence, target_sentence, emission, sources):
        positive_source_vocabulary = np.unique(np.reshape(sources, (-1)))
        
        emission_file = self.emission_file
        
        source_sent_ = []
        for widx in source_sentence:
            if widx == 0:
                break
            source_sent_.append(self.inverse_source_dict.get(widx, tools.UNK))
            
        target_sent_ = []
        for widx in target_sentence:
            if widx == 0:
                break
            target_sent_.append(self.inverse_target_dict.get(widx, tools.UNK))
            
        for i, t in enumerate(target_sent_):
            for j, [s,s_] in enumerate(zip(source_sent_, source_sentence)):
                emission_file.write(u'{} '.format(t))
                emission_file.write(u'{} '.format(s))
                emission_file.write(u'{} '.format(emission[i,np.where(positive_source_vocabulary==s_)[0][0]]))
                emission_file.write(u'\n')
                
    def get_alignment_file(self, source_sentence, target_sentence, q_star):
        alignment_file = io.open(self.make_file_path('alignment'), 'a',encoding='utf-8')
        source_sent = tools.idx_to_sent(self.inverse_source_dict, source_sentence)
        target_sent = tools.idx_to_sent(self.inverse_target_dict, target_sentence)
        alignment_file.write("--Source: " + source_sent + '\n')
        alignment_file.write("--Target: " + target_sent + '\n')
        for i in q_star:
            alignment_file.write("{} ".format(i))
        alignment_file.write("\n")
        alignment_file.close()