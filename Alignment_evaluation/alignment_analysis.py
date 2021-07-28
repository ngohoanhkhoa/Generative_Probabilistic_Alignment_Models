# Please refer to the thesis https://www.theses.fr/2021UPASG014 for any futher information.

# =============================================================================
# HOW TO COUNT LINKS
# + Align-link is a link existing in Reference or Prediction
# + No-link is a link not existing in Reference or Prediction
# + Null-link is a special case of No-link:
#    For_all i' in I, j' in J : i-j' and i'-j are not existing, i-j is a null link
# + Src-Null, Tgt-Null: Source or Target word is aligned to Null token.
# + Explain the table:
#   - The total number of links is I*J, we check each i-j for i in I and j in J
#   - Align-TP: a link is existing in both Prediction and Reference
#   - Align-FP: a link is existing in Prediction but Reference
#   - Align-TN: a link is not existing in both Prediction and Reference
#   - Align-FN: a link is not existing in Prediction but Reference
#       - In Align-FP, Align-TN and Align-FN, the table shows <X (Null: Y)> where X is the total number, 
#       Y represents the number of null-links:
#       - In Align-FP, Null-link in Reference
#       - In Align-TN, Null-link in Prediction
#       - In Align-FN, Null-link in Prediction
#   - One2Many: S - T (TP=Y,FP=X) where S is the number of source word aligning many T target words.
#   - Many2One: S - T (TP=Y,FP=X)
#   - Many2Many: S - T (Z) (TP=Y,FP=X) where Z is the total number of link
#   - Src-Rare: S - T (Z) where Z is the mean of fertility.
#
# + Explain the table for null-link:
#   - The total number of links is I+J, it means we have a set [1-null, 2-null, ..., I-null ..., null-J]
#   - Null-link: The total number of source word j and target word i that
#     For_all i' in I, j' in J : i-j' and i'-j are not existing
#   - Not-Null-link: The total number of source word j and target word i:
#     Exist i' in I that i'-j is existing OR Exist j' in J that i-j' is existing
#   - Null-TP: a null-link in both Prediction and Reference
#   - Null-FP: a null-link in Prediction but Reference
#   - Null-TN: a null-link not in both Prediction and Reference
#   - Null-FN: a null-link not in Prediction but Reference

# POS [Source Content word, Source Function word, Target Content word, Target Function word]

#                             Ref                Ref
#                            align             no-align
# Model   align         num_true_align TP   num_false_align FP
# Model   no-align      num_false_no FN     num_true_no TN

#                           Ref                Ref
#                          null               no-null
# Model   null         num_true_align TP   num_false_align FP
# Model   no-null      num_false_no FN     num_true_no TN
# =============================================================================

import numpy as np
import argparse

round_value = 3

def clean_sentence(line):
    line = line.replace("\n",'')
    
    return line

def clean_alignment(line):
    line = line.replace("\n",'')
    
    return line
                
def get_prediction(prediction_file_name):
    prediction_file = open(prediction_file_name, 'r', encoding='utf-8')
    prediction_output = []
    for line in prediction_file:
        prediction_output.append([ x for x in clean_alignment(line).split() ])
        
    return prediction_output

def get_reference(reference_file_name, ref_only=True):
    reference_file = open(reference_file_name, 'r', encoding='utf-8')
    
    reference_lines = reference_file.readlines()
    reference_lines = np.reshape(reference_lines, (np.int(len(reference_lines)/2), 2))

    ref = []
    
    if ref_only:
        for s, p in zip(reference_lines[:,0], reference_lines[:,1]):
            s = clean_alignment(s)
            p = clean_alignment(p)
            ref.append(s.split() + p.split())
        return ref
    else:
        sure = []
        possible = []
    
        for s, p in zip(reference_lines[:,0], reference_lines[:,1]):
            s = clean_alignment(s)
            p = clean_alignment(p)
            sure.append(s.split())
            possible.append(p.split())
            ref.append(s.split() + p.split())
        
        return ref, sure, possible
    
def get_corpus_file(corpus_file_name):
    corpus_file = open(corpus_file_name, 'r', encoding='utf-8')
    corpus_output = []
    for line in corpus_file:
        corpus_output.append([ x for x in clean_sentence(line).split() ])
        
    return corpus_output
    
def calculate_AER(reference_file_name, prediction_file_name):
    reference_set, sure_set, fuzzy_set = get_reference(reference_file_name, ref_only=False)
    prediction_set = get_prediction(prediction_file_name)
    
    sure_correct = 0.
    fuzzy_correct = 0.
    count_alignment = 0.
    count_sure = 0.
    
    for sure, fuzzy, prediction in zip(sure_set, fuzzy_set, prediction_set):
        for align in prediction:
            if align in sure:
                sure_correct+=1.
            if align in fuzzy:
                fuzzy_correct+=1.
                
        count_alignment += float(len(prediction))
        count_sure += float(len(sure))
    
    aer = 1. - (sure_correct*2 + fuzzy_correct)/ (count_alignment + count_sure)
    
    return np.round(aer, round_value)

def calculate_scores(tp, fp, tn, fn):
    acc = 0.
    if tp + fp + tn + fn != 0:
        acc = (tp+tn)/(tp + fp + tn + fn)
        
    precision = 0.
    if tp+fp != 0.:
        precision = tp/(tp+fp)
        
    recall = 0.
    if tp+fn != 0.:
        recall = tp/(tp+fn)
        
    f1 = 0.
    if precision+recall != 0:
        f1 = (2*precision*recall)/(precision+recall)
    
    acc = np.round(acc, round_value)
    precision = np.round(precision, round_value)
    recall = np.round(recall, round_value)
    f1 = np.round(f1, round_value)
    
    return acc, precision, recall, f1

def analyse_reference(reference_set,
                      sure_set, possible_set,
                      source_set, target_set):

    #-------------------------------------
    total_num_link = 0
    total_null_link = 0
    
    num_word_source = 0
    num_word_target = 0
    
    # Count for Reference
    num_align_ref = 0
    num_no_ref = 0
    
    num_sure = 0
    num_fuzzy = 0
    
    num_align_ref_one2one = 0
    num_align_ref_one2many_source = 0
    num_align_ref_one2many_target = 0
    num_align_ref_many2one_source = 0
    num_align_ref_many2one_target = 0
    num_align_ref_many2many = 0
    num_align_ref_many2many_source = 0
    num_align_ref_many2many_target = 0
    
    num_no_ref_no = 0
    num_no_ref_null = 0


    # Null
    num_source2null_ref = 0
    num_target2null_ref = 0
    num_source2notnull_ref = 0
    num_target2notnull_ref = 0
    
    num_source2null_ref_ratio_list = []
    num_target2null_ref_ratio_list = []
    
    for line in sure_set:
        for s in line:
            num_sure+= 1
    
    for line in possible_set:
        for p in line:
            num_fuzzy+= 1
    
    for idx, [source, target, ref] in enumerate(zip(source_set, target_set, reference_set)):
        
        source_len = len(source)
        target_len = len(target)
        
        total_num_link += ((source_len) * (target_len))
        total_null_link += ((source_len) + (target_len))
        
        num_word_source += source_len
        num_word_target += target_len
        

        #---------------------------
        # Null
        num_source2null_ref_sent = 0
        num_target2null_ref_sent = 0
        
        for idx_source in range(1, source_len+1):
            source_to_null_ref = True
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    source_to_null_ref = False
                    
            if source_to_null_ref:
                num_source2null_ref+=1
                num_source2null_ref_sent+=1
            else:
                num_source2notnull_ref+=1
                    
            
        for idx_target in range(1, target_len+1):
            target_to_null_ref = True
            for idx_source in range(1, source_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    target_to_null_ref = False
                    
            if target_to_null_ref:
                num_target2null_ref+=1
                num_target2null_ref_sent+=1
            else:
                num_target2notnull_ref+=1
                

                
        num_source2null_ref_ratio_list.append(num_source2null_ref_sent/source_len)
        num_target2null_ref_ratio_list.append(num_target2null_ref_sent/target_len)
        
        num_align_ref_one2many_source_list = []
        num_align_ref_many2one_target_list = []
        num_align_ref_many2many_source_list = []
        num_align_ref_many2many_target_list = []
        
        align_ref_one2one_list = []
        align_ref_one2many_list = []
        align_ref_many2one_list = []
        align_ref_many2many_list = []
        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                
                # Count number of links in Ref
                if align_check in ref:
                    num_align_ref +=1
                    
                    check_one2many = False
                    check_many2one = False
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in ref and align_check_ != align_check:
                            check_one2many = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in ref and align_check_ != align_check:
                            check_many2one = True
                    
                    if check_one2many is True and check_many2one is False:
                        num_align_ref_one2many_target +=1
                        align_ref_one2many_list.append(align_check)
                        if idx_source not in num_align_ref_one2many_source_list:
                            num_align_ref_one2many_source_list.append(idx_source)
                    if check_many2one is True and check_one2many is False:
                        num_align_ref_many2one_source +=1
                        align_ref_many2one_list.append(align_check)
                        if idx_target not in num_align_ref_many2one_target_list:
                            num_align_ref_many2one_target_list.append(idx_target)
                    if check_one2many is False and check_many2one is False:
                        num_align_ref_one2one +=1
                        align_ref_one2one_list.append(align_check)
                        
                # Count number of links not in Ref
                if align_check not in ref:
                    num_no_ref +=1
                    
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in ref:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in ref:
                            target_to_null = False
                            
                    if source_to_null and target_to_null:
                        num_no_ref_null +=1
                        
                    else:
                        num_no_ref_no +=1
                
                        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                # Count number of many2many links in Ref
                if align_check in ref:
                    
                    check_one2many = False
                    check_many2one = False
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in ref and align_check_ != align_check:
                            check_one2many = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in ref and align_check_ != align_check:
                            check_many2one = True
                            
                    if check_one2many is True and check_many2one is True:
                        if idx_source not in num_align_ref_many2many_source_list \
                        and idx_source not in num_align_ref_one2many_source_list:
                            num_align_ref_many2many_source_list.append(idx_source)
                        if idx_target not in num_align_ref_many2many_target_list \
                        and idx_target not in num_align_ref_many2one_target_list:
                            num_align_ref_many2many_target_list.append(idx_target)
                        num_align_ref_many2many+=1
                        align_ref_many2many_list.append(align_check)
                        
                        
        num_align_ref_one2many_source += len(num_align_ref_one2many_source_list)
        num_align_ref_many2one_target += len(num_align_ref_many2one_target_list)
        
        num_align_ref_many2many_source += len(num_align_ref_many2many_source_list)
        num_align_ref_many2many_target += len(num_align_ref_many2many_target_list)
    
    
    num_source2null_ref_ratio_mean = np.mean(num_source2null_ref_ratio_list)
    num_target2null_ref_ratio_mean = np.mean(num_target2null_ref_ratio_list)
    
    
    values = [num_word_source, num_word_target,
              total_num_link,
              num_sure, num_fuzzy,
              
              num_align_ref, num_no_ref, num_no_ref_no, num_no_ref_null,

              num_align_ref_one2one, 
              num_align_ref_one2many_source, num_align_ref_one2many_target,
              num_align_ref_many2one_source, num_align_ref_many2one_target,
              num_align_ref_many2many, num_align_ref_many2many_source, num_align_ref_many2many_target,

              total_null_link, num_source2null_ref, num_target2null_ref, num_source2notnull_ref, num_target2notnull_ref,
              np.round(num_source2null_ref_ratio_mean, round_value), np.round(num_target2null_ref_ratio_mean, round_value)]
    
    return values

def print_analysis_reference(values):
    print('----------------------------')
    print('Source words J: ', values[0] )
    print('Target words I: ', values[1] )
    print('All possible links I*J: ', values[2])
    print('Sure links: ', values[3])
    print('Fuzzy links: ', values[4])
    print('----------------------------')
    print()
    print('----------------------------')
    print('REFERENCE:')
    print('Alignment link: ', values[5])
    print('Non-exsisting links: ', values[6], end='')
    print(', including Null links: ', values[8])
    print('One-to-one links: ', values[9])
    print('One-to-many links: ', values[10],'-',values[11])
    print('Many-to-one links: ', values[12],'-',values[13])
    print('Many-to-many links: ', values[15],'-',values[16],'(number of links: ', values[14] ,')' )
    print()
    print('Words I+J: ', values[17])
    print('Aligned source words: ', values[20])
    print('Aligned target words: ', values[21])
    print('Unaligned source words: ', values[18], ' with average ratio ', values[22])
    print('Unaligned target words: ', values[19], ' with average ratio', values[23])
    print('----------------------------')
    print()

def analyse_prediction(prediction_set,
                       reference_set,
                       source_set,
                       target_set):

    # Count for Prediction
    num_align_pred = 0
    num_no_pred = 0
    num_align_pred_one2one = 0
    num_align_pred_one2many_source = 0
    num_align_pred_one2many_target = 0
    num_align_pred_many2one_source = 0
    num_align_pred_many2one_target = 0
    num_align_pred_many2many = 0
    num_align_pred_many2many_source = 0
    num_align_pred_many2many_target = 0

    num_no_pred_no = 0
    num_no_pred_null = 0
    
    #TP
    num_true_align_tp = 0
    
    num_true_align_tp_one2one_pred = 0
    num_true_align_tp_one2many_pred = 0
    num_true_align_tp_many2one_pred = 0
    num_true_align_tp_many2many_pred = 0
    
    #TN
    num_true_no_tn = 0
    num_true_no_tn_no_in_pred = 0
    num_true_no_tn_null_in_pred = 0
    
    #FN
    num_false_no_fn = 0
    num_false_no_fn_no_in_pred = 0
    num_false_no_fn_null_in_pred = 0
    
    #FP
    num_false_align_no_fp = 0
    num_false_align_no_fp_one2one_pred = 0
    num_false_align_no_fp_one2many_pred = 0
    num_false_align_no_fp_many2one_pred = 0
    num_false_align_no_fp_many2many_pred = 0

    num_false_align_no_fp_no_in_ref = 0
    num_false_align_no_fp_null_in_ref = 0

    # Null
    num_source2null_pred = 0
    num_target2null_pred = 0
    num_source2notnull_pred = 0
    num_target2notnull_pred = 0
    
    num_source2null_pred_tp = 0
    num_source2null_pred_fp = 0
    num_source2null_pred_tn = 0
    num_source2null_pred_fn = 0
    
    num_target2null_pred_tp = 0
    num_target2null_pred_fp = 0
    num_target2null_pred_tn = 0
    num_target2null_pred_fn = 0
    
    num_source2null_pred_ratio_list = []
    num_target2null_pred_ratio_list = []
    
    num_true_null_tp = 0
    num_false_not_null_fp = 0
    num_false_null_fn = 0
    num_true_not_null_tn = 0
    
    for idx, [source, target, pred, ref] in enumerate(zip(source_set, target_set,
                                                                      prediction_set, reference_set)):
        
        source_len = len(source)
        target_len = len(target)

        #---------------------------
        # Null
        num_source2null_pred_sent = 0
        num_target2null_pred_sent = 0
        
        for idx_source in range(1, source_len+1):
            source_to_null_ref = True
            source_to_null_pred = True
            
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    source_to_null_ref = False
                if align_check in pred:
                    source_to_null_pred = False
                    
                
            if source_to_null_pred:
                num_source2null_pred+=1
                num_source2null_pred_sent+=1
                
            else:
                num_source2notnull_pred+=1
                
            if source_to_null_ref and source_to_null_pred:
                num_true_null_tp+=1
                num_source2null_pred_tp+=1
                
            if not source_to_null_ref and not source_to_null_pred:
                num_true_not_null_tn+=1  
                num_source2null_pred_tn+=1
                
            if source_to_null_ref and not source_to_null_pred:
                num_false_null_fn+=1
                num_source2null_pred_fn+=1
                
            if not source_to_null_ref and source_to_null_pred:
                num_false_not_null_fp+=1
                num_source2null_pred_fp+=1
                
            
        for idx_target in range(1, target_len+1):
            target_to_null_ref = True
            target_to_null_pred = True
            
            for idx_source in range(1, source_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    target_to_null_ref = False
                if align_check in pred:
                    target_to_null_pred = False
                
            if target_to_null_pred:
                num_target2null_pred+=1
                num_target2null_pred_sent+=1
            else:
                num_target2notnull_pred+=1
                
            if target_to_null_ref and target_to_null_pred:
                num_true_null_tp+=1
                num_target2null_pred_tp+=1
            if not target_to_null_ref and not target_to_null_pred:
                num_true_not_null_tn+=1  
                num_target2null_pred_tn+=1
            if target_to_null_ref and not target_to_null_pred:
                num_false_null_fn+=1
                num_target2null_pred_fn+=1
            if not target_to_null_ref and target_to_null_pred:
                num_false_not_null_fp+=1
                num_target2null_pred_fp+=1
                
                
        num_source2null_pred_ratio_list.append(num_source2null_pred_sent/source_len)
        num_target2null_pred_ratio_list.append(num_target2null_pred_sent/target_len)
        
        
        num_align_pred_one2many_source_list = []
        num_align_pred_many2one_target_list = []
        num_align_pred_many2many_source_list = []
        num_align_pred_many2many_target_list = []
        
        align_pred_one2one_list = []
        align_pred_one2many_list = []
        align_pred_many2one_list = []
        align_pred_many2many_list = []
        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)

                # Count number of links in Prediction
                if align_check in pred:
                    num_align_pred +=1
                    
                    check_one2many = False
                    check_many2one = False
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred and align_check_ != align_check:
                            check_one2many = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred and align_check_ != align_check:
                            check_many2one = True
                    
                    if check_one2many is True and check_many2one is False:
                        num_align_pred_one2many_target +=1
                        align_pred_one2many_list.append(align_check)
                        if idx_source not in num_align_pred_one2many_source_list:
                            num_align_pred_one2many_source_list.append(idx_source)
                    if check_many2one is True and check_one2many is False:
                        num_align_pred_many2one_source +=1
                        align_pred_many2one_list.append(align_check)
                        if idx_target not in num_align_pred_many2one_target_list:
                            num_align_pred_many2one_target_list.append(idx_target)
                    if check_one2many is False and check_many2one is False:
                        num_align_pred_one2one +=1
                        align_pred_one2one_list.append(align_check)
                
                # Count number of links not in Prediction
                if align_check not in pred:
                    num_no_pred+=1
                    
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred:
                            target_to_null = False
                            
                    if source_to_null and target_to_null:
                        num_no_pred_null +=1
                    else:
                        num_no_pred_no +=1
                        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                
                # Count number of many2many links in Prediction
                if align_check in pred:
                    
                    check_one2many = False
                    check_many2one = False
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred and align_check_ != align_check:
                            check_one2many = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred and align_check_ != align_check:
                            check_many2one = True
                            
                    if check_one2many is True and check_many2one is True:
                        if idx_source not in num_align_pred_many2many_source_list \
                        and idx_source not in num_align_pred_one2many_source_list:
                            num_align_pred_many2many_source_list.append(idx_source)
                        if idx_target not in num_align_pred_many2many_target_list \
                        and idx_target not in num_align_pred_many2one_target_list:
                            num_align_pred_many2many_target_list.append(idx_target)
                        num_align_pred_many2many+=1
                        align_pred_many2many_list.append(align_check)
                        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                
                    
                # Count number of links in Prediction and in Ref: TP
                if align_check in pred and align_check in ref:
                    num_true_align_tp +=1
                    
                    if align_check in align_pred_one2one_list:
                        num_true_align_tp_one2one_pred+=1
                    if align_check in align_pred_one2many_list:
                        num_true_align_tp_one2many_pred+=1
                    if align_check in align_pred_many2one_list:
                        num_true_align_tp_many2one_pred+=1
                    if align_check in align_pred_many2many_list:
                        num_true_align_tp_many2many_pred+=1
                
                        
                # Count number of links not in Prediction and not in Ref: TN
                if align_check not in pred and align_check not in ref:
                    num_true_no_tn+=1
                    
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred:
                            target_to_null = False
                            
                    if source_to_null and target_to_null:
                        num_true_no_tn_null_in_pred +=1
                    else:
                        num_true_no_tn_no_in_pred +=1
                    
                # Count number of links in Prediction and not in Ref: FP
                if align_check in pred and align_check not in ref:
                    num_false_align_no_fp+=1
                    
                    if align_check in align_pred_one2one_list:
                        num_false_align_no_fp_one2one_pred+=1
                    if align_check in align_pred_one2many_list:
                        num_false_align_no_fp_one2many_pred+=1
                    if align_check in align_pred_many2one_list:
                        num_false_align_no_fp_many2one_pred+=1
                    if align_check in align_pred_many2many_list:
                        num_false_align_no_fp_many2many_pred+=1
                        
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in ref:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in ref:
                            target_to_null = False
                    
                    if source_to_null and target_to_null:
                        num_false_align_no_fp_null_in_ref +=1
                    else:
                        num_false_align_no_fp_no_in_ref +=1
                        
                # Count number of links not in Prediction and in Ref: FN
                if align_check not in pred and align_check in ref:
                    num_false_no_fn+=1
                    
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred:
                            target_to_null = False
                    
                    if source_to_null and target_to_null:
                        num_false_no_fn_null_in_pred +=1
                    else:
                        num_false_no_fn_no_in_pred +=1  
        
        num_align_pred_one2many_source += len(num_align_pred_one2many_source_list)
        num_align_pred_many2one_target += len(num_align_pred_many2one_target_list)
        
        num_align_pred_many2many_source += len(num_align_pred_many2many_source_list)
        num_align_pred_many2many_target += len(num_align_pred_many2many_target_list)
    
    num_source2null_pred_ratio_mean = np.mean(num_source2null_pred_ratio_list)
    num_target2null_pred_ratio_mean = np.mean(num_target2null_pred_ratio_list)
        
    acc, precision, recall, f1 = calculate_scores(num_true_align_tp, num_false_align_no_fp, num_true_no_tn, num_false_no_fn)
    
    null_acc, null_precision, null_recall, null_f1 = calculate_scores(num_true_null_tp, num_false_not_null_fp, num_true_not_null_tn, num_false_null_fn)
    
    values = [num_align_pred, num_no_pred, num_no_pred_no, num_no_pred_null,

             num_true_align_tp, 
             num_false_align_no_fp, num_false_align_no_fp_no_in_ref, num_false_align_no_fp_null_in_ref,
             num_false_no_fn, num_false_no_fn_no_in_pred, num_false_no_fn_null_in_pred,
             num_true_no_tn, num_true_no_tn_no_in_pred, num_true_no_tn_null_in_pred,
             
             acc, precision, recall, f1,
             
             num_align_pred_one2one,
             num_align_pred_one2many_source, num_align_pred_one2many_target,
             num_align_pred_many2one_source, num_align_pred_many2one_target,
             num_align_pred_many2many, num_align_pred_many2many_source, num_align_pred_many2many_target,
             
             num_true_align_tp_one2one_pred, num_true_align_tp_one2many_pred,
             num_true_align_tp_many2one_pred, num_true_align_tp_many2many_pred,
             num_false_align_no_fp_one2one_pred, num_false_align_no_fp_one2many_pred,
             num_false_align_no_fp_many2one_pred, num_false_align_no_fp_many2many_pred,
             
             num_source2null_pred, num_target2null_pred, num_source2notnull_pred, num_target2notnull_pred,
             
             num_source2null_pred_tp, num_source2null_pred_fp, num_source2null_pred_fn, num_source2null_pred_tn,
             num_target2null_pred_tp,num_target2null_pred_fp, num_target2null_pred_fn, num_target2null_pred_tn,
             
             np.round(num_source2null_pred_ratio_mean, round_value), np.round(num_target2null_pred_ratio_mean, round_value),
             num_true_null_tp, num_false_not_null_fp, num_false_null_fn, num_true_not_null_tn,
             
             null_acc, null_precision, null_recall, null_f1, round_value]

    return values


def print_analysis_prediction(values, aer):
    print('----------------------------')
    print('PREDICTION:')
    print('Alignment links: ', values[0])
    print('Non-exsisting links: ', values[1], end='')
    print(', including Null links: ', values[3])
    print()
    print('TP: ', values[4], '; FP: ', values[5], '; FN: ', values[8], '; TN: ', values[11])
    print('; Accuracy: ', values[14], '; Precision: ', values[15], '; Recall: ', values[16], '; F1: ', values[17])
    print('; AER: ', aer)
    print()
    print('One-to-one links: ', values[18], '; Correct: ', values[26])
    print('One-to-many links: ', values[19],'-',values[20], '; Correct: ', values[27])
    print('Many-to-one links: ', values[21],'-',values[22], '; Correct: ', values[28])
    print('Many-to-many links: ', values[24],'-',values[25],'(number of links: ', values[23] ,')', '; Correct: ', values[29])
    print()
    print('Aligned source words: ', values[36])
    print('Aligned target words: ', values[37])
    print('Unaligned source words: ', values[34], ' with average ratio ', values[46])
    print('; with TP: ', values[38], '; FP: ', values[39], '; FN: ', values[40], '; TN: ', values[41])
    print('Unaligned target words: ', values[35], ' with average ratio ', values[47])
    print('; with TP: ', values[42], '; FP: ', values[43], '; FN: ', values[44], '; TN: ', values[45])
    print()
    print('Null links with TP: ', values[48], '; FP: ', values[49], '; FN: ', values[50], '; TN: ', values[51])
    print('; Accuracy: ', values[52], '; Precision: ', values[53], '; Recall: ', values[54], '; F1: ', values[55])
    print('----------------------------')
    print()
    
#========================================
# FREQUENCY ANALYSIS
#========================================
def check_freq(word, vocab_freq, freq_max=1):
    if word not in vocab_freq.keys():
        return False
    else:
        if vocab_freq[word] <= freq_max:
            return True
        else:
            return False
        
def check_unk(word, vocab_freq):
    if word not in vocab_freq.keys():
        return True
    else:
        return False
    
def get_vocabulary_dict_freq(vocabulary):
    vocabulary = open(vocabulary, 'r', encoding='utf-8')
    dict_vocab = {}
    for line in vocabulary:
        word, freq = line.split(' ')
        dict_vocab[word] = int(freq.replace('\n', ''))
    return dict_vocab
    
def analyse_reference_frequence(reference_set,
                                source_set,
                                target_set,
                                vocabulary_freq_source,
                                vocabulary_freq_target,
                                freq_max):
    
    # Frequency
    num_align_ref_rare2many_list = []
    num_align_ref_many2rare_list = []
    num_align_ref_rare2many_source = 0
    num_align_ref_rare2many_target = 0
    num_align_ref_many2rare_source = 0
    num_align_ref_many2rare_target = 0
    
    # Frequency UNK
    num_align_ref_unk2many_list = []
    num_align_ref_many2unk_list = []
    num_align_ref_unk2many_source = 0
    num_align_ref_unk2many_target = 0
    num_align_ref_many2unk_source = 0
    num_align_ref_many2unk_target = 0
    
    for idx, [source, target, ref] in enumerate(zip(source_set, target_set, reference_set)):
        
        source_len = len(source)
        target_len = len(target)

        #---------------------------
        # Frequency
        for idx_source in range(1, source_len+1):
            if check_freq(source[idx_source-1], vocabulary_freq_source, freq_max):
                num_rare_2_target_word_ref= 0
                num_align_ref_rare2many_source += 1
                for idx_target in range(1, target_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                        
                    if align_check in ref:
                        num_rare_2_target_word_ref+=1
                        num_align_ref_rare2many_target+=1
                        
                num_align_ref_rare2many_list.append(num_rare_2_target_word_ref)
                
        for idx_target in range(1, target_len+1):
            if check_freq(target[idx_target-1], vocabulary_freq_target, freq_max):
                num_rare_2_source_word_ref = 0
                num_align_ref_many2rare_target+=1
                for idx_source in range(1, source_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in ref:
                        num_rare_2_source_word_ref+=1
                        num_align_ref_many2rare_source+=1
                        
                num_align_ref_many2rare_list.append(num_rare_2_source_word_ref)
        
        #---------------------------
        # Frequency UNK
        for idx_source in range(1, source_len+1):
            if check_unk(source[idx_source-1], vocabulary_freq_source):
                num_unk_2_target_word_ref= 0
                num_align_ref_unk2many_source += 1
                for idx_target in range(1, target_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                        
                    if align_check in ref:
                        num_unk_2_target_word_ref+=1
                        num_align_ref_unk2many_target+=1
                        
                num_align_ref_unk2many_list.append(num_unk_2_target_word_ref)
                
        
        for idx_target in range(1, target_len+1):
            if check_unk(target[idx_target-1], vocabulary_freq_target):
                num_unk_2_source_word_ref = 0
                num_align_ref_many2unk_target+=1
                for idx_source in range(1, source_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in ref:
                        num_unk_2_source_word_ref+=1
                        num_align_ref_many2unk_source+=1
                        
                num_align_ref_many2unk_list.append(num_unk_2_source_word_ref)
    
    if len(num_align_ref_rare2many_list) != 0:
        num_align_ref_rare2many_ratio_mean = np.mean(num_align_ref_rare2many_list)
    if len(num_align_ref_many2rare_list) != 0:
        num_align_ref_many2rare_ratio_mean = np.mean(num_align_ref_many2rare_list) 
    
    
    if len(num_align_ref_unk2many_list) != 0:
        num_align_ref_unk2many_ratio_mean = np.mean(num_align_ref_unk2many_list)
    if len(num_align_ref_many2unk_list) != 0:
        num_align_ref_many2unk_ratio_mean = np.mean(num_align_ref_many2unk_list) 
    
            

    values = [num_align_ref_rare2many_source, num_align_ref_rare2many_target,
              np.round(num_align_ref_rare2many_ratio_mean, round_value),
              num_align_ref_many2rare_source, num_align_ref_many2rare_target,
              np.round(num_align_ref_many2rare_ratio_mean, round_value),

              num_align_ref_unk2many_source, num_align_ref_unk2many_target,
              np.round(num_align_ref_unk2many_ratio_mean, round_value),
              num_align_ref_many2unk_source, num_align_ref_many2unk_target,
              np.round(num_align_ref_many2unk_ratio_mean, round_value)]
               
    return values

def print_analysis_reference_frequence(values):
    print('----------------------------')
    print('REFERENCE - FREQUENCY:')
    print('Rare-to-Target: ', values[0], '-', values[1], ' with ratio 1 -', values[2])
    print('Source-to-Rare: ', values[3], '-', values[4], ' with ratio ', values[5], '- 1')
    print('Unk-to-Target: ', values[0], '-', values[1], ' with ratio 1 -', values[2])
    print('Source-to-Unk: ', values[3], '-', values[4], ' with ratio ', values[5], '- 1')
    print('----------------------------')
    print()


def analyse_prediction_frequence(prediction_set,
                                 reference_set,
                                 source_set,
                                 target_set,
                                 vocabulary_freq_source,
                                 vocabulary_freq_target,
                                 freq_max):
    
    # Frequency
    
    num_align_model_rare2many_list = []
    num_align_model_many2rare_list = []
    num_align_model_rare2many_source = 0
    num_align_model_rare2many_target = 0
    num_align_model_many2rare_source = 0
    num_align_model_many2rare_target = 0
    
    num_align_model_rare2many_tp_list = []
    num_align_model_rare2many_tp = 0
    num_align_model_rare2many_fp = 0
    num_align_model_rare2many_fn = 0
    num_align_model_rare2many_tn = 0
    
    num_align_model_many2rare_tp_list = []
    num_align_model_many2rare_tp = 0
    num_align_model_many2rare_fp = 0
    num_align_model_many2rare_fn = 0
    num_align_model_many2rare_tn = 0
    
    # Frequency UNK
    
    num_align_model_unk2many_list = []
    num_align_model_many2unk_list = []
    num_align_model_unk2many_source = 0
    num_align_model_unk2many_target = 0
    num_align_model_many2unk_source = 0
    num_align_model_many2unk_target = 0
    
    num_align_model_unk2many_tp_list = []
    num_align_model_unk2many_tp = 0
    num_align_model_unk2many_fp = 0
    num_align_model_unk2many_fn = 0
    num_align_model_unk2many_tn = 0
    
    num_align_model_many2unk_tp_list = []
    num_align_model_many2unk_tp = 0
    num_align_model_many2unk_fp = 0
    num_align_model_many2unk_fn = 0
    num_align_model_many2unk_tn = 0
    
    
    for idx, [source, target, pred, ref] in enumerate(zip(source_set, target_set,
                                                          prediction_set, reference_set)):
        
        source_len = len(source)
        target_len = len(target)

        #---------------------------
        # Frequency
        for idx_source in range(1, source_len+1):
            if check_freq(source[idx_source-1], vocabulary_freq_source, freq_max):
                num_rare_2_target_word_model = 0
                num_rare_2_target_word_model_tp = 0
                num_align_model_rare2many_source += 1
                for idx_target in range(1, target_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in pred:
                        num_rare_2_target_word_model+=1
                        num_align_model_rare2many_target+=1
                        
                        
                    if align_check in pred and align_check in ref:
                        num_rare_2_target_word_model_tp+=1
                        num_align_model_rare2many_tp+=1
                    if align_check in pred and align_check not in ref:
                        num_align_model_rare2many_fp+=1
                    if align_check not in pred and align_check in ref:
                        num_align_model_rare2many_fn+=1
                    if align_check not in pred and align_check not in ref:
                        num_align_model_rare2many_tn+=1
                        
                    
                num_align_model_rare2many_list.append(num_rare_2_target_word_model)
                num_align_model_rare2many_tp_list.append(num_rare_2_target_word_model_tp)
                
        
        for idx_target in range(1, target_len+1):
            if check_freq(target[idx_target-1], vocabulary_freq_target, freq_max):
                num_rare_2_source_word_model = 0
                num_rare_2_source_word_model_tp = 0
                num_align_model_many2rare_target+=1
                for idx_source in range(1, source_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in pred:
                        num_rare_2_source_word_model+=1
                        num_align_model_many2rare_source+=1
                        
                    if align_check in pred and align_check in ref:
                        num_rare_2_source_word_model_tp+=1
                        num_align_model_many2rare_tp+=1
                    if align_check in pred and align_check not in ref:
                        num_align_model_many2rare_fp+=1
                    if align_check not in pred and align_check in ref:
                        num_align_model_many2rare_fn+=1
                    if align_check not in pred and align_check not in ref:
                        num_align_model_many2rare_tn+=1
                        
                num_align_model_many2rare_list.append(num_rare_2_source_word_model)
                num_align_model_many2rare_tp_list.append(num_rare_2_source_word_model_tp)
        
        #---------------------------
        # Frequency UNK
        for idx_source in range(1, source_len+1):
            if check_unk(source[idx_source-1], vocabulary_freq_source):
                num_unk_2_target_word_model = 0
                num_unk_2_target_word_model_tp = 0
                num_align_model_unk2many_source += 1
                for idx_target in range(1, target_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in pred:
                        num_unk_2_target_word_model+=1
                        num_align_model_unk2many_target+=1
                        
                        
                    if align_check in pred and align_check in ref:
                        num_unk_2_target_word_model_tp+=1
                        num_align_model_unk2many_tp+=1
                    if align_check in pred and align_check not in ref:
                        num_align_model_unk2many_fp+=1
                    if align_check not in pred and align_check in ref:
                        num_align_model_unk2many_fn+=1
                    if align_check not in pred and align_check not in ref:
                        num_align_model_unk2many_tn+=1
                        
                    
                num_align_model_unk2many_list.append(num_unk_2_target_word_model)
                num_align_model_unk2many_tp_list.append(num_unk_2_target_word_model_tp)
                
        
        for idx_target in range(1, target_len+1):
            if check_unk(target[idx_target-1], vocabulary_freq_target):
                num_unk_2_source_word_model = 0
                num_unk_2_source_word_model_tp = 0
                num_align_model_many2unk_target+=1
                
                for idx_source in range(1, source_len+1):
                    align_check = str(idx_source) +'-'+ str(idx_target)
                    
                    if align_check in pred:
                        num_unk_2_source_word_model+=1
                        num_align_model_many2unk_source+=1
                        
                    if align_check in pred and align_check in ref:
                        num_unk_2_source_word_model_tp+=1
                        num_align_model_many2unk_tp+=1
                    if align_check in pred and align_check not in ref:
                        num_align_model_many2unk_fp+=1
                    if align_check not in pred and align_check in ref:
                        num_align_model_many2unk_fn+=1
                    if align_check not in pred and align_check not in ref:
                        num_align_model_many2unk_tn+=1
                        
                num_align_model_many2unk_list.append(num_unk_2_source_word_model)
                num_align_model_many2unk_tp_list.append(num_unk_2_source_word_model_tp)
        
    if len(num_align_model_rare2many_list) != 0:
        num_align_model_rare2many_ratio_mean = np.mean(num_align_model_rare2many_list)
        num_align_model_rare2many_tp_mean = np.mean(np.array(num_align_model_rare2many_list) - np.array(num_align_model_rare2many_tp_list))
    if len(num_align_model_many2rare_list) != 0:
        num_align_model_many2rare_ratio_mean = np.mean(num_align_model_many2rare_list) 
        num_align_model_many2rare_tp_mean = np.mean(np.array(num_align_model_many2rare_list) - np.array(num_align_model_many2rare_tp_list))
    
    
    if len(num_align_model_unk2many_list) != 0:
        num_align_model_unk2many_ratio_mean = np.mean(num_align_model_unk2many_list)
        num_align_model_unk2many_tp_mean = np.mean(np.array(num_align_model_unk2many_list) - np.array(num_align_model_unk2many_tp_list))
    if len(num_align_model_many2unk_list) != 0:
        num_align_model_many2unk_ratio_mean = np.mean(num_align_model_many2unk_list) 
        num_align_model_many2unk_tp_mean = np.mean(np.array(num_align_model_many2unk_list) - np.array(num_align_model_many2unk_tp_list))
            
    num_align_model_rare2many_acc, \
    num_align_model_rare2many_precision, \
    num_align_model_rare2many_recall, \
    num_align_model_rare2many_f1 = calculate_scores(num_align_model_rare2many_tp,
                                                 num_align_model_rare2many_fp,
                                                 num_align_model_rare2many_tn,
                                                 num_align_model_rare2many_fn)
    
    num_align_model_many2rare_acc, \
    num_align_model_many2rare_precision, \
    num_align_model_many2rare_recall, \
    num_align_model_many2rare_f1 = calculate_scores(num_align_model_many2rare_tp,
                                                 num_align_model_many2rare_fp,
                                                 num_align_model_many2rare_tn,
                                                 num_align_model_many2rare_fn)
    
    num_align_model_unk2many_acc, \
    num_align_model_unk2many_precision, \
    num_align_model_unk2many_recall, \
    num_align_model_unk2many_f1 = calculate_scores(num_align_model_unk2many_tp,
                                                 num_align_model_unk2many_fp,
                                                 num_align_model_unk2many_tn,
                                                 num_align_model_unk2many_fn)
    
    num_align_model_many2unk_acc, \
    num_align_model_many2unk_precision, \
    num_align_model_many2unk_recall, \
    num_align_model_many2unk_f1 = calculate_scores(num_align_model_many2unk_tp,
                                                 num_align_model_many2unk_fp,
                                                 num_align_model_many2unk_tn,
                                                 num_align_model_many2unk_fn)
    
    
    values = [num_align_model_rare2many_source, num_align_model_rare2many_target, 
               np.round(num_align_model_rare2many_ratio_mean, round_value),
               num_align_model_many2rare_source, num_align_model_many2rare_target, 
               np.round(num_align_model_many2rare_ratio_mean, round_value),
               
               
               num_align_model_rare2many_tp, num_align_model_rare2many_fp, num_align_model_rare2many_fn, num_align_model_rare2many_tn,
               np.round(num_align_model_rare2many_tp_mean, round_value),
               num_align_model_rare2many_acc, num_align_model_rare2many_precision, num_align_model_rare2many_recall, num_align_model_rare2many_f1 ,
               
               num_align_model_many2rare_tp, num_align_model_many2rare_fp, num_align_model_many2rare_fn, num_align_model_many2rare_tn,
               np.round(num_align_model_many2rare_tp_mean, round_value),
               num_align_model_many2rare_acc, num_align_model_many2rare_precision, num_align_model_many2rare_recall, num_align_model_many2rare_f1,
               
               num_align_model_unk2many_source, num_align_model_unk2many_target, 
               np.round(num_align_model_unk2many_ratio_mean, round_value),
               num_align_model_many2unk_source, num_align_model_many2unk_target, 
               np.round(num_align_model_many2unk_ratio_mean, round_value),
               
               
               num_align_model_unk2many_tp, num_align_model_unk2many_fp, num_align_model_unk2many_fn, num_align_model_unk2many_tn,
               np.round(num_align_model_unk2many_tp_mean, round_value),
               num_align_model_unk2many_acc, num_align_model_unk2many_precision, num_align_model_unk2many_recall, num_align_model_unk2many_f1,
               
               num_align_model_many2unk_tp, num_align_model_many2unk_fp, num_align_model_many2unk_fn, num_align_model_many2unk_tn,
               np.round(num_align_model_many2unk_tp_mean, round_value),
               num_align_model_many2unk_acc, num_align_model_many2unk_precision, num_align_model_many2unk_recall, num_align_model_many2unk_f1]

    return values
    
def print_analysis_prediction_frequence(values):
    print('----------------------------')
    print('PREDICTION - FREQUENCY:')
    print('Rare-to-Target: ', values[0], '-', values[1], ' with ratio 1 -', values[2])
    print('Source-to-Rare: ', values[3], '-', values[4], ' with ratio ', values[5], '- 1')
    print('Unk-to-Target: ', values[24], '-', values[25], ' with ratio 1 -', values[26])
    print('Source-to-Unk: ', values[27], '-', values[28], ' with ratio ', values[29], '- 1')
    print()
    print('Rare-to-Target links with TP: ', values[6], '; FP: ', values[7], '; FN: ', values[8], '; TN: ', values[9])
    print('; Accuracy: ', values[11], '; Precision: ', values[12], '; Recall: ', values[13], '; F1: ', values[14])
    print('Source-to-Rare links with TP: ', values[15], '; FP: ', values[16], '; FN: ', values[17], '; TN: ', values[18])
    print('; Accuracy: ', values[20], '; Precision: ', values[21], '; Recall: ', values[22], '; F1: ', values[23])
    print()
    print('Unk-to-Target links with TP: ', values[30], '; FP: ', values[31], '; FN: ', values[32], '; TN: ', values[33])
    print('; Accuracy: ', values[35], '; Precision: ', values[36], '; Recall: ', values[37], '; F1: ', values[38])
    print('Source-to-Unk links with TP: ', values[39], '; FP: ', values[40], '; FN: ', values[41], '; TN: ', values[42])
    print('; Accuracy: ', values[44], '; Precision: ', values[45], '; Recall: ', values[46], '; F1: ', values[47])
    print('----------------------------')
    print()
    
#========================================
# PART-OF-SPEECH ANALYSIS
#========================================

pos_tag = list(['PUNCT', 'SYM', 'NUM', 'NOUN', 'PROPN', 
                'VERB','AUX', 'ADJ', 'ADV', 'PRON', 'CONJ', 
                'SCONJ' , 'DET', 'INTJ', 'ADP', 'PART', 'X'])
# Based on Spacy POS

def check_pos_content_word(pos):
    pos_content_tag = list(['NOUN','VERB','ADJ', 'ADV'])
    if pos in pos_content_tag:
        return True
    else:
        return False
    
def increase_number_pos_function_and_content_word(source_word, target_word, num_content_function_list):
    num_content_function_list_ = num_content_function_list
    if source_word != 'NULL':
        if check_pos_content_word(source_word):
            num_content_function_list_[0]+=1
        else:
            num_content_function_list_[1]+=1
            
    if target_word != 'NULL':
        if check_pos_content_word(target_word):
            num_content_function_list_[2]+=1
        else:
            num_content_function_list_[3]+=1
        
    return num_content_function_list_

def analyse_reference_pos(reference_set,
                          source_pos_set,
                          target_pos_set):
    
    # POS [Source Content word, Source Function word, Target Content word, Target Function word]
    num_align_ref_pos = [0, 0, 0, 0]
    num_no_ref_pos = [0, 0, 0, 0]
    
    num_no_ref_no_pos = [0, 0, 0, 0]
    num_no_ref_null_pos = [0, 0, 0, 0]
    
    num_2notnull_ref_pos = [0, 0, 0, 0]
    
    num_source2null_ref_pos = [0, 0, 0, 0]
    num_target2null_ref_pos = [0, 0, 0, 0]
    
    for idx, [source_pos, target_pos, ref] in enumerate(zip(source_pos_set, target_pos_set, reference_set)):
        
        source_len = len(source_pos)
        target_len = len(target_pos)

        #---------------------------
        # POS
        
        for idx_source in range(1, source_len+1):
            source_to_null_ref = True
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    source_to_null_ref = False
                    
            if source_to_null_ref:
                num_source2null_ref_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                        'NULL',
                                                                                        num_source2null_ref_pos)
            else:
                num_2notnull_ref_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                        'NULL',
                                                                                        num_2notnull_ref_pos)
                            
        for idx_target in range(1, target_len+1):
            target_to_null_ref = True
            for idx_source in range(1, source_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    target_to_null_ref = False
                    
            if target_to_null_ref:
                num_target2null_ref_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_target2null_ref_pos)
            else:
                num_2notnull_ref_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                     target_pos[idx_target-1],
                                                                                     num_2notnull_ref_pos)
                
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                # Count number of links in Ref
                if align_check in ref:
                    num_align_ref_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                      target_pos[idx_target-1],
                                                                                      num_align_ref_pos)

                # Count number of links not in Ref
                if align_check not in ref:
                    num_no_ref_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_ref_pos)
                    
                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in ref:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in ref:
                            target_to_null = False
                            
                    if source_to_null and target_to_null:
                        num_no_ref_null_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_ref_null_pos)
                        
                    else:
                        num_no_ref_no_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_ref_no_pos)

    values = [num_align_ref_pos, 
              num_no_ref_pos, 
              num_no_ref_no_pos, 
              num_no_ref_null_pos,
              
              num_source2null_ref_pos, 
              num_target2null_ref_pos]
               
    return values   
    
    
def print_analysis_reference_pos(values):
    # POS [Source Content word, Source Function word, Target Content word, Target Function word]
    print('----------------------------')
    print('REFERENCE - PART OF SPEECH (Content and function words):')
    print('Aligned source: Content: ', values[0][0], '; Function: ', values[0][1])
    print('Aligned target: Content: ', values[0][2], '; Function: ', values[0][3])
    print()
    print('Unaligned source: Content: ', values[4][0], '; Function: ', values[4][1])
    print('Unaligned target: Content: ', values[5][2], '; Function: ', values[5][3])
    print('----------------------------')
    print()
    
    
def analyse_prediction_pos(prediction_set,
                           reference_set,
                           source_pos_set,
                           target_pos_set):
    
    # POS [Source Content word, Source Function word, Target Content word, Target Function word]
    num_align_model_pos = [0, 0, 0, 0]
    num_no_model_pos = [0, 0, 0, 0]
    
    num_no_model_no_pos = [0, 0, 0, 0]
    num_no_model_null_pos = [0, 0, 0, 0]
    
    num_true_align_tp_pos = [0, 0, 0, 0]
    num_true_no_tn_pos = [0, 0, 0, 0]
    num_false_no_fn_pos = [0, 0, 0, 0]
    num_false_align_no_fp_pos = [0, 0, 0, 0]
    
    num_source2null_model_pos = [0, 0, 0, 0]
    num_target2null_model_pos = [0, 0, 0, 0]
    num_2notnull_model_pos = [0, 0, 0, 0]
    
    num_true_null_tp_pos = [0, 0, 0, 0]
    num_false_not_null_fp_pos = [0, 0, 0, 0]
    num_false_null_fn_pos = [0, 0, 0, 0]
    num_true_not_null_tn_pos = [0, 0, 0, 0]
    
    for idx, [source_pos, target_pos, pred, ref] in enumerate(zip(source_pos_set, target_pos_set, 
                                                                  prediction_set, reference_set)):
        
        source_len = len(source_pos)
        target_len = len(target_pos) 

        #---------------------------
        # POS
        
        for idx_source in range(1, source_len+1):
            source_to_null_ref = True
            source_to_null_model = True
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                    source_to_null_ref = False
                if align_check in pred:
                    source_to_null_model = False

            if source_to_null_model:
                num_source2null_model_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                        'NULL',
                                                                                        num_source2null_model_pos)
            else:
                num_2notnull_model_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                        'NULL',
                                                                                        num_2notnull_model_pos)
                
            if source_to_null_ref and source_to_null_model:
                num_true_null_tp_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                     'NULL',
                                                                                     num_true_null_tp_pos)
            if not source_to_null_ref and not source_to_null_model:
                num_true_not_null_tn_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                     'NULL',
                                                                                     num_true_not_null_tn_pos)
            if source_to_null_ref and not source_to_null_model:
                num_false_null_fn_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                     'NULL',
                                                                                     num_false_null_fn_pos)
            if not source_to_null_ref and source_to_null_model:
                num_false_not_null_fp_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                     'NULL',
                                                                                     num_false_not_null_fp_pos)
            
        for idx_target in range(1, target_len+1):
            target_to_null_ref = True
            target_to_null_model = True
            for idx_source in range(1, source_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                if align_check in ref:
                        source_to_null_ref = False
                if align_check in pred:
                    target_to_null_model = False
                    
                
            if target_to_null_model:
                num_target2null_model_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_target2null_model_pos)
            else:
                num_2notnull_model_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                     target_pos[idx_target-1],
                                                                                     num_2notnull_model_pos)
                
            if target_to_null_ref and target_to_null_model:
                num_true_null_tp_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_true_null_tp_pos)
            if not target_to_null_ref and not target_to_null_model:
                num_true_not_null_tn_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_true_not_null_tn_pos)
            if target_to_null_ref and not target_to_null_model:
                num_false_null_fn_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_false_null_fn_pos)
            if not target_to_null_ref and target_to_null_model:
                num_false_not_null_fp_pos = increase_number_pos_function_and_content_word('NULL',
                                                                                        target_pos[idx_target-1],
                                                                                        num_false_not_null_fp_pos)
        
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                
                # Count number of links in Prediction
                if align_check in pred:
                    num_align_model_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_align_model_pos)

                # Count number of links not in Prediction
                if align_check not in pred:
                    num_no_model_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_model_pos)

                    source_to_null = True
                    for idx_target_ in range(1, target_len+1):
                        align_check_ = str(idx_source) +'-'+ str(idx_target_)
                        if align_check_ in pred:
                            source_to_null = False
                    
                    target_to_null = True
                    for idx_source_ in range(1, source_len+1):
                        align_check_ = str(idx_source_) +'-'+ str(idx_target)
                        if align_check_ in pred:
                            target_to_null = False
                                
                    if source_to_null and target_to_null:
                        num_no_model_null_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_model_null_pos)
                    else:
                        num_no_model_no_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_no_model_no_pos)
    
        for idx_source in range(1, source_len+1):
            for idx_target in range(1, target_len+1):
                align_check = str(idx_source) +'-'+ str(idx_target)
                
                    
                # Count number of links in Prediction and in Ref: TP
                if align_check in pred and align_check in ref:
                    num_true_align_tp_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_true_align_tp_pos)
                    

                
                        
                # Count number of links not in Prediction and not in Ref: TN
                if align_check not in pred and align_check not in ref:
                    num_true_no_tn_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_true_no_tn_pos)
                    
                    
                    
                # Count number of links in Prediction and not in Ref: FP
                if align_check in pred and align_check not in ref:
                    num_false_align_no_fp_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_false_align_no_fp_pos)
                    
                    
                        
                # Count number of links not in Prediction and in Ref: FN
                if align_check not in pred and align_check in ref:
                    num_false_no_fn_pos = increase_number_pos_function_and_content_word(source_pos[idx_source-1],
                                                                                   target_pos[idx_target-1],
                                                                                   num_false_no_fn_pos)
                    
                            
    values = [num_align_model_pos, 
              num_no_model_pos, 
              num_no_model_no_pos, 
              num_no_model_null_pos,
              num_true_align_tp_pos, num_false_align_no_fp_pos, num_false_no_fn_pos, num_true_no_tn_pos,
              
              num_source2null_model_pos, num_target2null_model_pos,
              num_true_null_tp_pos, num_false_not_null_fp_pos, num_false_null_fn_pos, num_true_not_null_tn_pos]
               
    return values   
    
    
def print_analysis_prediction_pos(values):
    # POS [Source Content word, Source Function word, Target Content word, Target Function word]
    print('----------------------------')
    print('PREDICTION - PART OF SPEECH (Content and function words):')
    print('Aligned source: Content: ', values[0][0], '; Function: ', values[0][1])
    print('Aligned target: Content: ', values[0][2], '; Function: ', values[0][3])
    print()
    print('Unaligned source: Content: ', values[8][0], '(correct: ', values[10][0],')', 
          '; Function: ', values[8][1], '(correct: ', values[10][1],')')
    print('Unaligned target: Content: ', values[9][2], '(correct: ', values[10][2],')' ,
          '; Function: ', values[9][3], '(correct: ', values[10][3],')')
    print()
    print('Alignment links:')
    print('TP: Source_Content: ', values[4][0], '; Source_Function: ', values[4][1], 
          '; Target_Content: ', values[4][2], '; Target_Function: ', values[4][3])
    print('FP: Source_Content: ', values[5][0], '; Source_Function: ', values[5][1], 
          '; Target_Content: ', values[5][2], '; Target_Function: ', values[5][3])
    print('FN: Source_Content: ', values[6][0], '; Source_Function: ', values[6][1], 
          '; Target_Content: ', values[6][2], '; Target_Function: ', values[6][3])
    print('TN: Source_Content: ', values[7][0], '; Source_Function: ', values[7][1], 
          '; Target_Content: ', values[7][2], '; Target_Function: ', values[7][3])
    print('----------------------------')
    print()    
    
    
    
parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    
parser.add_argument('-r', '--reference', help="Reference links; each sentence has two lines: one for sure links and another for fuzzy links", type=str)
parser.add_argument('-s', '--source', help="Source corpus", type=str)
parser.add_argument('-t', '--target', help="Target corpus", type=str)
parser.add_argument('-p', '--prediction', default='', help="Predicted alignment links", type=str)

parser.add_argument('--show_only_aer', default=False, help="Only show AER")

parser.add_argument('--show_frequency', default=False, help="Show rare/unknown word analyses")
parser.add_argument('-fm','--frequency_max', default=10, help="Show rare/unknown word analyses", type=int)
parser.add_argument('-fs', '--frequency_source', help="Frequency for source words, run build_word_frequency.py train_corpus_directory", type=str)
parser.add_argument('-ft', '--frequency_target', help="Frequency for target words, run build_word_frequency.py train_corpus_directory", type=str)

parser.add_argument('--show_part_of_speech', default=False, help="Show part-of-speech analyses")
parser.add_argument('-ps', '--pos_source', help="POS for source words, this file is created by replace each word by its POS", type=str)
parser.add_argument('-pt', '--pos_target', help="POS for target words, this file is created by replace each word by its POS", type=str)

args = parser.parse_args()

reference_set, sure_set, fuzzy_set = get_reference(args.reference, ref_only=False)

source_set = get_corpus_file(args.source)
target_set = get_corpus_file(args.target)

if args.show_frequency:
    vocabulary_freq_source = get_vocabulary_dict_freq(args.frequency_source)
    vocabulary_freq_target = get_vocabulary_dict_freq(args.frequency_target)
    
if args.show_part_of_speech:
    source_pos_set = get_corpus_file(args.pos_source)
    target_pos_set = get_corpus_file(args.pos_target)

print()
print("WORD ALIGNMENT ANALYSES")
print('============================')
print('Source sentence with J words')
print('Target sentence with I words')
print('============================')
print()

if args.show_only_aer:
    aer = calculate_AER(args.prediction, args.reference)
    print('AER: ', aer)
else:
    if args.prediction == '':
        ref_results = analyse_reference(reference_set, sure_set, fuzzy_set, source_set, target_set)
        print_analysis_reference(ref_results)
        
        if args.show_frequency:     
            ref_freq_results = analyse_reference_frequence(reference_set, source_set, target_set, 
                                                            vocabulary_freq_source, vocabulary_freq_target,
                                                            args.frequency_max)
            print_analysis_reference_frequence(ref_freq_results)
            
        if args.show_part_of_speech:
            ref_pos_results = analyse_reference_pos(reference_set,
                                            source_pos_set,
                                            target_pos_set)
    
            print_analysis_reference_pos(ref_pos_results)
            
    else:
        prediction_set = get_prediction(args.prediction)
        
        ref_results = analyse_reference(reference_set, sure_set, fuzzy_set, source_set, target_set)
        print_analysis_reference(ref_results)
            
        aer = calculate_AER(args.prediction, args.reference)
        pred_results = analyse_prediction(prediction_set, reference_set, source_set, target_set)
        print_analysis_prediction(pred_results, aer)
            
        if args.show_frequency:
            print('============================')
            print()
            ref_freq_results = analyse_reference_frequence(reference_set, source_set, target_set, 
                                                            vocabulary_freq_source, vocabulary_freq_target,
                                                            args.frequency_max)
            print_analysis_reference_frequence(ref_freq_results)
            
            pred_freq_results = analyse_prediction_frequence(prediction_set,
                                                                  reference_set, source_set, target_set, 
                                                                  vocabulary_freq_source, vocabulary_freq_target,
                                                                  args.frequency_max)
            print_analysis_prediction_frequence(pred_freq_results)
        
        if args.show_part_of_speech:
            print()
            print('============================')
            ref_pos_results = analyse_reference_pos(reference_set,
                                            source_pos_set,
                                            target_pos_set)
    
            print_analysis_reference_pos(ref_pos_results)
    
    
            pred_pos_results = analyse_prediction_pos(prediction_set, reference_set,
                                                          source_pos_set,target_pos_set)
    
            print_analysis_prediction_pos(pred_pos_results)

        