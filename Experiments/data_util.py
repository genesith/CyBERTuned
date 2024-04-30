from pdb import set_trace
from datasets import load_dataset
from consts import *
import os
from copy import deepcopy
from unidecode import unidecode
import string
from tqdm import tqdm
from pdb import set_trace

from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc 

def load_splits_suffix(ext, data_path, splits = ['train', 'dev', 'test']):
    ext_txt = ext
    if ext == 'text':
        ext_txt = 'txt'
    if ext == 'json':
        ext_txt = 'jsonl'
    splits_dict = {}
    for split in splits:
        splits_dict[split]= os.path.join(data_path,split+'.'+ext_txt)
    return load_dataset(ext, data_files=splits_dict)

def get_BIO_labels(task_name):
    raw_labels = task2labels[task_name]
    labels = ['O']
    for label in raw_labels:
        labels.append('B-'+label)
        labels.append('I-'+label)    
    return labels

#'Ġ' -> return False
# spcial chars and non-ascii -> return True
# Others -> return False 
def is_combine_char(char_): 
    if char_ == 'Ġ' or char_.isspace():
        return False

    is_spcial_char = not (char_.isalpha() or char_.isspace() or char_.isnumeric())
    if is_spcial_char or not char_.isascii(): 
        return True
    else:
        return False
'''
Sort spans and labels'''
def sort_spans_labels(spans, labels):
    # spans.sort()
    spans_, labels_ = [], []
    for s, l in sorted(zip(spans, labels), key=lambda x: x[0][0]): 
        spans_.append(s)
        labels_.append(l)

    return spans_, labels_

def adj_spans_after(pos, adj_len, spans):
    spans_ = []
    for span in spans: 
        if span[0] < pos: 
            spans_.append(span)
        else:
            spans_.append((span[0]+adj_len, span[1]+adj_len))

    return spans_

def tokenize_with_BIO_span(tokenizer, text, spans, labels):
    IS_WORDPIECE = False
    if tokenizer.tokenize("     ") == tokenizer.tokenize(" ") and tokenizer.tokenize("TEST") == tokenizer.tokenize("test"):
        IS_WORDPIECE = True
    spans, labels = sort_spans_labels(spans, labels)
    text_ = ""
    # convert non-ascii code and adjust span acoordingly 
    for i, c in enumerate(text):
        if not c.isascii():
            c_ = unidecode(c)
            if len(c) != len(c_):
                spans = adj_spans_after(i, len(c_)-len(c), spans)
            text_ += c_
        else:
            c_ = c
            text_ += c_

    # Get tokens 
    tokenized = tokenizer(text_, return_offsets_mapping=True, add_special_tokens=False)
    target_tokens_ids = tokenized.input_ids
    tok_pos_list = tokenized.offset_mapping

    
    # Get labels for each token
    t_labels = ['O' for tok in tok_pos_list]
    
    for span, label in zip(spans, labels):
        ss, se = span[0], span[1]
        was_inside = False
        for i, (ts, te) in enumerate(tok_pos_list):
            if (ts>=ss and ts<se) or (te>ss and te<=se):
                if not was_inside:
                    prefix = 'B-'
                else:
                    prefix = 'I-'
                t_labels[i] = prefix+label
                was_inside = True
            elif was_inside:
                break


    return target_tokens_ids, t_labels

def tokenize_with_BIO_span_old(tokenizer, text, spans, labels):
    
    IS_WORDPIECE = False
    if tokenizer.tokenize("     ") == tokenizer.tokenize(" "):
        IS_WORDPIECE = True

    target_tokens = tokenizer.tokenize(text)
    
    # sort span and label 
    # spans.sort()
    spans_, labels_ = deepcopy(spans), deepcopy(labels)
    spans, labels = [], []
    for s, l in sorted(zip(spans_, labels_), key=lambda x: x[0][0]): 
        spans.append(s)
        labels.append(l)

    # Remove dups
    prev_span = None
    dup_indices = []
    for i, span in enumerate(spans):
        if span == prev_span: 
            dup_indices.append(i)
        prev_span = span
    
    c = 0
    for i in dup_indices:
        spans.pop(i-c)
        labels.pop(i-c)
        c += 1

    # Check ovelap condition
    prev_max = 0
    for span in spans:
        assert prev_max <= span[0] #enforce no overlap condition
        prev_max = span[1]
    
    # Divide into sub sentences
    sub_sents =[]
    sub_labels =[]
    prev_max = 0
    for idx, span in enumerate(spans):
        sub_sents.append(text[prev_max:span[0]])
        sub_labels.append('O')
        sub_sents.append(text[span[0]:span[1]])
        sub_labels.append(labels[idx])
        prev_max = span[1]
    sub_sents.append(text[prev_max:])
    sub_labels.append('O')

    if not IS_WORDPIECE:
        for idx, sent in enumerate(sub_sents):
            if sent == "":
                continue
            while sent[-1]==" " and idx<len(sub_sents)-1:
                sub_sents[idx+1] = " " + sub_sents[idx+1]
                sent = sent[:-1]
                sub_sents[idx] = sent
                if sent == "":
                    break
        for idx, sent in enumerate(sub_sents):
            if sent == "":
                continue
            # print(idx, len(sub_sents))
            if idx<(len(sub_sents)-1) and len(sub_sents[idx+1])> 0 and is_combine_char(sent[-1]) and is_combine_char(sub_sents[idx+1][0]):
                # set_trace()
                sent = sent + sub_sents[idx+1][0] 
                sub_sents[idx] = sent
                sub_sents[idx+1] = sub_sents[idx+1][1:]    

    fin_tokens = []
    fin_labels = []
    for idx, sub_sent in enumerate(sub_sents):
        if sub_sent == "":
            continue
        new_tokens = tokenizer.tokenize(sub_sent)
        is_b_token = True
        for tok in new_tokens:
            if tok != target_tokens[0]:
                set_trace()
            assert tok == target_tokens[0]
            
            target_tokens = target_tokens[1:]
            fin_tokens.append(tok)
            label = sub_labels[idx]

            if label == 'O':
                fin_labels.append('O')
            elif is_b_token:
                fin_labels.append("B-"+label)
                is_b_token = False
            else:
                fin_labels.append("I-"+label)
                
    return tokenizer.convert_tokens_to_ids(fin_tokens), fin_labels


"""
Test code for debugging

"""
# import jsonlines
# import random
# from transformers import AutoTokenizer

# with jsonlines.open('data/CyNER/dev.jsonl') as f:
#     all = [a for a in f]

# tok = AutoTokenizer.from_pretrained('bert-base-uncased')
# # tok = AutoTokenizer.from_pretrained('roberta-base')

# for X in random.sample(all, 1):
#     # X = all[j]
#     text = X['article_text']
#     spans = X['span_pos']
#     labels = X['label']

#     print(X['span_text'], labels)
#     # target_tokens, _, target_label = tokenize_with_BIO_span(tok,text,spans,labels)
#     target_tokens, _, target_labels = tokenize_with_BIO_span(tok,text,spans,labels)
#     for i, (t, l) in enumerate(zip(target_tokens, target_labels)):
#         # if l != 'O':
#         print(t,l)
#     print("\n\n")

def special_tokens_to_tokenizer_and_model(tokenizer, model, special_tokens):
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    if model:
        model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def get_classification_results(all_preds, all_logits, all_labels):

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_logits[:,1])
    roc_auc = metrics.auc(fpr, tpr)

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_logits[:,1])
    f1 = f1_score(all_labels, all_preds)
    pr_auc = auc(recalls, precisions)

    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    acc = accuracy_score(all_labels, all_preds)
    return precision, recall, f1, acc, roc_auc, pr_auc

def find_preds(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, '-'.join(labels[i][1:])])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


def create_MalwareTextDB_task(dataset, tokenizer, max_length, number_of_tokens=180, num_choices = 5):
    query = "What {} is indicated by the phrase <{}> in this context?\n\nContext: {}<{}>{}\n\nAnswer: "
    tokeds= []
    prefixes = tokenizer(dataset['prefix_text'], add_special_tokens=False).input_ids
    prefixes = [tokenizer.decode(a[-number_of_tokens:]) for a in prefixes]
    suffixes = tokenizer(dataset['suffix_text'], add_special_tokens=False).input_ids
    suffixes = [tokenizer.decode(a[:number_of_tokens]) for a in suffixes]
    
    QuestionHeads = [query.format(_type, _target_text, _prefix, _target_text, _suffix) for _type,_target_text,_prefix,_suffix in zip(dataset['type'], dataset['target_text'], prefixes, suffixes)]
    QuestionHeads = [[question] * num_choices for question in QuestionHeads]
    flat_Q = sum(QuestionHeads,[])
    flat_A = sum(dataset['choices'],[])
    flat_QA = [q+a for q,a in zip(flat_Q,flat_A)]

    toked = tokenizer(flat_QA)
    assert len(max(toked.input_ids, key=len))<512
    
    return {k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in toked.items()}
