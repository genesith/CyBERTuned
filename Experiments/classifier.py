import datetime
from datasets import DatasetDict, Dataset
from collections import defaultdict
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification,\
    AutoModel, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback, BertTokenizer, set_seed,\
    AutoModelForSeq2SeqLM, AutoModelForMultipleChoice
import datetime
from pytz import timezone
import os
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from typing import Optional, Union
from dataclasses import dataclass

from torch.cuda import device_count

import evaluate
import numpy as np
import seqeval.metrics as seqmetric
# from seqeval.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report
from data_util import *

from sklearn.metrics import classification_report
from finetuning_settings import ALL_SETTINGS

from torch.nn import CrossEntropyLoss

from consts import task2labels, data_path

from sklearn.metrics import f1_score

# batch_size = 16 


@dataclass

class DataCollatorForMultipleChoice:

    """

    Data collator that will dynamically pad the inputs for multiple choice received.

    """

    tokenizer: PreTrainedTokenizerBase

    padding: Union[bool, str, PaddingStrategy] = True

    max_length: Optional[int] = 512

    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = [feature.pop(label_name) for feature in features]

        batch_size = len(features)

        num_choices = len(features[0]["input_ids"])

        flattened_features = [

            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features

        ]

        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(

            flattened_features,

            padding=self.padding,

            max_length=self.max_length,

            pad_to_multiple_of=self.pad_to_multiple_of,

            return_tensors="pt",

        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch

def f1_metric(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    choices = list(dict.fromkeys(list(labels.flatten())))
    choices.remove(0)
    if -100 in choices:
        choices.remove(-100)
    return {'f1': f1_score(labels.flatten(),preds.flatten(), average='micro',labels=choices)}

def loose_eval(labels, preds,types):
    types = types.copy()
    if 'O' in types:
        types.remove('O')
    for lab in types:
        if "I-" in lab:
            types.remove(lab)
    types = [a.split("B-")[1] for a in types]
    
    counters_p = {x: list() for x in types} #predictions
    counters_a = {x: list() for x in types} #answers
    counters_c = {x: list() for x in types} #correct predictions
    res = dict.fromkeys(types)

    found_preds = find_preds(preds)
    found_answers = find_preds(labels)

    total_num_gold, total_num_correct, total_num_proposed = 0,0,0
    for a,b,key in found_preds:
        counters_p[key].append((a,b))
    for a,b,key in found_answers:
        counters_a[key].append((a,b))
    
    for key in types:
        preds = counters_p[key]
        answers = counters_a[key]
        
        num_gold = len(answers)
        total_num_gold += num_gold

        corrects = counters_c[key]
        for pred_start, pred_end in preds:
            for answer_start,answer_end in answers:
                if not (pred_end <answer_start or pred_start > answer_end):
                    corrects.append((answer_start, answer_end))
                    answers.remove((answer_start, answer_end))
        
        num_correct = len(corrects)
        num_proposed = len(preds)
        total_num_correct += num_correct
        total_num_proposed += num_proposed
        
        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        print(key, precision,recall, f1)
        res[key] = (precision,recall,f1)


    if total_num_proposed != 0:
            precision = total_num_correct / total_num_proposed
    else:
        precision = 0

    if total_num_gold != 0:
        recall = total_num_correct / total_num_gold
    else:
        recall = 0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    print('\nTotal', precision,recall, f1)
    res['Total'] = (precision,recall,f1)
    return res


# Given word spans and token spans, group tokens into words
def match_token_spans_to_words(token_spans, word_spans,is_deberta=False):

    if is_deberta:
        word_spans = [(max(a-1,0),b) for a,b in word_spans]
    word_tokens =[]
    curr_word_index = 0
    curr_word_tokens = []
    for i,tok in enumerate(token_spans):
        curr_word_start, curr_word_end = word_spans[curr_word_index]
        while tok[0]>curr_word_end:
            curr_word_index +=1
            word_tokens.append(curr_word_tokens)
            curr_word_tokens=[]
            curr_word_start, curr_word_end = word_spans[curr_word_index]
        assert tok[0]>=curr_word_start
        if tok[1] == curr_word_end:
            curr_word_tokens.append(i)
            # Doing this unconditionally breaks when multiple tokens are allowed in a single character
            # Therefore, do this check
            if i<len(token_spans)-1 and token_spans[i+1][1] <= curr_word_end:
                continue
            curr_word_index +=1
            word_tokens.append(curr_word_tokens)
            curr_word_tokens=[]
        elif tok[1]<curr_word_end:
            curr_word_tokens.append(i)
        else:
            assert False
    return word_tokens


# Prepare a dataset, labeled with index spans, into a dataset of BIO tokens.
# When using a raw dataset with given word labels, utilize those USE_TOKEN_LABELS
# Else, labels are given as spans, which should be assigned one B and rest I.
def tokenize_BIO_function(examples, tokenizer, max_size, labels2idx):
    IS_BERTSTYLE = False
    if tokenizer.tokenize(" space text") == tokenizer.tokenize("space test"):
        IS_BERTSTYLE = True

    USE_TOKEN_LABELS = False and 'tokens' in examples.column_names 
    texts = examples['article_text']
    spans = examples['span_pos']
    labels = examples['label']
    if USE_TOKEN_LABELS:
        words = examples ['tokens']
        word_labels = examples['token_labels']

    all_inputs = []
    all_labels =[]
    all_attention_masks = []
    

    labels2idx['N/A'] = CrossEntropyLoss().ignore_index
    if USE_TOKEN_LABELS:
        for text,span,label,word,word_label in zip(texts,spans,labels, words, word_labels):
            tokenized = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            # Make a list of tokens for each word
            offset = 0
            word_offsets=[]
            for w in word:
                word_offsets.append((offset,offset+len(w)))
                offset+= len(w)+1
            is_Roberta=False
            if IS_BERTSTYLE:
                # BERT style
                tokens_of_words = match_token_spans_to_words(tokenized['offset_mapping'], word_offsets)
            elif tokenized['offset_mapping'][-1][0] == tokenized['offset_mapping'][-1][1] and tokenized['offset_mapping'][-1][0] == len(text):
                # RoBERTa style
                tokens_of_words = match_token_spans_to_words(tokenized['offset_mapping'][:-1], word_offsets)
                is_Roberta= True
            else:
                # DeBERTa style
                if tokenized['offset_mapping'][-1][1]>word_offsets[-1][1]:
                    tokens_of_words = match_token_spans_to_words(tokenized['offset_mapping'][:-1], word_offsets,True)
                    is_Roberta = True
                else:
                    tokens_of_words = match_token_spans_to_words(tokenized['offset_mapping'], word_offsets,True)
            
            
            token_labels = []
            for tokens_of_word, label in zip(tokens_of_words, word_label):
                word_head = [1] + [0]*(len(tokens_of_word)-1)
                
                tok_label = [label] + ['N/A']*(len(tokens_of_word)-1)
                token_labels.extend(tok_label)
            if is_Roberta:
                
                token_labels.extend(['N/A'])
            
            label_indices = [labels2idx[a] for a in token_labels]
            # If a sequence is too long len(tokens) = k * max_size + l . Divide into (k+1) parts.
            token_len = len(tokenized['input_ids'])
            num_parts = token_len//max_size+ 1
            for i in range(num_parts):
                start = i*(token_len)//num_parts
                end = (i+1)*(token_len)//num_parts
                pad_len = max_size-(end-start)

                input_ind = tokenized['input_ids'][start:end]
                input_ind.extend([0]*(pad_len))
                all_inputs.append(input_ind)

                label_ind = label_indices[start:end]
                label_ind.extend([0]*(pad_len))
                all_labels.append(label_ind)

                attention_mask = ([1] * (end-start))
                attention_mask.extend([0] * pad_len)
                all_attention_masks.append(attention_mask)

            
        return {'input_ids':all_inputs,
        'labels':all_labels,
        'attention_mask': all_attention_masks,
        }
    else:
        for text,span,label in zip(texts,spans,labels):
            if span != []:
                span, label = zip(*sorted(zip(span,label), key= lambda x:x[0]))
            input_ids, token_labels =tokenize_with_BIO_span(tokenizer, text, span, label)
            label_indices = [labels2idx[a] for a in token_labels]
            # If a sequence is too long len(tokens) = k * max_size + l . Divide into (k+1) parts.
            length = len(input_ids)
            num_parts = length//max_size+ 1
            for i in range(num_parts):
                start = i*(length)//num_parts
                end = (i+1)*(length)//num_parts
                pad_len = max_size-(end-start)

                input_ind = input_ids[start:end]
                input_ind.extend([0]*(pad_len))
                all_inputs.append(input_ind)

                label_ind = label_indices[start:end]
                label_ind.extend([0]*(pad_len))
                all_labels.append(label_ind)

                attention_mask = ([1] * (end-start))
                attention_mask.extend([0] * pad_len)
                all_attention_masks.append(attention_mask)

            
        return {'input_ids':all_inputs,
        'labels':all_labels,
        'attention_mask': all_attention_masks}



# Train classifier, token or sequence
def train_classifier(task_name, pretrained_path, random_seed, prefix="",settings=None, PARAM_FREEZE=False):
    num_gpu = device_count()
    batch_size=int(32/num_gpu)
    
    IS_SEQ = task_name in SEQUENCE_TASKS
    IS_MULTICHOICE = task_name in MULTICHOICE_TASKS
    print(f"start training on rs - {random_seed}")
    print(f"model path: {pretrained_path}")
    print(f"data path: {os.path.join(data_path,task_name)}")
    
    all_datasets = load_splits_suffix('json', os.path.join(data_path,task_name), ['train','dev'])
    
    if IS_SEQ or IS_MULTICHOICE:
        possible_labels = task2labels[task_name]
    else:
        possible_labels = get_BIO_labels(task_name)
        labels2idx = {}
        for i, lab in enumerate(possible_labels):
            labels2idx[lab] = i

    set_seed(random_seed)
    
    save_name = pretrained_path
    
    if "/" in save_name:
        save_name = os.path.basename(pretrained_path)
    
    config = AutoConfig.from_pretrained(pretrained_path, num_labels=len(possible_labels), add_pooling_layer=False)
    #if "CyBERT" in pretrained_path: Fixed by removing file!
    #    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    if "deberta-v3" in pretrained_path:
        from transformers import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v2-xlarge')
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    # If the pretrained model was trained on Token Classification, it needs to know it should be overridden
    ignore_mismatched_sizes = 'Experiment' in pretrained_path

    if IS_MULTICHOICE:
        model = AutoModelForMultipleChoice.from_pretrained(pretrained_path, config =config, ignore_mismatched_sizes=ignore_mismatched_sizes)
    else:
        AutoClassifier = AutoModelForSequenceClassification if IS_SEQ else AutoModelForTokenClassification
        model = AutoClassifier.from_pretrained(pretrained_path, config =config, ignore_mismatched_sizes=ignore_mismatched_sizes)
    if PARAM_FREEZE:
        for param in model.base_model.parameters():
            param.requires_grad = False
    if IS_MULTICHOICE:
        if task_name == 'MalwareTextDB_New':
            tk_datasets_tr =  all_datasets['train'].map(lambda x: create_MalwareTextDB_task(x, tokenizer=tokenizer, max_length= 512), batched=True)
            tk_datasets_dev = all_datasets['dev'].map(lambda x: create_MalwareTextDB_task(x, tokenizer=tokenizer, max_length= 512), batched=True)
        else:
            raise Exception("Task is not implemented")
    elif IS_SEQ:
        tk_datasets = all_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True,load_from_cache_file=False)
        tk_datasets_tr = tk_datasets['train']
        tk_datasets_dev = tk_datasets['dev']
    else:
        tk_datasets_tr = Dataset.from_dict(tokenize_BIO_function(all_datasets['train'], tokenizer, 512, labels2idx))
        tk_datasets_dev = Dataset.from_dict(tokenize_BIO_function(all_datasets['dev'], tokenizer, 512, labels2idx))
    
    starttime = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%m-%d.%X")
    runname = prefix+save_name + "." + str(random_seed) + "." + starttime # for saving checkpoint while training
    
    
    stop_patience=4

    if IS_SEQ or IS_MULTICHOICE:
        strategy='steps'
        
    else:
        strategy='epoch'
        
    training_args = TrainingArguments(
        os.path.join('training_runs', task_name, runname),
        learning_rate=2e-5, do_eval=True,
        per_device_train_batch_size=1, 
        save_total_limit = stop_patience,
        per_device_eval_batch_size=batch_size, num_train_epochs = 20, warmup_ratio=0.06,
        load_best_model_at_end=True, seed=random_seed,
        logging_strategy=strategy, save_strategy =strategy, evaluation_strategy=strategy, 
        report_to='tensorboard',
        logging_dir='logs_here/'+runname,
        logging_first_step= True
        )
    
    if IS_SEQ or IS_MULTICHOICE:
        step_size = 200
        training_args.logging_steps=step_size
        training_args.save_steps =step_size
        training_args.eval_steps = step_size
    
    #update with settings
    if settings:
        custom_settings = settings
    elif task_name in ALL_SETTINGS:
        custom_settings = ALL_SETTINGS[task_name]
    else:
        print("Unknown task and setting")
        return -1
    
    training_args.learning_rate = custom_settings['learning_rate']
    training_args.per_device_train_batch_size = custom_settings['total_batch_size']//num_gpu
    if 'accumulate' in custom_settings:
        k = custom_settings['accumulate']
        training_args.per_device_train_batch_size = (custom_settings['total_batch_size']//num_gpu)//k
        training_args.gradient_accumulation_steps =k
    assert custom_settings['total_batch_size']%num_gpu==0

    if IS_SEQ:
        metrics = None
    elif IS_MULTICHOICE:
        accuracy = evaluate.load('accuracy')
        def acc_metric(eval_pred):

            predictions, labels = eval_pred

            predictions = np.argmax(predictions, axis=1)            
            return accuracy.compute(predictions=predictions, references=labels)
        metrics = acc_metric
    else:
        metrics = f1_metric
    trainer = Trainer(
        model=model, args=training_args, 
        train_dataset=tk_datasets_tr, eval_dataset=tk_datasets_dev,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=stop_patience)], compute_metrics=metrics,
        data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer) if IS_MULTICHOICE else None
    )
    trainer.train()

    save_path = os.path.join(
        f"exp_models/", 
        task_name, runname)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(os.path.join('training_runs', task_name, runname))
    tokenizer.save_pretrained(save_path)
    trainer.save_state()
    return save_path

def eval_model(task_name, finetuned_path, batch_size,LOOSE_EVAL=False,return_all_metrics=False):
    IS_SEQ = task_name in SEQUENCE_TASKS
    IS_MULTICHOICE = task_name in MULTICHOICE_TASKS
    if 'CASIE' in task_name or 'CySecED' in task_name:
        LOOSE_EVAL = True
    print(f"model path: {finetuned_path}")
    all_datasets = load_splits_suffix('json', os.path.join(data_path,task_name), ['test'])
    if IS_SEQ or IS_MULTICHOICE:
        possible_labels = task2labels[task_name]
    else:
        possible_labels = get_BIO_labels(task_name)
        labels2idx = {}
        for i, lab in enumerate(possible_labels):
            labels2idx[lab] = i

    # if "CyBERT" in finetuned_path:
    #     tokenizer = BertTokenizer.from_pretrained(finetuned_path,max_len=512)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path,max_len=512)
        #tokenizer = AutoTokenizer.from_pretrained('roberta-base',max_len=512)
    
    if IS_MULTICHOICE:
        if task_name == 'MalwareTextDB_New':
            test_dataset =  all_datasets['test'].map(lambda x: create_MalwareTextDB_task(x, tokenizer=tokenizer, max_length= 512), batched=True)
        else:
            raise Exception("Task is not implemented")
    elif IS_SEQ:
        tk_datasets = all_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True,load_from_cache_file=False)
        test_dataset = tk_datasets['test']
    else:
        test_dict=tokenize_BIO_function(all_datasets['test'], tokenizer, 512, labels2idx)
        test_dataset = Dataset.from_dict(test_dict)
    
    if IS_MULTICHOICE:
        model = AutoModelForMultipleChoice.from_pretrained(finetuned_path)
    else:
        AutoClassifier = AutoModelForSequenceClassification if IS_SEQ else AutoModelForTokenClassification
        model = AutoClassifier.from_pretrained(finetuned_path)
    print("TEST RES ", "="*15)

    training_args = TrainingArguments(
        os.path.join('eval_runs',task_name,finetuned_path), 
        do_eval=True, do_predict=True,
        per_device_eval_batch_size=batch_size)
    trainer = Trainer(args=training_args, model= model,\
                      data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer) if IS_MULTICHOICE else None)
    test_outs=trainer.predict(test_dataset=test_dataset)
    preds, labels = test_outs.predictions.argmax(-1), test_outs.label_ids

    if IS_SEQ:
        print(classification_report(labels, preds))
        return f1_score(labels,preds)
    elif IS_MULTICHOICE:
        accuracy = evaluate.load('accuracy')
        return accuracy.compute(predictions=preds, references=labels)['accuracy']
    
    ignores = labels ==-100
    preds = preds[~ignores]
    labels = labels[~ignores]
    
    preds2 = [[possible_labels[a] for a in preds]]
    labels2 = [[possible_labels[a] for a in labels]]

    if LOOSE_EVAL:
        loose_evaled = loose_eval(labels2[0], preds2[0], possible_labels)
        if False:
            return loose_evaled
        if return_all_metrics:
            return loose_evaled['Total']
        else:
            return loose_evaled['Total'][-1]
    

    # precision =  seqmetric.precision_score(labels2,preds2)
    # recall = seqmetric.recall_score(labels2,preds2)
    f1 = seqmetric.f1_score(labels2,preds2)
    # acc =  seqmetric.accuracy_score(labels2,preds2)
    rep=seqmetric.classification_report(labels2,preds2)
    print(rep)
    if return_all_metrics:
        precision =  seqmetric.precision_score(labels2,preds2)
        recall = seqmetric.recall_score(labels2,preds2)
        return precision, recall,f1
    
    return f1


