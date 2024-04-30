import os
import csv
import json
import jsonlines
import argparse
from collections import defaultdict
from random import random
import pandas as pd
import re

from sklearn.model_selection import train_test_split

from consts import ALL_DATASETS

from pdb import set_trace

parser = argparse.ArgumentParser(description='Preprocess datasets for tasks')
parser.add_argument('--datasets', nargs='*', help='specify datasets')
parser.add_argument('--all_datasets', action='store_true')



args = parser.parse_args()

'''Return train, dev, and test jsonl  '''
def data_split(json_objs):
    key_list = [jobj['article_idx'] for jobj in json_objs]

    train_keys, test_keys = train_test_split(key_list, train_size=0.8, random_state=1234)
    dev_keys, test_keys = train_test_split(test_keys, train_size=0.5, random_state=1234)
    
    train_keys, dev_keys, test_keys = set(train_keys), set(dev_keys), set(test_keys)

    train_jsonl, dev_jsonl, test_jsonl = list(), list(), list()

    for d in json_objs:
        if d['article_idx'] in train_keys: 
            train_jsonl.append(d)
        elif d['article_idx'] in dev_keys: 
            dev_jsonl.append(d)
        elif d['article_idx'] in test_keys: 
            test_jsonl.append(d)
    
    return [('train', train_jsonl), ('dev', dev_jsonl), ('test', test_jsonl)]

'''Save jsonl file
This is used when the raw files are splitted already'''
def save_jsonl(fname, jsonl):
    with jsonlines.open(fname, mode='w') as f:
        f.write_all(jsonl)
    print(f"{fname} saved; # Text: {len(jsonl)}")

'''Split and save a large jsonl file'''
def split_and_save(resultpath, jsonl_in):
    for p in data_split(jsonl_in):
        set_type, jsonl = p[0], p[1]
        fname = f"{resultpath}/{set_type}.jsonl"
        save_jsonl(fname, jsonl)


def find_sublist(list1, sublist):
            indices = []
            start = 0
            while True:
                if list1[start:start+len(sublist)] == sublist:
                    indices.append(start)
                    start += len(sublist)
                else:
                    start += 1
                if start >= len(list1):
                    return indices



# CYNER is in a format of Word\tTag\n format.
# It seems to be sentence-tokenized with \n, but this can be quite fickle.
# So we'll only consider \n following a '.' as actual new lines.
# The real problem is that the original dataset has no distinction when a new document occurs, which causes some awkward transitions.
# we manually annotate for new documents and mark it with "NEW_DOCUMENT\n"
def preprocess_CyNER(rawpath, resultpath):
    keyword = "NEW_DOCUMENT\n"
    if not os.path.exists(f"{resultpath}"):
        os.mkdir(f"{resultpath}")
    if not os.path.exists(f"{resultpath}_document"):
        os.mkdir(f"{resultpath}_document")
    
    doc_idx = 0
    article_idx = 0
    for split in ['valid','test','train']: 
        out_jsonl = list()
        span_pos, labels, token_list = list(), list(), list()
        raw_tokens, raw_labels = list(), list()
        text = ""
        start_flag = False
        prev_token, prev_iob = '', ''
        with open(os.path.join(rawpath+"_fixed", (split+".txt"))) as f:
            B_count = 0
            for line in f:
                # End of a sentence 
                if line == '\n' and prev_token == '.' and prev_iob == 'O':
                    json_obj = {
                        'doc_idx': doc_idx,
                        'article_idx': article_idx,
                        'span_pos': span_pos,
                        'label': labels,
                        'span_text': token_list,
                        'article_text': text,
                        'tokens': raw_tokens,
                        'token_labels': raw_labels
                    }
                    out_jsonl.append(json_obj)
                    article_idx += 1
                    text = ""
                    span_pos, labels, token_list = list(), list(), list()
                    raw_tokens, raw_labels = list(), list()
                    start_flag = False
                    continue
                if line == keyword:
                    if start_flag:
                        start_flag = False
                        span_pos.append((start_offset, end_offset))
                        labels.append(label_t)
                        token_list.append(text[start_offset:end_offset])
                    json_obj = {
                        'doc_idx': doc_idx,
                        'article_idx': article_idx,
                        'span_pos': span_pos,
                        'label': labels,
                        'span_text': token_list,
                        'article_text': text,
                        'tokens': raw_tokens,
                        'token_labels': raw_labels
                    }
                    out_jsonl.append(json_obj)
                    article_idx =0
                    doc_idx +=1
                    text = ""
                    span_pos, labels, token_list = list(), list(), list()
                    raw_tokens, raw_labels = list(), list()
                    start_flag = False
                    continue
                if line == '\n':
                    token, iob_label = '', None
                    continue
                else:
                    token, iob_label = line.strip().split('\t')
                    raw_tokens.append(token)
                    raw_labels.append(iob_label)
                
                # if started already, should make a new one
                if iob_label and start_flag and iob_label[0] == 'B':
                    start_flag = False
                    span_pos.append((start_offset, end_offset))
                    labels.append(label_t)
                    token_list.append(text[start_offset:end_offset])

                # Find start offset
                if iob_label and iob_label[0] == 'B':
                    B_count +=1
                    label_t = iob_label.strip().split('-')[-1]
                    start_offset = len(text)
                    start_flag = True

                # Reconstruction
                text = text + token + " " 
                
                # Find end offset  
                if iob_label and iob_label[0] in ['B', 'I']:
                    end_offset = len(text) - 1

                if iob_label and start_flag and iob_label[0] == 'O':
                    start_flag = False
                    span_pos.append((start_offset, end_offset))
                    labels.append(label_t)
                    token_list.append(text[start_offset:end_offset])

                prev_token, prev_iob = token, iob_label
        
            if len(text) >5:
                json_obj = {
                        'doc_idx': doc_idx,
                        'article_idx': article_idx,
                        'span_pos': span_pos,
                        'label': labels,
                        'span_text': token_list,
                        'article_text': text,
                        'tokens': raw_tokens,
                        'token_labels': raw_labels
                    }
                out_jsonl.append(json_obj)
            else:
                print("This was not handled", json_obj)
        span_count = 0
        for k in out_jsonl:
            span_count += len(k['span_pos'])
        assert span_count == B_count
        
        if split == 'valid':
            out_path = f"{resultpath}/dev.jsonl"
            out_path2 = f"{resultpath}_document/dev.jsonl"
        else:
            out_path = f"{resultpath}/{split}.jsonl"
            out_path2 = f"{resultpath}_document/{split}.jsonl"

        save_jsonl(out_path, out_jsonl) 


        docs = dict()
        for js in out_jsonl:
            doc_idx = js['doc_idx']
            if doc_idx in docs.keys():
                doc = docs[doc_idx]
                text = doc['article_text']
                doc['article_idx'] = js['article_idx']
                for sp in [[[len(text)+ a for a in b] for b in js['span_pos']]]:
                    doc['span_pos'].extend(sp)
                doc['label'].extend(js['label'])
                doc['span_text'].extend(js['span_text'])
                doc['article_text']+=js['article_text']
                doc['tokens'].extend(js['tokens'])
                doc['token_labels'].extend(js['token_labels'])
            else:
                docs[doc_idx] = js
        
        docs_l = [docs[i] for i in docs.keys()]
        for doc in docs_l:
            for i,span in enumerate(doc['span_pos']):
                assert (doc['article_text'][span[0]:span[1]] == doc['span_text'][i])
        
        save_jsonl(out_path2, docs_l)
            

def preprocess_CySecED(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}"):
        os.mkdir(f"{resultpath}")
    # Each object has an id that shows its tokens in docid.sentencenumber format
    # Make a dict (key: docid) of dicts (key: sentencenumber)

    for split in ['dev','test','train']:
        # some edits were made to fix incomplete annotations
        
        FIXED = True
        if FIXED and split in ['test', 'train']:
            splitf = split +"_fixed"
        else:
            splitf = split
        with open(os.path.join(rawpath, (splitf+".json"))) as f:
            jdata = json.load(f)

        docs_dict = dict()
        for x in jdata:
            a,b = x['id'].split(".")
            b = int(b.split("sentence")[1])
            if a not in docs_dict:
                docs_dict[a] = dict()
                for i, info in enumerate(x['doc']):
                    docs_dict[a][i]= info
            if x['label'] == "None":
                continue
            B = True
            
            for place in range(x['trigger_start'],x['trigger_end']+1):
                if B:
                    docs_dict[a][b]['triggers'][place] = "B-"+x['label']
                    B = False
                else:
                    docs_dict[a][b]['triggers'][place] = "I-"+x['label']

        # sanity check for ensuring all document triggers have corresponding labels
        for doc in docs_dict:
            for i in docs_dict[doc]:
                li = docs_dict[doc][i]['triggers']
                if FIXED:
                   assert 1 not in (list(dict.fromkeys(li)))
                #code for fixing, manual annotations done here
                #use fixed ones later
                elif 1 in (list(dict.fromkeys(li))):
                    for prints in range(max(0,i-3),min(i+2, len(docs_dict[doc])+1)):
                        print([a for a in zip(docs_dict[doc][prints]['triggers'], docs_dict[doc][prints]['token'])])
                    print([a for a in zip(li, docs_dict[doc][i]['token'])])
                    newid = f"{doc}.sentence{i}"
                    newjson = {'id': newid, 'trigger_start':0, 'trigger_end':0, 'label':"something"}
                    # manually insert trigger indices and label here and use
                    # jdata.append(newjson)
                    pass

        

        #after sanity checks, actual processing begins here
        out_jsonl = list()

        #this is a commonly occuring string in the dataset, we will remove it
        filterthis = [
            ['(', 'adsbygoogle', '=', 'window.adsbygoogle', '||', '[', ']', ')', '.push', '(', '{}', ')', ';'],
            ['(', 'adsbygoogle', '=', 'window.adsbygoogle', '||', '[', ']', ')', '.push', '(', '{','}', ')', ';']]
        
        
        
        for doc_id, doc in docs_dict.items():
            doc_tokens =[]
            doc_labels = []
            for sen_num, sen in doc.items():
                sen_tokens = sen['token']
                sen_labels = sen['triggers']
                for filterphrase in filterthis:
                    filterindices = find_sublist(sen_tokens, filterphrase)
                    if filterindices:
                        filterindices.reverse()
                        for indice in filterindices:
                            del sen_tokens[indice:indice+len(filterphrase)]
                            del sen_labels[indice:indice+len(filterphrase)]

                assert '.push' not in sen_tokens
                doc_tokens.extend(sen_tokens)
                doc_labels.extend(sen_labels)

            assert len(doc_tokens) == len(doc_labels)
            doc_labels = [a if a!=0 else 'O' for a in doc_labels]
            doc_text =""
            doc_spans =[]
            doc_span_text = []
            doc_span_label =[]
            for ind, (tok, lab) in enumerate(zip (doc_tokens, doc_labels)):
                if lab != 'O' and "B-" in lab:
                    label_only = lab[2:]
                    span_start = len(doc_text)
                    full_toks = tok
                    next_inds = ind+1
                    if "I" in str(doc_labels[next_inds]):
                        pass
                    while doc_labels[next_inds]=="I-"+label_only:
                        full_toks += " " + doc_tokens[next_inds]
                        next_inds +=1
                    doc_spans.append([span_start,span_start+len(full_toks)])
                    doc_span_label.append(label_only)
                doc_text += tok + " "
            for span_start,span_end in doc_spans:
                doc_span_text.append(doc_text[span_start:span_end])
            json_obj = {
                    'doc_idx': doc_id,
                    'span_pos': doc_spans,
                    'label': doc_span_label,
                    'span_text': doc_span_text,
                    'article_text': doc_text,
                    'tokens': doc_tokens,
                    'token_labels': doc_labels

                }
            out_jsonl.append(json_obj)

        out_path = f"{resultpath}/{split}.jsonl"

        save_jsonl(out_path, out_jsonl) 

# Do not use this since there are multiple errors in the token annotations...
def preprocess_MalwareTextDB_T2_from_tokens(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}_T2/"):
        os.mkdir(f"{resultpath}_T2/")
    
    article_idx = 0 # article index

    # we use test_1 as the test as the paper suggests, and add the unused test sets to the dev set
    dev_total = []
    for data_type in ['train', 'dev', 'test_1', 'test_2', 'test_3']:
        rawpath_ = os.path.join(f"{rawpath}-2.0", f'data/{data_type}/tokenized')
    
        out_jsonl = list()
        span_pos, labels, token_list = list(), list(), list()
        
        text = ""
        start_flag = False
        prev_token, prev_iob = '', ''

        data_files = [p for p in os.listdir(rawpath_) if ".tokens" in p]
        for data_fil in data_files:
            with open(os.path.join(rawpath_, data_fil)) as f:
                for line in f:
                    # End of a sentence 
                    if (line == ' \n' and prev_token == '.' and prev_iob == 'O') or (line == '\n' and prev_token == '.' and prev_iob == 'O'):
                        json_obj = {
                            'article_idx': article_idx,
                            'span_pos': span_pos,
                            'label': labels,
                            'span_text': token_list,
                            'article_text': text
                        }
                        out_jsonl.append(json_obj)
                        article_idx += 1
                        text = ""
                        span_pos, labels, token_list = list(), list(), list()
                        continue
                    
                    # '\n' in the middle of the sentence 
                    if line == '\n' or line == ' \n':
                        token, iob_label = '\n', None
                    else:
                        assert line.count(" ") == 1 
                        if len(line.strip().split(' ')) != 2:
                            print("Found an odd line at %s, skipping..."%(data_fil))
                            print(line)
                            continue
                        token, iob_label = line.strip().split(' ')
                        token = token.replace("\xad", "")
                    
                    # Find start offset
                    if iob_label and iob_label[0] == 'B':
                        label_t = iob_label.strip().split('-')[-1]
                        start_offset = len(text)
                        start_flag = True

                    # Reconstruction
                    text = text + token + " " 
                    
                    # Find end offset  
                    if iob_label and iob_label[0] in ['B', 'I']:
                        end_offset = len(text) - 1

                    if iob_label and start_flag and iob_label[0] == 'O':
                        start_flag = False
                        span_pos.append((start_offset, end_offset))
                        labels.append(label_t)
                        token_list.append(text[start_offset:end_offset])

                    prev_token, prev_iob = token, iob_label
        
        if data_type in ['dev', 'test_1', 'test_2', 'test_3']:
            dev_total.extend(out_jsonl)
            continue
        elif 'test' in data_type:
            out_path = f"{resultpath}_T2/test.jsonl"
        else:
            out_path = f"{resultpath}_T2/{data_type}.jsonl"

        save_jsonl(out_path, out_jsonl) 
    
    out_path = f"{resultpath}_T2/dev.jsonl"
    save_jsonl(out_path, dev_total)

def preprocess_MalwareTextDB(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}_T2/"):
        os.mkdir(f"{resultpath}_T2/")
    if not os.path.exists(f"{resultpath}_T4/"):
        os.mkdir(f"{resultpath}_T4/")
    
    idx = 0 # article index
    
    for data_type in ['train', 'dev', 'test_1', 'test_3']:

        rawpath_ = os.path.join(f"{rawpath}-2.0", f'data/{data_type}/annotations')

        # Output jsonl object initialization
        jsonl_t2, jsonl_t4 = list(), list()

        for fname in os.listdir(rawpath_):
            if not '.ann' in fname:
                continue
            # Read article text
            with open(os.path.join(rawpath_, f"{'.'.join(fname.split('.')[:-1])}.txt")) as f:
                text = f.read()

            # Read annotation file
            with open(os.path.join(rawpath_, fname), 'r') as f:
                tokenid2span = defaultdict(dict)

                json_obj_t2 = dict()
                json_obj_t2['article_idx'] = idx
                json_obj_t2['article_text'] = text
                json_obj_t2['span_pos'], json_obj_t2['label'], json_obj_t2['span_text'] = list(), list(), list()
                json_obj_t2['fname'] = fname

                json_obj_t4 = dict()
                json_obj_t4['article_idx'] = idx
                json_obj_t4['article_text'] = text
                json_obj_t4['span_pos'], json_obj_t4['label'], json_obj_t4['span_text'] = list(), list(), list()

                for line in f:
                    tmp = line.strip().split('\t')

                    # Extraction for Task 2
                    if tmp[0][0] == 'T':
                        split_ = tmp[1].split(' ')
                        label, start_offset, end_offset = split_[0], split_[1], split_[-1]
                        if label == "Subject" or label == "Object":
                            label = "Entity"
                        start_offset, end_offset = int(start_offset), int(end_offset)

                        json_obj_t2['span_pos'].append((start_offset, end_offset))
                        json_obj_t2['label'].append(label)
                        json_obj_t2['span_text'].append(text[start_offset: end_offset])

                        tokenid = tmp[0] # there is an id for each token in each article 
                        tokenid2span[tokenid]['span_pos'] = (start_offset, end_offset)
                        tokenid2span[tokenid]['span_text'] = tmp[-1]

                    # Extraction for Task 4
                    elif tmp[0][0] == 'A':                    
                        t4_category, tokenid, t4_label = tmp[1].split(' ')
                        
                        json_obj_t4['span_pos'].append(tokenid2span[tokenid]['span_pos'])
                        json_obj_t4['label'].append(t4_label)
                        json_obj_t4['span_text'].append(tokenid2span[tokenid]['span_text'])

            # Finish extracting from an article, append to jsonl
            jsonl_t2.append(json_obj_t2)
            jsonl_t4.append(json_obj_t4)
            
            # Next article index 
            idx += 1
        
        # Sort by key
        jsonl_t2 = sorted(jsonl_t2, key=lambda d: d['article_idx'])
        jsonl_t4 = sorted(jsonl_t4, key=lambda d: d['article_idx'])
        
        # Save json
        if data_type == 'test_1':
            save_jsonl(f"{resultpath}_T2/test.jsonl", jsonl_t2)
        elif data_type == 'test_3':
            save_jsonl(f"{resultpath}_T4/test.jsonl", jsonl_t4)
        else:
            save_jsonl(f"{resultpath}_T2/{data_type}.jsonl", jsonl_t2)
            save_jsonl(f"{resultpath}_T4/{data_type}.jsonl", jsonl_t4)

    # split_and_save(resultpath, )

def preprocess_MalwareTextDB_T1(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}_T1/"):
        os.mkdir(f"{resultpath}_T1/")
    
    idx = 0 # sentence index

    # output jsonl object initialization
    jsonl_t1 = list()

    for data_type in ['train', 'dev', 'test_1']:
        rawpath_ = os.path.join(f"{rawpath}-2.0", f'data/{data_type}/tokenized')

        for fname in os.listdir(rawpath_):
            # read BIO
            with open(os.path.join(rawpath_, f"{'.'.join(fname.split('.')[:-1])}.tokens")) as f:
                tokens = f.read()
                sentences = tokens.split('\n \n')       # split sentence
                
                for sent in sentences:
                    tokens = [tok for tok in sent.split('\n') if tok]
                    sentence = [token.split(' ')[0] for token in tokens]
                    if not sentence:
                        continue
                    sentence = ' '.join(sentence[:-1]) + sentence[-1]
                    bio_tag = list(set([token.split(' ')[1] for token in tokens]))
                    validation = 1 if len(bio_tag) > 1 else 0
                    json_out = {
                        'article_idx': idx,
                        'text': sentence,
                        'label': validation
                    }
                    jsonl_t1.append(json_out)
                    idx += 1

    # Sort by key
    jsonl_t1 = sorted(jsonl_t1, key=lambda d: d['article_idx'])

    # Save jsonl
    split_and_save(f"{resultpath}_T1", jsonl_t1)

def preprocess_CASIE(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}_T1/"):
        os.mkdir(f"{resultpath}_T1/")
    if not os.path.exists(f"{resultpath}_T2/"):
        os.mkdir(f"{resultpath}_T2/")
    
    jsonl_t1, jsonl_t2 = list(), list()
    
    for fname in os.listdir(rawpath):
        # Filter out non-json file
        if "json" not in fname: 
            continue

        # Output json obejct initialization
        idx = int(fname.split('.')[0])
        json_obj_t1 = dict()
        json_obj_t1['article_idx'] = idx
        
        json_obj_t2 = dict()
        json_obj_t2['article_idx'] = idx
        
        json_obj_t1['span_pos'], json_obj_t1['label'], json_obj_t1['span_text'] = list(), list(), list()
        json_obj_t2['span_pos'], json_obj_t2['label'], json_obj_t2['span_text'] = list(), list(), list()

        # Start extraction
        with open(os.path.join(rawpath, fname)) as f:
            data = json.load(f)
            text = data['content']
            json_obj_t1['article_text'] = text
            json_obj_t2['article_text'] = text

            hopper = data['cyberevent']['hopper']
            for h in hopper:
                events = h['events']
                for e in events:
                    # Extraction for Task 1: nugget detection 
                    nugget = e['nugget']
                    nugget_type = e['type']
                    nugget_subtype = e['subtype']
                    nugget_s = nugget['startOffset']
                    negget_e = nugget['endOffset']
                    nugget_label = nugget_type + '.' + nugget_subtype
                    
                    # Fill json object for T1
                    json_obj_t1['span_pos'].append((nugget_s, negget_e))
                    json_obj_t1['label'].append(nugget_label)
                    json_obj_t1['span_text'].append(nugget['text'])
                    
                    # Extraction for Task 2: argument detection
                    if not 'argument' in e:
                        continue
                    argumets = e['argument'] 
                    for arg in argumets:
                        arg_s = arg['startOffset']
                        arg_e = arg['endOffset']
                        arg_type = arg['type']
                        
                        # Fill json object for T2
                        json_obj_t2['span_pos'].append((arg_s, arg_e))
                        json_obj_t2['label'].append(arg_type)
                        json_obj_t2['span_text'].append(arg['text'])
            
            # jsonl_t1.append(json_obj_t1)
            # jsonl_t2.append(json_obj_t2)
    
            t1_add_flag = True
            for i ,s in enumerate(json_obj_t1['span_pos']):
                if json_obj_t1['span_text'][i] != json_obj_t1['article_text'][s[0]:s[1]]: 
                    t1_add_flag = False
                    break
            t2_add_flag = True
            for i ,s in enumerate(json_obj_t2['span_pos']):
                if json_obj_t2['span_text'][i] != json_obj_t1['article_text'][s[0]:s[1]]: 
                    t2_add_flag = False
                    break
            if t1_add_flag:
                jsonl_t1.append(json_obj_t1)
            if t2_add_flag:
                jsonl_t2.append(json_obj_t2)
    
    # Sort by key
    jsonl_t1 = sorted(jsonl_t1, key=lambda d: d['article_idx'])
    jsonl_t2 = sorted(jsonl_t2, key=lambda d: d['article_idx'])

    # Save json
    split_and_save(f"{resultpath}_T1", jsonl_t1)
    split_and_save(f"{resultpath}_T2", jsonl_t2)

def preprocess_TwitterThreats(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}_T1/"):
        os.mkdir(f"{resultpath}_T1/")
    if not os.path.exists(f"{resultpath}_T2/"):
        os.mkdir(f"{resultpath}_T2/")
    
    jsonl_t1 = list()
    jsonl_t2 = list()

    with open(os.path.join(rawpath, 'tweets_raw_data.json'), 'r') as f:
        tweets_list = json.load(f)

    for tweets in tweets_list:
        text_p = tweets['text'] + ' ' + '[SEP]' + ' ' + tweets['curr_ner'][0]
        validation_t1 = 1 if tweets['existence_anno'] == 'have_threat' else 0
        jsonl_t1.append({'article_idx': tweets['id'], 'text': text_p, 'label': validation_t1})
        if validation_t1 == 1:
            validation_t2 = 1 if tweets['severity_anno'] == 'severe' else 0
            jsonl_t2.append({'article_idx': tweets['id'], 'text': text_p, 'label': validation_t2})
    
    # Sort by key
    jsonl_t1 = sorted(jsonl_t1, key=lambda d: d['article_idx'])
    jsonl_t2 = sorted(jsonl_t2, key=lambda d: d['article_idx'])

    # Save json
    split_and_save(f"{resultpath}_T1", jsonl_t1)
    split_and_save(f"{resultpath}_T2", jsonl_t2)

def preprocess_CYDEC(rawpath, resultpath):
    if not os.path.exists(f"{resultpath}/"):
        os.mkdir(f"{resultpath}/")
    
    jsonl_obj = list()

    with open(os.path.join(rawpath, 'cydec.csv'), 'r') as f:
        tweets_reader = csv.reader(f, delimiter=';')
        for idx, row in enumerate(tweets_reader):
            if idx == 0:
                continue
            if idx == 2001:
                break            
            jsonl_obj.append({'article_idx': row[0], 'text': row[1], 'label': int(row[2])})
    
    # Sort by key
    jsonl_obj = sorted(jsonl_obj, key=lambda d: d['article_idx'])

    # Save json
    split_and_save(f"{resultpath}", jsonl_obj)


def do_preprocess(dataset):
    rawfol = "raw_data"
    datafol = "data"
    if dataset == 'CySecED':
        rawpath = os.path.join(rawfol,'CySecED')
        resultpath = os.path.join(datafol,'CySecED')
        preprocess_CySecED(rawpath, resultpath)
    elif dataset == "CASIE":
        rawpath = os.path.join(rawfol,'CASIE')
        resultpath = os.path.join(datafol,'CASIE')
        preprocess_CASIE(rawpath, resultpath)
    elif dataset == "MalwareTextDB":
        rawpath = os.path.join(rawfol,'MalwareTextDB')
        resultpath = os.path.join(datafol,'MalwareTextDB')
        preprocess_MalwareTextDB(rawpath, resultpath)
        preprocess_MalwareTextDB_T1(rawpath, resultpath)
    elif dataset == "TwitterThreats":
        rawpath = os.path.join(rawfol,'TwitterThreats')
        resultpath = os.path.join(datafol,'TwitterThreats')
        preprocess_TwitterThreats(rawpath, resultpath)
    elif dataset == "CYDEC":
        rawpath = os.path.join(rawfol,'CYDEC')
        resultpath = os.path.join(datafol,'CYDEC')
        preprocess_CYDEC(rawpath, resultpath)
    elif dataset == "CyNER":
        rawpath = os.path.join(rawfol,'CyNER')
        resultpath = os.path.join(datafol,'CyNER')
        preprocess_CyNER(rawpath, resultpath)  

def main():
    if args.all_datasets:
        datasets = ALL_DATASETS
    else:
        if not args.datasets:
            print("No datasets given, shutting down.")
            return -1
        datasets = args.datasets
        try:
            for ds in datasets:
                assert ds in ALL_DATASETS
        except AssertionError:
            print("Did not recognize following dataset: ", ds, " shutting down.\nDataset should be in ", ALL_DATASETS)
            return -1
    print("Preprocessing following tasks: ", ",".join(datasets))

    for ds in datasets:
        do_preprocess(ds)


if __name__ == '__main__':
    main()
