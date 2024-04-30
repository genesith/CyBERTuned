from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed
import torch
import json
import os
from tqdm import tqdm
import sys


cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from Pretraining.data_utils import DataCollatorForTokClassandLanguageModeling, DatasetForTokenPrediction

INNLE= 100
NEAR_NLE = 200
        
tok = AutoTokenizer.from_pretrained('roberta-base')

def nlemat(tok_labels, target_label, vicinity, innle = INNLE, nearnle = NEAR_NLE):
    shape = tok_labels.shape
    boolmat = torch.zeros(shape,dtype=torch.bool)
    for i in target_label:
        boolmat |= tok_labels == i
    
    shifted_l = torch.zeros_like(boolmat).to(bool)
    shifted_r = torch.zeros_like(boolmat).to(bool)
    for k in range(vicinity):
        j = k + 1
        shifted_l |= torch.cat((boolmat[:, j:], torch.zeros([shape[0], j])), dim=1).to(bool)
        shifted_r |= torch.cat((torch.zeros([shape[0], j]), boolmat[:, :-j]), dim=1).to(bool)

    nearnles = (shifted_l | shifted_r)
    
    res = torch.zeros(shape)
    res[nearnles] = nearnle
    res[boolmat] = innle
    return res


def create_dataset_jargon_mask(cutoff=5000, spaced=True):
    file_path = "Dataset/valid.txt"

    target_path = f"FillMaskTest/mlm_jargon_test_spaced_{cutoff}.pt" if spaced else f"FillMaskTest/mlm_jargon_test_{cutoff}.pt"
    
    if os.path.exists(target_path):
        print("Already exists at", target_path)
        return 0
    jargon_file = "jargon_tokens_spaced.json" if spaced else "jargon_tokens.json"
    with open(jargon_file) as f:
        jd = json.load(f)
        jargon_list = torch.tensor([int(a) for a in list(jd.keys()) if int(a) > cutoff])
        print("Retaining %d words from original list of %d words"%(len(jargon_list), len(jd)))
    
    dataset= DatasetForTokenPrediction(tok,file_path, 512, replace_these=[])
        
    for line in tqdm(dataset.examples):
        line['labels'] = line['input_ids'].clone()
        jargon_indices = torch.isin(line['input_ids'], jargon_list)
        line['input_ids'][jargon_indices]=tok.mask_token_id
        line['labels'][~jargon_indices] = -100
    
    all_input_ids = [a['input_ids'] for a in dataset.examples]
    all_labels = [a['labels'] for a in dataset.examples]
    all_tok_labels = [a['tok_labels'] for a in dataset.examples]
    savethis = {}
    savethis['input_ids'] = torch.stack(all_input_ids)
    savethis['labels'] = torch.stack(all_labels)
    savethis['tok_labels'] = torch.stack(all_tok_labels)
    torch.save(savethis, target_path)

def get_model_scores(model_dir, target_dir, N =70):
    # Saving logits is way too big, but we need softmax or preds anyway.
    # Since we only want the softmaxes or preds for the indices that are masked,
    # we can compute those and store just those.
    

    test_data = torch.load(target_dir)
    if model_dir not in test_data:
        with torch.no_grad():
            cuda = torch.device('cuda')
            model = AutoModelForMaskedLM.from_pretrained(model_dir)
            model.eval()
            model.to(cuda)
            input_ids=test_data['input_ids'].to(cuda)
            labels_ = test_data['labels'].to(cuda)
            maskeds_ = labels_!=-100

            softandpreds = []
            for i in tqdm(range(len(input_ids)//N +1)):
                logits= model(input_ids[N*i: min(len(input_ids), N*(i+1))]).logits
                # if len(logits_acc)!=0:
                labels = labels_[N*i: min(len(input_ids), N*(i+1))]
                sm = logits.softmax(-1)
                pred = logits.argmax(-1)
                maskeds= maskeds_[N*i: min(len(input_ids), N*(i+1))]
                i_indices, j_indices = maskeds.nonzero(as_tuple=True)
                softmaxes = sm[i_indices, j_indices, labels[i_indices, j_indices]]
                preds = pred[i_indices, j_indices]
                softandpreds.append(torch.vstack([softmaxes, preds, (N*i)+i_indices, j_indices]).cpu())
        
        test_data[model_dir] = torch.hstack([a for a in softandpreds]).T
        
    torch.save(test_data, target_dir)
    
    return test_data
                
    

def test_model(model_dir, target_dir, vicinity=10, nletypes = [1,2,3,4,5,6,7]):
    
    test_data = get_model_scores(model_dir, target_dir,N =70)


    model_reses = test_data[model_dir]
    softmaxes, preds, Is, Js = model_reses.T
    nle_mat = nlemat(test_data['tok_labels'], nletypes, vicinity)
    
    innles = (nle_mat==INNLE)[Is.to(int),Js.to(int)]
    nearnles = (nle_mat==NEAR_NLE)[Is.to(int),Js.to(int)]
    outnles = (nle_mat==0)[Is.to(int),Js.to(int)]
    
    in_softmaxes = softmaxes[innles]
    near_softmaxes = softmaxes[nearnles]
    out_softmaxes = softmaxes[outnles]

    corrects =test_data['labels'][Is.to(int),Js.to(int)]==preds
    in_corrects = corrects[innles]
    near_corrects = corrects[nearnles]
    out_corrects = corrects[outnles]

    reses= {}
    reses["In-NLE acc"]= float(in_corrects.sum()/len(in_corrects))
    reses["Near-NLE acc"]= float(near_corrects.sum()/len(near_corrects))
    reses["Non-NLE acc"] = float((near_corrects.sum() + out_corrects.sum())/(len(near_corrects)+len(out_corrects)))
    
    print ("Total %d masked, %d in, %d near, %d out"%(len(corrects), len(in_corrects), len(near_corrects), len(out_corrects)))

    return reses

experiments_dir = ['og_run0717-0724',  'replace_all0706-1013',
                   'mlm_on_all0706-1012', 'only_ling_mlm0718-0948', 
                       'semi_ling_mlm0706-1004', 'semi_ling_no_tok0719-0041']
experiments_dir += ['roberta-base']


import pandas as pd
if __name__ == "__main__":
    cutoff = 25000
    spaced = True
    create_dataset_jargon_mask(cutoff=cutoff, spaced= spaced)
    target_dir = f"FillMaskTest/mlm_jargon_test_spaced_{cutoff}.pt" if spaced else f"FillMaskTest/mlm_jargon_test_{cutoff}.pt"
    all_reses = {}

    # experiments_dir.reverse()

    for mod in experiments_dir:
        print(mod)
        # get_model_scores(mod)
        if "Experiments" in mod:
            mod_n = mod.split("/")[-1]
        else:
            mod_n = mod
        all_reses[mod_n] = test_model(mod,target_dir, vicinity=20, nletypes=[3,4,5,6,7])
    
    columns = sorted(list(all_reses[next(iter(all_reses))].keys()))
    
    flattened_data = [
        {'Name': name, **{col: info[col] for col in columns}}  # Sort the columns alphabetically
        for name, info in all_reses.items()
    ]

    df = pd.DataFrame(flattened_data)
    
    spacedt = "spaced_" if spaced else ""
    output_file = f"output_jargon_{spacedt}{cutoff}.xlsx"
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Sheet1', index=False)

    writer.save()
