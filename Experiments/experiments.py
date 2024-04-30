from classifier import train_classifier, eval_model
import os
import json
import argparse
from statistics import stdev, mean,median

parser = argparse.ArgumentParser(description='bs,lr,sv,eval')
parser.add_argument('--bs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--sv', type=int)
parser.add_argument('--eval', action='store_true') # only do eval
parser.add_argument('--all', action='store_true')

args = parser.parse_args()

settings = None


task_names = ['CASIE_T1', 'CYDEC', 'TwitterThreats_T1']
task_names += ['CyNER', 'CySecED']
task_names += ['MalwareTextDB_New']

def all_experiments(sv, TRAIN_MODE= True, EVAL_MODE = True):
    BASELINE_TRAIN = ['ehsanaghaei/SecureBERT', 'ehsanaghaei/SecureBERT_Plus', 'roberta-base', 'models/CyBERT', 'markusbayer/CySecBERT']

    experiments_dir = []
    
    PREFIX = "TruffleV1_" + str(sv)+"_"
    seed_val = sv
    for task_name in task_names:
        try:
            with open('Scores/scores_'+task_name+'.json') as f:
                jres = json.load(f)
        except:
            jres = {}
            with open('Scores/scores_'+task_name+'.json','w') as f:
                json.dump(jres, f,indent=2)

        if TRAIN_MODE:
            for experiment_model in experiments_dir:
                cutname = PREFIX.split("_")[0] + experiment_model.split("/")[-1] if "/" in experiment_model else experiment_model
                if (cutname not in jres) or (str(sv) not in jres[cutname]):
                    train_classifier(task_name, experiment_model,  seed_val, PREFIX+task_name)
                else:
                    print("Already exists:", task_name, experiment_model, seed_val)
            for baseline in BASELINE_TRAIN:
                cutname =  PREFIX.split("_")[0] + (baseline.split("/")[-1] if "/" in baseline else baseline)
                if (cutname not in jres) or (str(sv) not in jres[cutname]):
                    train_classifier(task_name, baseline,  seed_val, PREFIX+task_name)
                else:
                    print("Already exists:", task_name, cutname, seed_val)
        if EVAL_MODE:
            target_dir =os.path.join('finished_models', task_name)
            models_dir = [a for a in os.listdir(target_dir) if PREFIX in a]
            reses = {}
            with open('Scores/scores_'+task_name+'.json') as f:
                jres = json.load(f)
            for model_dir in models_dir:
                shortname = PREFIX.split("_")[0] + model_dir.split(("."+str(sv)))[0].split(task_name)[1]
                if shortname in jres and str(sv) in jres[shortname]:
                    continue
                reses[shortname]=eval_model(task_name, os.path.join(target_dir, model_dir), 32)
            for a in reses:
                print(a, reses[a])
                if a not in jres:
                    jres[a] = {}
                jres[a][str(seed_val)] = reses[a]
            with open('Scores/scores_'+task_name+'.json') as f:
                jres_new = jres
                jres = json.load(f)
                for key in jres_new:
                    if key in jres:
                        jres[key].update(jres_new[key])
                    else:
                        jres[key] = jres_new[key]
            with open('Scores/scores_'+task_name+'.json','w') as f:
                json.dump(jres, f,indent=2)


def hyperparameter_search(bs, lr):

    settings= {
        'learning_rate': lr,
        'total_batch_size': bs
        }
    model_dir = 'roberta-base'

    PREFIX = f'HPARAMv1_{lr}_{bs}'
    task_names = ['CySecED']
    seed_val = 607

    with open('hp_results.json') as f:
        X = json.load(f)
        if PREFIX in X:
            print(PREFIX, "already exists in results")
    reses = {}
    for task_name in task_names:
        target_dir =os.path.join('finished_models', task_name)
        if len([a for a in os.listdir(target_dir) if PREFIX+task_name in a])==0:
            train_classifier(task_name, model_dir,  seed_val, PREFIX+task_name, settings)
        models_dir = [a for a in os.listdir(target_dir) if PREFIX+task_name in a]
        assert len(models_dir) ==1
        reses[task_name] = eval_model(task_name, os.path.join(target_dir, models_dir[0]), 32 )
    with open('hp_results.json') as f:
        X = json.load(f)
        X[PREFIX] = reses
    with open('hp_results.json','w') as f:
        json.dump(X, f,indent=2)
        
                    
if args.bs and args.lr:
    hyperparameter_search(args.bs, args.lr)
if args.sv:
    reses = all_experiments(args.sv, TRAIN_MODE=(not args.eval))
elif args.all:
    svs = [1234,12345,222,1996,1113]
    for sv in svs:
        reses = all_experiments(sv, TRAIN_MODE=(not args.eval))
        reses = all_experiments(sv*10, TRAIN_MODE=(not args.eval))
