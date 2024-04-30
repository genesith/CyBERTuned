import argparse
import json
from statistics import stdev, mean,median
from scipy import stats
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--task', nargs='*', help='specify tasks')
parser.add_argument('--og')
parser.add_argument('--prefix', default="")
args = parser.parse_args()


final_res = {}

for task in args.task:
    with open('scores_'+task+'.json') as f:
        jres = json.load(f)
    print("Med, Mean, Model")
    if args.og:
        og_run = [m for m in jres if args.og in m][0]
    
    for model in jres:
        if args.prefix not in model:
            continue
        
        vals = [a for a in jres[model].values()]
        
        if args.prefix:
            pmodel = model.split(args.prefix)[1]
        else:
            pmodel = model

        print("%.3f, %.3f,"%(median(vals), mean(vals)), pmodel, len(vals))
        final_res[pmodel] = {"median": median(vals), "mean": mean(vals)}
        if args.og:
            o_vals = [a for a in jres[og_run].values()]
            t_stat, p_value = stats.ttest_ind(o_vals, vals, equal_var=False)
            print("P-value is ", p_value)
            final_res[pmodel]['p-val'] = p_value
    
    
