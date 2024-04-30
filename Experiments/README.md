# Experiments

Code to evaluate language models on downstream cybersecurity tasks

# Files

## preprocess.py, preproess_MalwareTextDB.py

Preprocesses raw data from datasets into data form to be used by Transformers.
Raw datasets must be downloaded from their respective distributors.

Datasets:
- CyNER (Named Entity Recognition, Tokens)
- CySecED (Event Detection, Tokens)
- CASIE_T1 (Event Detection, Sequence)
- CYDEC (Event Detection, Sequence)
- TwitterThreats_T1 (Event Detection, Sequence)
- MalwareTextDB_New (MultiChoice of Malware attributes)


## experiments.py

1. Hyperparameter search

```
python experiments.py --lr 5e-5 --bs 8
```

2. Do full experiments

```
python experiments.py --all
```

## probe_fill_mask.py

Probing experiments.