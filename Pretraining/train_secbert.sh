### Pretraining Exps
# CUDA_VISIBLE_DEVICES=0,1 python run_mlm.py train_secbert_modern_nle.json semi_ling_mlm
# CUDA_VISIBLE_DEVICES=2,3 python run_mlm.py train_secbert_modern_nle.json og_run
# CUDA_VISIBLE_DEVICES=6,7 python run_mlm.py train_secbert_modern_nle.json only_ling_mlm
# CUDA_VISIBLE_DEVICES=4,5 python run_mlm.py train_secbert_modern_nle.json mlm_on_all
# CUDA_VISIBLE_DEVICES=6,7 python run_mlm.py train_secbert_modern_nle.json semi_ling_no_tok
# CUDA_VISIBLE_DEVICES=4,5 python run_mlm.py train_secbert_modern_nle.json replace_all

### Full Pretraining
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py train_secbert_modern_full.json semi_ling_mlm
