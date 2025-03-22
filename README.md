# CLQT


### Pretrain Model 
## bert-base-mulitlingual-cased 
## xlm-roberta-base
## Contrastive Learning


### quick start 
## python main.py --do_train --do_contrastive_pretrain --exp_type acs_mtl --tfm_type xlmr --model_name_or_path xlm-roberta-base --contrastive_data_dir ./data/contrastive/


### conda install core packages
torch==1.10.1
numpy==1.23.0
transformers==4.23.1 
sentencepiece==0.1.97
tokenizer==0.13.1
sacremoses==0.0.53
tqdm==4.64.1
