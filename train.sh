#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

#Train the model only with LM
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name GPT2 --incontext > train_log_focus/e2_L20_LM_gpt.log &
# CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name BART --incontext --wandb --train_batch_size=2 > train_log_focus/e2_L20_LM_bart.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name transformer-decoder --incontext > train_log_focus/e2_L20_LM_decoder.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name transformer-encdec --incontext > train_log_focus/e2_L20_LM_encdec.log &
rm -rf ./models/train_focus_BART_E2_L10/*
#Train the model with Knowledge Selection, Persona Selection and Language Modeling
# CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name GPT2 --incontext --debug=True > train_log_focus/e2_L20_gpt.log &
CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --debug --train_batch_size=2  > train_log_focus/e2_L20_bart.log &
# CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext --wandb --train_batch_size=2  > train_log_focus/e2_L20_bart.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name transformer-decoder --incontext > train_log_focus/e2_L20_decoder.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name transformer-encdec --incontext > train_log_focus/e2_L20_encdec.log &

#Train the model w/o Knowledge Selection
#CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --kn_coef 0 --flag E2_L10_WO_KS --model_name BART --incontext > train_log_focus/e2_L20_bart_WO_KS.log &
#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --kn_coef 0 --flag E2_L10_WO_KS --model_name GPT2 --incontext > train_log_focus/e2_L20_gpt_WO_KS.log &

#Train the model w/o Persona Selection
# CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --ps_coef 0 --flag E2_L10_WO_PS --model_name GPT2 --incontext > train_log_focus/e2_L20_gpt_WO_PS.log &
# CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py --n_epochs 2 --lm_coef 10 --ps_coef 0 --flag E2_L10_WO_PS --model_name BART --incontext --wandb --train_batch_size=2 > train_log_focus/e2_L20_bart_WO_PS.log &

echo




