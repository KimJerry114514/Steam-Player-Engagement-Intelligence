#!/bin/bash
# Final classifier (ft_large_v2): fine-tune multilingual-e5-large on the 9,720 labeled reviews, then score ALL negative
# reviews ("rest" = data/neg_text.parquet). Exact settings of the 2026-09-27 run. Needs a CUDA GPU.
# Inputs in gpu_work/data/: labels_train_v2.parquet, label_items_v2.json (see ../outputs), neg_text.parquet (01b).
export PYTHONIOENCODING=utf-8 FT_DIR=ft_large_v2 FT_MODEL=intfloat/multilingual-e5-large FREEZE_LAYERS=12 BS=16 \
       MAXLEN_TRAIN=192 EPOCHS=4 LR=2e-5 MAXLEN_INFER=128 LABELS=labels_train_v2.parquet ITEMS=label_items_v2.json
python w20_finetune.py rest
