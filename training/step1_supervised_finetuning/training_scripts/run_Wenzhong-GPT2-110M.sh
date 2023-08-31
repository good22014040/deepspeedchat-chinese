#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# running command
# cd Delta-Chat/training/step1_supervised_finetuning
# bash training_scripts/run_Wenzhong-GPT2-110M.sh ../../output/actor-models/Wenzhong-GPT2-110M
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path local/sft_jsonfile \
   --data_split 1,0,0 \
   --model_name_or_path IDEA-CCNL/Wenzhong-GPT2-110M \
   --human_text "\\\n\\\n人類:" \
   --assistant_text "\\\n\\\n助理:" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 8 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
