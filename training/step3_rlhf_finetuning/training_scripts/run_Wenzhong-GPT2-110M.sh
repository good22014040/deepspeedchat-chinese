#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# running command
# cd Delta-Chat/training/step3_rlhf_finetuning
# bash training_scripts/run_Wenzhong-GPT2-110M.sh ../../output/actor-models/Wenzhong-GPT2-110M ../../output/reward-models/Wenzhong-GPT2-110M 0 0 ../../output/ppo-models/Wenzhong-GPT2-110M
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi
mkdir -p $OUTPUT

Actor_Lr=1e-5
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path local/ppo_jsonfile \
   --data_split 0,0,10 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --unsup_coef 1 \
   --unsupervised_dataset_name pleisto/wikipedia-cn-20230720-filtered \
   --unsupervised_dataset_config_name pleisto/wikipedia-cn-20230720-filtered \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 2 \
   --per_device_training_batch_size 2 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 768 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 32 \
   --disable_actor_dropout \
   --num_warmup_steps 50 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_ema \
   --output_dir $OUTPUT \
   --print_answers \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
    &> $OUTPUT/training.log