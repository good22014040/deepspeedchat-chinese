# generate training data
cd data
echo "--generate training data"
bash generate_data.sh

# sft
cd ../training/step1_supervised_finetuning
echo "--step1_supervised_finetuning"
bash training_scripts/run_Wenzhong-GPT2-110M.sh ../../output/actor-models/Wenzhong-GPT2-110M

# rm
cd ../step2_reward_model_finetuning
echo "--step2_reward_model_finetuning"
bash training_scripts/run_Wenzhong-GPT2-110M.sh ../../output/reward-models/Wenzhong-GPT2-110M

# ppo
cd ../step3_rlhf_finetuning
echo "--step3_rlhf_finetuning"
bash training_scripts/run_Wenzhong-GPT2-110M.sh \
    ../../output/actor-models/Wenzhong-GPT2-110M \
    ../../output/reward-models/Wenzhong-GPT2-110M \
    0 0 ../../output/ppo-models/Wenzhong-GPT2-110M