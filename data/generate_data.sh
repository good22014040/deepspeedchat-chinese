python generate_data.py \
    --tokenizer_name_or_path IDEA-CCNL/Wenzhong-GPT2-110M \
    --human_text '\n\n人類:' \
    --assistant_text '\n\n助理:' \
    --data_split 2,4,4 \
    --split \
    --max_length 1024 \
    --prompt_max_length 768 \