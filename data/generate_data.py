import argparse
import json
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Path to tokenizer from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        '--train_data_path',
        type=str,
        default=None,
        help='the path of training data'
    )
    parser.add_argument(
        '--eval_data_path',
        type=str,
        default=None,
        help='the path of eval data'
    )
    parser.add_argument(
        "--human_text",
        type=str,
        default='\n\n人類:',
        help=
        "the beginning of human prompt",
    )
    parser.add_argument(
        "--assistant_text",
        type=str,
        default='\n\n助理:',
        help=
        "the beginning of assistant response",
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'split data. For example the split `2,4,4` will use 20% of data for sft, 40% for rm and 40% for ppo.'
    )
    parser.add_argument('--split',
                        action='store_true',
                        help='Whether to split the data')
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help="max length of tokenized data ")
    parser.add_argument(
        '--prompt_max_length',
        type=int,
        default=768,
        help="prompt max length of tokenized data ")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.train_data_path != None and args.eval_data_path!= None:
        # use local data
        with open(args.train_data_path) as f:
            train_datas = json.load(f)
        with open(args.eval_data_path) as f:
            eval_datas = json.load(f)
    else:
        # use dataset, need to login through "huggingface-cli login"
        datas = load_dataset('drc-8/chinese-rm-static')
        train_datas = datas['train']
        eval_datas = datas['test']


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    generate_data(args, train_datas, tokenizer, "train.json")
    generate_data(args, eval_datas, tokenizer, "eval.json")

def generate_data(args, datas, tokenizer, save_path):
    # remove too long data and add eos token after assistant response
    new_datas = []
    eos_token = tokenizer.eos_token
    human_text = args.human_text.replace("\\n", "\n")
    assistant_text = args.assistant_text.replace("\\n", "\n")
    prompt_max_length = args.prompt_max_length
    max_length = args.max_length

    for i, data in enumerate(tqdm(datas)):
        if len(data['prompt']) == 0 or len(data['chosen']) == 0 or len(data['rejected']) == 0:# no data
            continue
        elif data["prompt"].count(human_text) != data["prompt"].count(assistant_text): # format error
            continue
        # add eos token after assistant response
        data['prompt'] = data['prompt'].replace(human_text, eos_token+human_text)
        if data['prompt'][:len(eos_token)] == eos_token:
            data['prompt'] = data['prompt'][len(eos_token):]
        else:
            continue
        
        # remove too long data
        prompt = tokenizer(data['prompt'], return_tensors='pt').input_ids
        chosen = tokenizer(data['chosen'], return_tensors='pt').input_ids
        rejected = tokenizer(data['rejected'], return_tensors='pt').input_ids
        if prompt.size(1) + chosen.size(1) > max_length or \
            prompt.size(1) + rejected.size(1) > max_length or \
            prompt.size(1) > prompt_max_length:
            continue
        new_datas.append(data)
    
    with open(save_path, "w") as f:
        json.dump(new_datas, f, ensure_ascii=False, indent = 4)

    if args.split:
        split_data(args, new_datas, save_path)

def split_data(args, datas, save_path):
    split = [int(s) for s in (args.data_split).split(",")]
    split_name = ["sft", "rm", "ppo"]

    random.shuffle(datas)
    length = len(datas)
    split_sum =sum(split)
    start = 0
    for i in range(len(split)):
        end = start + split[i]
        
        os.makedirs(split_name[i], exist_ok=True)
        with open(f"{split_name[i]}/{save_path}", "w") as f:
            json.dump(datas[start*length//split_sum:end*length//split_sum], f, ensure_ascii=False, indent=4)
        
        start = end


if __name__ == "__main__":
    main()