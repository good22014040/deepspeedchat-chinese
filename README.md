程式基於[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)進行修改，原始README.md請參照README_deepspeed.md

## 訓練
在目錄底下輸入以下指令即可訓練IDEA-CCNL/Wenzhong-GPT2-110M，也可自行替換成其他模型
```shell=
$ bash train.sh
```
# 測試
在目錄底下輸入以下指令即可測試訓練好的IDEA-CCNL/Wenzhong-GPT2-110M模型
```shell=
$ python chat.py --path output/ppo-models/Wenzhong-GPT2-110M/actor_ema \
                --tokenizer_name_or_path IDEA-CCNL/Wenzhong-GPT2-110M
```

## 資料前處理
train.sh中有透過data/generate_data.py產生訓練資料  
處理內容包含
+ 清除過長資料，truncate過的資料並不是很適合用來訓練
+ 在助理的回覆後新增eos token
+ 將清理過的資料分配到sft, rm, ppo三個資料夾以供後續訓練

執行時會從drc-8/chinese-rm-static下載資料，如果沒有權限也可讀取本地檔案，格式請依照example_format，並透過train_data_path/eval_data_path指定參數
|Args|Explanation|
|-|-|
|--tokenizer_name_or_path|使用甚麼模型的tokenizer|
|--train_data_path|訓練資料的路徑|
|--eval_data_path|評估資料的路徑|
|--human_text|人類prompt前的內容|
|--assistant_text|模型response前的內容|
|--data_split|資料怎麼分配，3個數值代表sft, rm, ppo階段得到的數據比例|
|--split|要不要切成sft, rm, ppo|
|--max_length|prompt+response的最大長度|
|--prompt_max_length|prompt的最大長度|

## 修改的code
### training/step1_supervised_finetuning/main.py
+ parser新增參數human_text/assistant_text，用來指定資料集人類及助理的開頭  
    ex : 
    >\n\n人類:你好嗎？\n\n助理:我很好。\n\n人類:晚安\n\n助理:晚安，瑪卡巴卡
    
    human_text = \n\n人類  
    assistant_text = \n\n助理  
    但要注意在traing_script中要寫成"\\\\\\n\\\\\\n人類"(每個換行符號3根斜線)才能正常讀取到換行符號
+ 刪除所有print_throughput，gpt2不支援，額外修改效益不大
### training/step2_reward_model_finetuning/main.py
+ 每個epoch評估時輸出train acc，確定有無過擬合
### training/step3_rlhf_finetuning/main.py
+ 刪除time與print_throughput_step3，對訓練沒影響而且gpt2不支援

### training/utils/data/data_utils.py
+ parser新增sft, rm, ppo的資料集讀取，方便對單獨某個資料集進行修正
+ PromptDataset中新增label_dataset，方便step1訓練時忽視掉部分內容，原本label就是input_ids
+ create_dataset_split中新增label_dataset，並在step1對label進行處理，將助理的回覆以外的部分設成-100  
    對應路徑都新增human_text/assistant_text參數  
    包含
    - create_prompt_dataset
    - create_dataset
    - create_dataset_split
+ get_unsupervised_data新增中文wiki的特殊處理

### training/utils/data/raw_datasets.py
+ 新增LocalJsonFileSplitDataset，分別讀取3種不同的分割(sft, rm, ppo)

### training/utils/model/reward_model.py
+ forward修改取得最後位置的方式，改成找最後一個不是padding的token
+ forward_value也進行相似修正

### training/step3_rlhf_finetuning/ppo_trainer.py
+ \_generate_sequence中新增min_length與eos_token，基本上沒影響，但模型訓練得很糟時可以避免報錯  
    透過hasattr(self.actor_model,"model")先確定屬性存不存在，基本上只有gpt2需要
+ generate_experience中將prompt前的padding先移到後方，然後再取得reward和value，以避免padding的位置影響預測效果，之後value再移動對應數量到前方
+ train_rlhf中取得新的value時也進行上面那種移動