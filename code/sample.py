import numpy as np 
import pandas as pd

import random 
random.seed(12345)
from sklearn.model_selection import train_test_split

human_input = "/home/nguyen/hnt/tweet_interpret_sum/datasets/labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_final2.csv"
gpt_input = "/home/nguyen/2024www/data/chatgpt/processed_nequake/all.csv"

human_data = pd.read_csv(human_input)
gpt_data = pd.read_csv(gpt_input)

print(human_data.shape)
print(human_data.columns)
human_data = human_data[['tweet_id', 'tweet_text', 'corrected_label', 'informative_content']]
human_data.columns = ['tweet_id', 'tweet_text', 'cls_labels', 'human_rationales']

print(gpt_data.shape)
print(gpt_data.columns)
gpt_data = gpt_data[['tweet_id', 'tweet_text', 'corrected_label', 'rationales']]
gpt_data.columns = ['tweet_id','tweet_text', 'cls_labels', 'gpt_rationales']

print("human: ", human_data['human_rationales'].apply(lambda x: len(str(x).split())).mean())
print("gpt:", gpt_data['gpt_rationales'].apply(lambda x: len(str(x).split())).mean())
for i in range(30):
    print("--> human:", human_data.iloc[i]['human_rationales'])
    # print("GPT:", gpt_data.iloc[i]['gpt_rationales'])

human_sampled, _ = train_test_split(
    human_data,
    train_size=100,
    stratify=human_data["cls_labels"],
    random_state=42
)

# These are the sampled indices
idx = human_sampled.index

print("human: ", human_sampled['cls_labels'].value_counts())

gpt_sampled = gpt_data.iloc[idx]
print("gpt: ", gpt_sampled['cls_labels'].value_counts())

human_sampled.to_csv('../data/human.csv', sep = '\t', index = False)
gpt_sampled.to_csv('../data/gpt.csv', sep = '\t', index = False)