
import argparse
import utils
import pandas as pd
from pipelineTrainer import Trainer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import f1_score
import numpy as np
import torch 
import random
import warnings
warnings.filterwarnings("ignore")
random.seed(12345)
np.random.seed(67891)
torch.manual_seed(54321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', type = str, default = '../data/gpt.csv')
    parser.add_argument('-random_state', type = int, default = 12)
    parser.add_argument('-model_config', type = str, default = 'vinai/bertweet-base')
    parser.add_argument('-cls_hidden_size', type = int, default = 128)
    parser.add_argument('-exp_hidden_size', type = int, default = 128)
    parser.add_argument('--ignored_exp_percent', type = float, default = 0.9)
    parser.add_argument('-lr', type = float, default = 2e-5, help = 'learning rate')
    parser.add_argument('-max_len', type = int, default = 128)
    parser.add_argument('-n_epochs', type = int, default = 10)
    parser.add_argument('-n_folds', type = int, default = 5)
    parser.add_argument('-patience', type = int, default = 3)
    parser.add_argument('-test_size', type = int, default = 0.15)
    parser.add_argument('-train_batch_size', type = int, default = 16)
    parser.add_argument('-test_batch_size', type = int, default = 128)
    parser.add_argument('-sep_exp_token', type = str, default = ' _sep_exp_token_ ')
    parser.add_argument('-id_col', type = str, default = 'tweet_id')
    parser.add_argument('-text_col', type = str, default = 'tweet_text')
    parser.add_argument('-label_col', type = str, default = 'cls_labels')
    parser.add_argument('-exp_col', type = str, default = 'gpt_rationales')
    parser.add_argument('-device', type = str, default = 'cuda')
    parser.add_argument('-event_type', type = str, default = 'quake', help = 'quake/typhoon')


    label_event_map = {'quake': ['injured_or_dead_people', 'affected_people_and_evacuations', \
                        'infrastructure_and_utilities_damage', 'rescue_volunteering_and_donation_effort',  \
                        'other_useful_information', 'not_related_or_irrelevant'],
                        'typhoon': ['caution_and_advice', 'affected_people_and_evacuations', \
                        'infrastructure_and_utilities_damage', 'rescue_volunteering_and_donation_effort',  \
                        'other_useful_information', 'not_related_or_irrelevant']}
    prepro_exp = 'prepro_exp'
    prepro_text = 'prepro_text'
    prepro_label = 'prepro_label'

    args= parser.parse_args()
    labels = label_event_map[args.event_type]
    label_idx_map = {label:i for i, label in enumerate(labels)}
    idx_label_map = {idx:label for label, idx in label_idx_map.items()}

    data = pd.read_csv(args.input_path)

    #preprocess exp
    data[prepro_exp] = data[args.exp_col].apply(lambda x: str(x).strip().replace('[SEP]', args.sep_exp_token))
    data[prepro_exp] = data[prepro_exp].apply(lambda x: utils.preprocess_text(x))
    data[prepro_exp] = data[prepro_exp].apply(lambda x: [y.strip() for y in x.split(args.sep_exp_token)])

    #preprocess text
    data[prepro_text] = data[args.text_col].apply(lambda x: utils.preprocess_text(x))

    #preprocess label
    data[prepro_label] = data[args.label_col].apply(lambda x: label_idx_map[x])

    data.drop_duplicates(subset = prepro_text, inplace = True)
    print("--> Data: ", data.shape)

    # initialize models
    text_data = np.array(data[prepro_text])
    exp_data = np.array(data[prepro_exp])
    cls_data = np.array(data[prepro_label])
    
    trainer = Trainer(args, text_data, cls_data, exp_data)
     #cross-validate 
    kfold = StratifiedShuffleSplit(n_splits = args.n_folds, test_size = args.test_size, random_state = args.random_state)
    fold = 0
    fold_outs = []
    for train_indices, remain_indices in kfold.split(text_data, cls_data):
        valid_indices, test_indices = train_test_split(remain_indices, test_size = 0.5, random_state = args.random_state,
                                    stratify = cls_data[remain_indices])
        print("==============================FOLD {}==============================".format(fold))
        out = trainer.eval(train_indices, valid_indices, test_indices, idx_label_map)
        fold_outs.append(out)
        fold+=1
    print("===================================================================")
    print("__________________________Final evaluation_________________________")
    for i in range(len(fold_outs[0])):
        cls_f1 = sum(out[i]['cls_f1'] for out in fold_outs)/len(fold_outs)
        exp_f1 = sum(out[i]['exp_f1'] for out in fold_outs)/len(fold_outs)
        print("--> labeled rationales: {}%, cls_f1: {}, exp_f1: {}\n".format(i*10, cls_f1, exp_f1))

    