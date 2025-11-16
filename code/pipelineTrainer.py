
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from expTrainer import EXPTrainer
from clsTrainer import CLSTrainer
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import copy
import utils

class Trainer:
    def __init__(self, args, data = None, cls_labels = None, exp_labels = None):
        self.args = args
        self.exp_labels = exp_labels
        self.cls_labels = cls_labels
        self.num_classes = len(set(cls_labels))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_config)

        self.data = np.array([self.tokenizer.cls_token + " " + x + " " + self.tokenizer.sep_token for x in data])
        
        # preprocess and convert data to tensor
        self.exp_labels = exp_labels
        self.cls_classes = len(set(cls_labels))
        self.cls_labels = torch.tensor(cls_labels, dtype = torch.long)
        self.tokenized_data, self.input_ids, self.attention_mask, self.tokenized_data_slides = \
            utils.tokenize_text(self.tokenizer, self.data, self.tokenizer.pad_token)
        self.exp_labels_mapping = utils.map_exp_labels(self.tokenizer, self.data, exp_labels)
        self.tokenized_data, self.input_ids = np.array(self.tokenized_data,dtype = object), torch.tensor(self.input_ids, dtype = torch.long)
        self.attention_mask, self.tokenized_data_slides = torch.tensor(self.attention_mask, dtype = torch.long), np.array(self.tokenized_data_slides,dtype = object)
        self.exp_labels_mapping = torch.tensor(self.exp_labels_mapping, dtype = torch.long)

        self.clsTrainer = CLSTrainer(args, self.input_ids, self.attention_mask, self.cls_labels, self.cls_classes)
        self.expTrainer = EXPTrainer(args, self.input_ids, self.attention_mask, self.exp_labels_mapping, self.tokenized_data_slides)


    def eval(self, train_indices = None, valid_indices = None, test_indices = None, idx_label_map = None):
        
        """train, eval, test"""
        train_data, valid_data, test_data = self.data[train_indices], self.data[valid_indices], self.data[test_indices]
        train_tokenized_data, valid_tokenized_data, test_tokenized_data = self.tokenized_data[train_indices], self.tokenized_data[valid_indices], self.tokenized_data[test_indices]
        train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
        train_attention_mask, valid_attention_mask, test_attention_mask = self.attention_mask[train_indices], self.attention_mask[valid_indices], self.attention_mask[test_indices]
        train_tokenized_data_slides, valid_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[valid_indices], self.tokenized_data_slides[test_indices]
        train_exp_labels, valid_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[valid_indices], self.exp_labels_mapping[test_indices]
        train_cls_labels, valid_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[valid_indices], self.cls_labels[test_indices]

        num_added_instances = int(0.1*len(train_data))
        results = []
        original_train_exp_labels = copy.deepcopy(train_exp_labels)

        # sample x% data without exp labels for initial training
        if self.args.ignored_exp_percent != 0:
            
            current_exp_indices, ignored_exp_indices = train_test_split([i for i in range(len(train_cls_labels))], 
                                                      test_size = self.args.ignored_exp_percent, random_state = self.args.random_state,
                                                      stratify = train_cls_labels)
            train_exp_labels[ignored_exp_indices] = -1
            

            # labels = train_cls_labels[current_exp_indices]
            # label_id, counts = labels.unique(return_counts = True)
            # for id, count in zip(label_id, counts):
            #     print("Label {} --{}".format(int(id), count))
            # input()
            while True:
                current_data_percent = 1-len(ignored_exp_indices)/len(train_data)
                print("++++++++++++++++++++Train size (%d-%d-%.2f)++++++++++++++++++++" %(len(train_data),len(current_exp_indices), current_data_percent))
                print(">> Phase 1 ____________________________________")
                self.expTrainer.eval(train_indices, valid_indices, test_indices, train_exp_labels)
                

                # extract predicted exp for cls classifier
                test_exp_preds, _, _ = self.expTrainer.predict(test_input_ids, test_attention_mask, batch_size=self.args.test_batch_size)
                test_exp_true = utils.max_pooling(test_exp_labels, test_tokenized_data_slides)
                test_exp_pred = utils.max_pooling(test_exp_preds, test_tokenized_data_slides)

                # evaluate on the test set
                exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true, test_exp_pred)])
                exp_r = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true, test_exp_pred)])
                exp_p = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true, test_exp_pred)])

                print(" ----------------------------------------------")
                print("|exp_f1: %.3f, exp_r: %.3f, exp_p: %.3f     |" %(exp_f1, exp_r, exp_p))
                print(" ----------------------------------------------")
        

                # replace predicted non-rationales with *
                train_exp_preds, train_exp_probs, _ = self.expTrainer.predict(train_input_ids, train_attention_mask, batch_size = self.args.test_batch_size)
                valid_exp_preds, valid_exp_probs, _ = self.expTrainer.predict(valid_input_ids, valid_attention_mask, batch_size = self.args.test_batch_size)


                star_id = self.tokenizer.convert_tokens_to_ids("*")
                special_tokens = [0, 1, 2]
                stacked_train_exp_preds = torch.stack(train_exp_preds)
                stacked_valid_exp_preds = torch.stack(valid_exp_preds)
                stacked_test_exp_preds = torch.stack(test_exp_preds)
                
                train_mask = (stacked_train_exp_preds == 0) & ~torch.isin(train_input_ids, torch.tensor(special_tokens))
                valid_mask = (stacked_valid_exp_preds == 0) & ~torch.isin(valid_input_ids, torch.tensor(special_tokens))
                test_mask = (stacked_test_exp_preds == 0) & ~torch.isin(test_input_ids, torch.tensor(special_tokens))
                
                train_input_ids_masked = train_input_ids.clone()
                train_input_ids_masked[train_mask] = star_id

                valid_input_ids_masked = valid_input_ids.clone()
                valid_input_ids_masked[valid_mask] = star_id

                test_input_ids_masked = test_input_ids.clone()
                test_input_ids_masked[test_mask] = star_id
                
                self.expTrainer.set_device('cpu')
                
                print(">> Phase 2 ____________________________________")
                self.clsTrainer.eval(train_indices, valid_indices, test_indices, train_input_ids_masked, valid_input_ids_masked, test_input_ids_masked)
                cls_preds, _, _ = self.clsTrainer.predict(test_input_ids_masked, test_attention_mask, batch_size = self.args.test_batch_size)
                self.clsTrainer.set_device('cpu')
                print(" ----------------------------------------------")
                cls_f1 = f1_score(cls_preds, test_cls_labels, average = 'macro')
                print("|cls_f1: %.3f                                  |" %(cls_f1))
                print(" ----------------------------------------------")
                results.append({"exp_f1": exp_f1, "exp_r": exp_r, "exp_p": exp_p, "cls_f1": cls_f1})
                self.clsTrainer.set_device('cpu')

                # select the next x% to add into the training set
                stacked_train_exp_probs = torch.stack(train_exp_probs)

                probs = 1 - (stacked_train_exp_preds * stacked_train_exp_probs).sum(axis = 1) / stacked_train_exp_preds.sum(axis = 1)

                if ignored_exp_indices is None:
                    break
                if len(current_exp_indices)/len(original_train_exp_labels) >= 0.9:
                    break
                
                probs[current_exp_indices] = -1 # assign probs for already selected items = 1
                probs = torch.nan_to_num(probs, -2)

                next_indices = [int(i) for i in probs.topk(num_added_instances).indices]

                ignored_exp_indices = [idx for idx in ignored_exp_indices if idx not in next_indices]
                current_exp_indices += next_indices 
                # print("new ignored list: %d, considering list: %d" %(len(ignored_exp_indices), len(current_exp_indices)))

                # update exp labels with new labels
                for idx in range(len(train_exp_labels)):
                    if idx in next_indices:
                        train_exp_labels[idx] = original_train_exp_labels[idx]
                #print distribution of labels
                labels = train_cls_labels[current_exp_indices]
                label_id, counts = labels.unique(return_counts = True)
                # for id, count in zip(label_id, counts):
                #     print("Label: {} --{}".format(int(id), count))
              
                

        return results


