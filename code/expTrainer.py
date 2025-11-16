import torch
import torch.nn as nn
import transformers
from transformers import AdamW, AutoModel
import numpy as np 
from sklearn.metrics import f1_score
import time
import utils
import random

class EXPClassifier(nn.Module):
    """ base model """
    def __init__(self, model_config = 'vinai/bertweet-base', exp_hidden_size = 64):
        super(EXPClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_config)
        self.exp_hidden_size = exp_hidden_size
        self.exp_gru = nn.GRU(self.base_model.config.hidden_size, exp_hidden_size)
        self.exp_linear = nn.Linear(exp_hidden_size, 1, bias = True)
        self.exp_out = nn.Sigmoid()

    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask = attention_mask)[0]
        return self.exp_out(self.exp_linear(self.exp_gru(outputs)[0])).squeeze() * attention_mask


class EXPTrainer:
    def __init__(self, args, input_ids, attention_mask, exp_labels, exp_slides):
        self.args = args
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.exp_labels = exp_labels
        self.exp_slides = exp_slides
        self.model = None
        self.optimizer = None

    def fit(self, input_ids, attention_mask, exp_labels, exp_criterion, batch_size = 128):
        """fit model"""
        self.model.train()
        total_loss = 0
        epoch_indices = random.sample([i for i in range(len(input_ids))], len(input_ids))
        batch_num = 0

        for batch_start in range(0, len(input_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(input_ids))
            batch_indices = epoch_indices[batch_start : batch_end]
            batch_input_ids = input_ids[batch_indices]
            batch_attention_mask = attention_mask[batch_indices]
            batch_exp_labels = exp_labels[batch_indices]

            exp_preds = self.model(batch_input_ids.to(self.args.device), batch_attention_mask.to(self.args.device))
            
            idx = batch_exp_labels.sum(dim = -1) >=0
            if not idx.any():
                continue
            loss = exp_criterion(exp_preds[idx, :], batch_exp_labels[idx, :].to(self.args.device).float()).mean(dim = -1).sum()
            
            self.optimizer.zero_grad()
            total_loss += loss.item()
            batch_num += 1
            loss.backward()
            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()
        return total_loss/batch_num

    def predict(self, input_ids, attention_mask, exp_labels = None, exp_criterion=None, batch_size = 128):
        """make prediction"""
        self.model.eval()
        exp_preds = []
        exp_probs = []
        total_loss = 0
        batch_num = 0

        with torch.no_grad():
            for batch_start in range(0, len(input_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start : batch_end]
                batch_attention_mask = attention_mask[batch_start : batch_end]

                exp_outs = self.model(batch_input_ids.to(self.args.device), batch_attention_mask.to(self.args.device))

                if exp_labels is not None and exp_criterion is not None:
                    batch_exp_labels = exp_labels[batch_start : batch_end]
                    loss = exp_criterion(exp_outs, batch_exp_labels.to(self.args.device).float()).mean(dim = -1).sum()
                    total_loss += loss.item()
                
                exp_probs += exp_outs.cpu()
                exp_outs = torch.round(exp_outs).long().cpu()
                exp_preds += exp_outs
                batch_num +=1 
        
        return exp_preds, exp_probs, total_loss/batch_num

    def set_device(self, device):
        if self.model is not None:
            self.model.to(device)

    def eval(self, train_indices, valid_indices, test_indices, train_exp_labels= None):
        """train, eval, test"""
        train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
        train_attention_mask, valid_attention_mask, test_attention_mask = self.attention_mask[train_indices], self.attention_mask[valid_indices], self.attention_mask[test_indices]
        train_exp_slides, valid_exp_slides, test_exp_slides = self.exp_slides[train_indices], self.exp_slides[valid_indices], self.exp_slides[test_indices]
        valid_exp_labels, test_exp_labels = self.exp_labels[valid_indices], self.exp_labels[test_indices]
        if train_exp_labels is None:
            train_exp_labels = self.exp_labels[train_indices]
        
        #initialize base model
        self.model = EXPClassifier(model_config = self.args.model_config, exp_hidden_size = self.args.exp_hidden_size)
        self.model.to(self.args.device)

        n_batches = int(np.ceil(len(train_indices)/self.args.train_batch_size))

        self.optimizer = AdamW(self.model.parameters(), lr = self.args.lr, eps = 1e-8)
        total_step = n_batches * self.args.n_epochs 

        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_step)

        # exp_criterion = nn.BCELoss(reduction = 'none')
        exp_criterion = utils.resampling_rebalanced_crossentropy(seq_reduction = 'mean')

        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_input_ids, train_attention_mask, train_exp_labels, exp_criterion, self.args.train_batch_size)

            #evaluate on validation set
            exp_pred_labels, exp_probs_labels, valid_loss = self.predict(valid_input_ids, valid_attention_mask,
                        valid_exp_labels, exp_criterion, self.args.test_batch_size)
            
            exp_true = utils.max_pooling(valid_exp_labels, valid_exp_slides)
            exp_pred = utils.max_pooling(exp_pred_labels, valid_exp_slides)

            exp_f1 = np.mean([f1_score(y_pred, y_true) for y_pred, y_true in zip(exp_true, exp_pred)])


            print("Epoch: %d, train_loss: %.3f, valid_loss: %.3f, exp_f1: %.3f, time: %.3f" %(epoch,
                            train_loss, valid_loss, exp_f1, time.time() - begin_time))
            
        



