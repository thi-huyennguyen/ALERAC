import utils
import torch
import torch.nn as nn
import random
import transformers
import numpy as np
import time
from transformers import AdamW, AutoModel

class CLSClassifier(nn.Module):
    """base model"""
    def __init__(self, model_config = "vinai/bertweet-base", hidden_size = 768, num_classes = 6):
        super(CLSClassifier, self).__init__()
        self.classifier = AutoModel.from_pretrained(model_config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.classifier.config.hidden_size, num_classes, bias = True)
    
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, return_dict=None):
        outputs = self.classifier(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict = return_dict)
        outputs = self.linear(self.dropout(outputs[1]))
        return outputs

class CLSTrainer:
    def __init__(self, args, input_ids, attention_mask, cls_labels, num_classes):
        self.args = args
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.cls_labels = cls_labels 
        self.num_classes = num_classes
        self.model = None
        self.optimizer = None

    def predict(self, input_ids, attention_mask, labels = None, cls_criterion = None, batch_size = 128):
        """make prediction"""
        self.model.eval()

        total_loss = 0
        cls_outs = []
        cls_probs = []
        batch_num = 0
        with torch.no_grad():
            for batch_start in range(0, len(input_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start : batch_end]
                batch_attention_mask = attention_mask[batch_start : batch_end]
                out = self.model(batch_input_ids.to(self.args.device), batch_attention_mask.to(self.args.device))
                if labels is not None and cls_criterion is not None:
                    batch_labels = labels[batch_start : batch_end]
                    loss = cls_criterion(out, batch_labels.to(self.args.device)).mean(dim = -1).sum()
                    total_loss += loss.item()
                
                out = nn.Softmax(dim = -1)(out)
                cls_outs += out.max(dim = -1).indices.cpu()
                cls_probs += out.cpu()
                batch_num += 1
        return cls_outs, cls_probs, total_loss/batch_num
            
    def set_device(self, device):
        if self.model is not None:
            self.model.to(device)

    def fit(self, input_ids, attention_mask, cls_labels, cls_criterion, batch_size = 128):
        """fit model"""
        self.model.train()
        train_loss = 0 
        epoch_indices = random.sample([i for i in range(len(input_ids))], len(input_ids))
        batch_num = 0
        for batch_start in range(0, len(input_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(input_ids))
            batch_input_ids = input_ids[batch_start : batch_end]
            batch_attention_mask = attention_mask[batch_start : batch_end]
            batch_labels = cls_labels[batch_start : batch_end]

            out = self.model(batch_input_ids.to(self.args.device), batch_attention_mask.to(self.args.device))
            self.optimizer.zero_grad()
            loss = cls_criterion(out, batch_labels.to(self.args.device)).mean(dim = -1).sum()
            train_loss += loss.item()
            batch_num +=1 
            loss.backward()

            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()
        return train_loss/batch_num
    
    def eval(self, train_indices = None, valid_indices = None, test_indices = None, train_input_ids = None, valid_input_ids = None, test_input_ids = None):
        """evaluation"""
        if train_indices is None and train_input_ids is None:
            raise ValueError("Both train_ids and train_indices are None")
        
        if train_input_ids is None:
            train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
        train_attention_mask, valid_attention_mask, test_attention_mask = self.attention_mask[train_indices], self.attention_mask[valid_indices], self.attention_mask[test_indices]
        train_labels, valid_labels, test_labels = self.cls_labels[train_indices], self.cls_labels[valid_indices], self.cls_labels[test_indices]

        self.model = CLSClassifier(self.args.model_config, hidden_size = self.args.cls_hidden_size, num_classes = self.num_classes)
        self.model.to(self.args.device)

        self.optimizer = AdamW(self.model.parameters(), self.args.lr, eps = 1e-8)
        num_batches = int(np.ceil(len(train_indices))/self.args.train_batch_size)
        total_step = num_batches * self.args.n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = total_step)

        cls_criterion = nn.CrossEntropyLoss(reduction = 'none')

        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_input_ids, train_attention_mask, train_labels, cls_criterion, self.args.train_batch_size)

            cls_outs, cls_probs, valid_loss = self.predict(valid_input_ids, valid_attention_mask, valid_labels, cls_criterion, self.args.test_batch_size)
            print("Epoch: %d, train loss: %.3f, valid loss: %.3f, time: %.3f" %(epoch, train_loss, valid_loss, time.time()-begin_time))


        # print("..................Training ends!..................")
       