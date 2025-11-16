from tweet_preprocessing import tokenizeRawTweetText
import re
import numpy as np 
import torch


def preprocess_text(text):
    text = tokenizeRawTweetText(text)
    text = text.replace("\\n", " ")
    text = re.sub("^`|^\'|^\"|\'$|\"$|^(rt )", '', text).strip()
    text = re.sub("^: ", '', text)
    text = re.sub(" +", ' ', text).strip()
    return text

def tokenize_text(tokenizer, sents, padding_token = '<pad>'):
    """
        :param sents: list[str], list of untokenized sentences
        @return: tokenized_lists: list[str], list of tokenized tokens
        @return: tokens_id_list_padded: list[int], list of ids of tokenized tokens
        @return: tokens_spans, list[(int, int)], list of start_span, end_span to recover original unknown words
                    
    """
    tokens_list = []
    tokens_spans = []
    for sent in sents:
        tokens = []
        spans = []
        start_w = 0
        for w in sent.split(" "):
            tokenized = tokenizer.tokenize(w)
            if start_w+len(tokenized)>= 128:
                tokens.extend(tokenizer.tokenize(tokenizer.sep_token))
                end_w = len(tokens)
                spans.append((start_w, end_w))
                break
            tokens.extend(tokenized)
            end_w = len(tokens)
            spans.append((start_w, end_w))
            start_w = end_w
        tokens_list.append(tokens)
        tokens_spans.append(spans)
    # pad sentences
    tokens_list_padded = pad_sents(tokens_list, padding_token)
    attention_masks  = np.asarray(tokens_list_padded) != padding_token
    tokens_id_list_padded =[tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    return tokens_list, tokens_id_list_padded, attention_masks, tokens_spans


def pad_sents(sents, pad_token):
    """
        :param sents: list[list[str]] list of tokenized sentences
        :param pad_token: int, pad token id
        @returns sents_padded: list[list[int]], list of tokenized sentences with padding shape(batch_size, max_sentence_length)
    """
    sents_padded = []
    max_len = max(len(s) for s in sents)
    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)
    return sents_padded

def max_pooling(y_values, data_slides, prob = False):
    """
        merge token-level labels to obtain labels at word-level
    """
    pooled_values = []
    
    for y, y_slides in zip(y_values, data_slides):
        try: 
            pooled_y = []
            for tup in y_slides:
                if prob == True:
                    pooled_y.append(round(float(max(y[tup[0]:tup[1]])), 2))
                else:
                    pooled_y.append(int(max(y[tup[0]:tup[1]])))
            pooled_values.append(pooled_y)
        except Exception as e:
            print("Exception: ")
            print(e)
    return pooled_values

def map_exp_labels(tokenizer, sents, exp_labels):
    """
        Assign exp labels at token-level
        :param sents: list[str], list of input sentences
        :param exp_labels: list[list[str]], list of explanation snippets of input sentences
        @return exp_labels_mapping: list[int], list of 0/1 to specify whether the token is a part of explanation
        
    """
    tokenized_sents = [' '.join(tokenizer.tokenize(sent)).strip() for sent in sents]
    
    tokenized_exps = [[' '.join(tokenizer.tokenize(exp)).strip() for exp in exps] for exps in exp_labels]
    exp_labels = [text_to_exp_labels(sent, exp) for sent, exp in zip(tokenized_sents, tokenized_exps)]
    # padding 0
    max_length = max([len(sent.split(" ")) for sent in tokenized_sents])
    exp_labels = [label+[0]*(max_length-len(label)) for label in exp_labels]
    return exp_labels

def text_to_exp_labels(org_text, explan_text):
    """
        param org_text: str, original text
        param explan_text: list[str], explanation snippets
        @return list[int], list of 0/1 values to label whether each token in org_text is a part of explan_text 
    """
    labels = org_text

    #sort rationales in order of length
    exp = {x:len(x.split(' ')) for x in explan_text}
    exp = {k: v for k, v in sorted(exp.items(), key=lambda x: x[1], reverse=True)}
    try:
        
        for chunk, lenx in exp.items():
            labels = re.sub(re.escape(chunk), 'U '*len(chunk.split( )), labels)
        labels = re.sub('[^U ]', '0', labels)
        labels = re.sub('  ', ' ', labels).strip().split(" ")
        labels = [1 if i == 'U' else 0 for i in labels]
    except Exception as e:
        print("Error...")
        print(e)
        
    return labels

def resampling_rebalanced_crossentropy(seq_reduction = 'none'):
    def loss(y_pred, y_true):
        prior_pos = torch.mean(y_true, dim=-1, keepdims=True)
        prior_neg = torch.mean(1-y_true, dim=-1, keepdim=True)
        eps=1e-10
        weight = y_true / (prior_pos + eps) + (1 - y_true) / (prior_neg + eps)
        ret =  -weight * (y_true * (torch.log(y_pred + eps)) + (1 - y_true) * (torch.log(1 - y_pred + eps)))
        if torch.mean(ret, dim = -1).shape[0] == 0:
            print("y_true: ", y_true)
            print("y_pred: ", y_pred)
            print(torch.mean(ret, dim=-1))
        if seq_reduction == 'mean':
            return torch.mean(ret, dim=-1)
        elif seq_reduction == 'none':
            return ret
    return loss