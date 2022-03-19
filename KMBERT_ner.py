from pretrain.examples.extract_features import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from kumc_dataset import InputDataset
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import numpy as np
import pandas as pd
import scipy.stats
import pickle, re
import torch
import json
import os
import csv
import argparse

def to_np(t):
    return t.cpu().detach().numpy()

parser = argparse.ArgumentParser(description='KUMC NER')
parser.add_argument('--pretrained', type=str)
args = parser.parse_args()

pretrained_model = args.pretrained

tag_dict = dict()
tag_dict['B-BD'] = 0
tag_dict['I-BD'] = 0
tag_dict['B-DS'] = 1
tag_dict['I-DS'] = 1
tag_dict['B-ST'] = 2
tag_dict['I-ST'] = 2
tag_dict['O'] = 3

raw_tag_dict = dict()
raw_tag_dict['B-BD'] = 0
raw_tag_dict['I-BD'] = 4
raw_tag_dict['B-DS'] = 1
raw_tag_dict['I-DS'] = 5
raw_tag_dict['B-ST'] = 2
raw_tag_dict['I-ST'] = 6
raw_tag_dict['O'] = 3


# HYPERPARAMETERS
device = 'cpu' # gpu usage will be updated
epoch = 5
learning_rate = 2e-5
batch_size = 32
seq_len = 128
num_label = 4
savepath = "res_ner/"

if pretrained_model=='kmbert':
    tokenizer = BertTokenizer(vocab_file="./kmbert/kmbert_vocab.txt", do_lower_case=False)
    model = BertForTokenClassification.from_pretrained('./kmbert/', num_labels=num_label)
    savepath += "kmbert/"
if pretrained_model=='kmbert_vocab':
    tokenizer = BertTokenizer(vocab_file="./kmbert_vocab/kmbert_vocab.txt", do_lower_case=False)
    model = BertForTokenClassification.from_pretrained('./kmbert_vocab/', num_labels=num_label)
    savepath += "kmbert_vocab/"

if not os.path.isdir(savepath):
    os.mkdir(savepath)

model.to(device)
loss_fct = CrossEntropyLoss(ignore_index=-1).to(device)



def read_ner_dataset(filename):
    reader = open(filename, 'r', encoding='utf-8')
    data = []
    for line in reader:
        data.append(line)
    ner_ori = []
    for i, v in enumerate(data):
        if v == '\n':
            ner_ori.append(['-', '-', '-'])
        elif '-DOCSTART-' in v:
            continue
        else:
            ner_ori.append(v.strip().split(' '))
    ner_ori = np.array(ner_ori)
    sentence, tag, raw_tag = [], [], []
    tmp_sent, tmp_tag, tmp_raw_tag = [], [], []
    for i, v in enumerate(ner_ori):
        if v[0] == '-' and v[1] == '-' and v[2] == '-':
            if tmp_sent or tmp_tag:
                sentence.append(tmp_sent)
                tag.append(tmp_tag)
                raw_tag.append(tmp_raw_tag)
            tmp_sent, tmp_tag, tmp_raw_tag = [], [], []
            continue
        tokens = tokenizer.tokenize(v[0])
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tmp_sent.extend(token_ids)
        tmp_tag.extend(list(np.repeat(tag_dict[v[2]], len(tokens))))
        tmp_raw_tag.extend(list(np.repeat(raw_tag_dict[v[2]], len(tokens))))
    return sentence, tag, raw_tag

def get_matched(pred, label, ignore_idx=-1):
    idx = label!=ignore_idx
    matched = ((pred==label)[idx]).sum()
    target = idx.sum()
    return matched, target

savepath_ = savepath
for fold in range(5):
    filename_train = 'Path for NER train dataset %d (5-fold cross validation)'%(fold+1)
    filename_test = 'Path for NER test dataset %d (5-fold cross validation)'%(fold+1)
    savepath = savepath_ + "%d/"%(fold+1)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)


    if pretrained_model=='kmbert':
        model = BertForTokenClassification.from_pretrained('./kmbert/', num_labels=num_label)
    if pretrained_model=='kmbert_vocab':
        model = BertForTokenClassification.from_pretrained('./kmbert_vocab/', num_labels=num_label)


    model.to(device)

    print("======= Fold %d ========" % (fold+1))

    sent_train, tag_train, raw_tag_train = read_ner_dataset(filename_train)
    sent_test, tag_test, raw_tag_test = read_ner_dataset(filename_test)

    # DATA
    token_cls = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    token_sep = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

    input_ids_train = pad_sequences(sent_train, maxlen=seq_len, dtype="long", truncating="post", padding="post")
    attention_masks_train = np.array([[float(i > 0) for i in ii] for ii in input_ids_train])
    labels_train = pad_sequences(tag_train, maxlen=seq_len, dtype="long", truncating="post", padding="post", value=-1)
    labels_raw_train = pad_sequences(raw_tag_train, maxlen=seq_len, dtype="long", truncating="post", padding="post", value=-1)

    input_ids_test = pad_sequences(sent_test, maxlen=seq_len, dtype="long", truncating="post", padding="post")
    attention_masks_test = np.array([[float(i > 0) for i in ii] for ii in input_ids_test])
    labels_test = pad_sequences(tag_test, maxlen=seq_len, dtype="long", truncating="post", padding="post", value=-1)
    labels_raw_test = pad_sequences(raw_tag_test, maxlen=seq_len, dtype="long", truncating="post", padding="post", value=-1)


    # DATASET & DATALOADER
    input_ids_train = torch.tensor(input_ids_train)
    attention_masks_train = torch.tensor(attention_masks_train)
    labels_train = torch.tensor(labels_train)
    labels_raw_train = torch.tensor(labels_raw_train)

    train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train, labels_raw_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    input_ids_test = torch.tensor(input_ids_test)
    attention_masks_test = torch.tensor(attention_masks_test)
    labels_test = torch.tensor(labels_test)
    labels_raw_test = torch.tensor(labels_raw_test)

    test_data = TensorDataset(input_ids_test, attention_masks_test, labels_test, labels_raw_test)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)


    # Fine-tuning hyper-parameter
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optim = Adam(optimizer_grouped_parameters, lr=learning_rate)
    # optim = Adam(model.parameters(), lr=learning_rate)



    out_file_train = open(savepath + 'train_'+str(batch_size)+'_'+str(learning_rate)+'.txt', 'wt', encoding='utf-8', newline='')
    tsv_writer_train = csv.writer(out_file_train, delimiter='\t')
    tsv_writer_train.writerow(['Epoch', 'Loss', 'Unit Accuracy'])
    out_file_train.flush()

    out_file_test = open(savepath + 'test_'+str(batch_size)+'_'+str(learning_rate)+'.txt', 'wt', encoding='utf-8', newline='')
    tsv_writer_test = csv.writer(out_file_test, delimiter='\t')
    # tsv_writer_test.writerow(['Epoch',
    #                           'Avg. precision', 'Avg. recall', 'Avg. F1',
    #                           'BD precision', 'BD recall', 'BD F1',
    #                           'DS precision', 'DS recall', 'DS F1',
    #                           'ST precision', 'ST recall', 'ST F1',
    #                           'Loss', 'Unit Accuracy', 'Micro Accuracy'])
    tsv_writer_test.writerow(['Epoch', 'Micro F1',
                              'Avg. precision', 'Avg. recall', 'Avg. F1',
                              'BBD precision', 'BBD recall', 'BBD F1',
                              'BDS precision', 'BDS recall', 'BDS F1',
                              'BST precision', 'BST recall', 'BST F1',
                              # 'IBD precision', 'IBD recall', 'IBD F1',
                              # 'IDS precision', 'IDS recall', 'IDS F1',
                              # 'IST precision', 'IST recall', 'IST F1',
                              'Loss', 'Unit Accuracy', 'Micro Accuracy'])
    out_file_test.flush()

    for e in range(epoch):
        str_code = "TRAIN"
        loss_sum = 0.0
        matched_sum = 0.0
        N_target = 0.0
        data_iter = tqdm(enumerate(train_dataloader),
                            desc="Epoch_%s:%d" % (str_code, e+1),
                            total=len(train_dataloader),
                            bar_format="{l_bar}{r_bar}")

        # Training loop
        model.train()
        for i, data in data_iter:
            data = [x.to(device) for x in data]

            pred = model(input_ids=data[0],
                         attention_mask=data[1],
                         token_type_ids=None)

            # CrossEntropy Loss
            loss = loss_fct(pred.view(-1, num_label), data[2].view(-1))
            _, ner_pred = torch.max(pred, 2)

            # matched
            ner_loss = loss.item()
            loss_sum += ner_loss*data[0].shape[0]
            matched, target = get_matched(to_np(ner_pred), to_np(data[2]))
            matched_sum += matched
            N_target += target

            optim.zero_grad()
            loss.backward()
            optim.step()

            # print("\nLoss: %.4f / Accuracy: %.2f" % (ner_loss, matched/target))

            del data

        loss_t = loss_sum/input_ids_train.shape[0]
        accuracy = matched_sum/N_target

        tsv_writer_train.writerow([e+1, loss_t, accuracy])
        out_file_train.flush()

    #########################################################

        str_code = "TEST"
        loss_sum = 0.0
        matched_sum = 0.0
        N_target = 0.0
        t_matched = 0.0
        N_count = 0.0
        res_ner = []
        data_iter = tqdm(enumerate(test_dataloader),
                         desc="Epoch_%s:%d" % (str_code, e + 1),
                         total=len(test_dataloader),
                         bar_format="{l_bar}{r_bar}")

        out_file_res = open(savepath + 'res_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(e+1) + '.txt', 'wt',
                             encoding='utf-8', newline='')
        tsv_writer_res = csv.writer(out_file_res, delimiter='\t')
        tsv_writer_res.writerow(['Text', 'Label', 'Pred'])



        # Test loop
        model.eval()
        for _, data in data_iter:
            data = [x.to(device) for x in data]

            pred = model(input_ids=data[0],
                         attention_mask=data[1],
                         token_type_ids=None)

            # CrossEntropy Loss
            loss = loss_fct(pred.view(-1, num_label), data[2].view(-1))
            _, ner_pred = torch.max(pred, 2)

            # matched
            ner_loss = loss.item()
            loss_sum += ner_loss * data[0].shape[0]
            matched, target = get_matched(to_np(ner_pred), to_np(data[2]))
            matched_sum += matched
            N_target += target


            t_ids = to_np(data[0])
            t_pred = to_np(pred)
            t_label = to_np(data[2])
            t_raw_label = to_np(data[3])
            for i, v in enumerate(t_ids):
                txt = tokenizer.convert_ids_to_tokens(v)
                pred_sample = np.zeros(len(tag_dict))
                tmp_txt = ''
                tmp_label = -1
                tmp_raw_label = -1
                for j, u in enumerate(txt):
                    if u=='[PAD]':
                        tsv_writer_res.writerow([])
                        break
                    if u.startswith('##'):
                        pred_sample += t_pred[i][j]
                        tmp_txt += u[2:]
                    if not u.startswith("##"):
                        tmp_txt = u
                        pred_sample = t_pred[i][j]
                        tmp_label = t_label[i][j]
                        tmp_raw_label = t_raw_label[i][j]
                        N_count += 1.0
                    if j==(seq_len-1):
                        pred_label = pred_sample.argmax()
                        true_label = tmp_label
                        tsv_writer_res.writerow([tmp_txt, true_label, pred_label, tmp_raw_label])
                        res_ner.append([true_label, pred_label])
                        if pred_label==true_label:
                            t_matched += 1.0
                        tsv_writer_res.writerow([])
                        break
                    if not txt[j+1].startswith('##'):
                        pred_label = pred_sample.argmax()
                        true_label = tmp_label
                        tsv_writer_res.writerow([tmp_txt, true_label, pred_label, tmp_raw_label])
                        res_ner.append([true_label, pred_label])
                        if pred_label==true_label:
                            t_matched += 1.0
                    out_file_res.flush()

            # print("\nLoss: %.4f / Unit Accuracy: %.2f" % (ner_loss, matched/target))
            del data
        out_file_res.close()

        res_ner = np.array(res_ner)
        performance = np.zeros((3, 3))
        perf_matched = res_ner[:, 0]==res_ner[:, 1]
        for kk in range(3):
            precision_idx = (res_ner[:, 1] == kk)
            recall_idx = (res_ner[:, 0] == kk)
            if precision_idx.sum()==0:
                performance[kk, 0] = 0
            else:
                performance[kk, 0] = (perf_matched[precision_idx].sum()) / (precision_idx.sum())
            if recall_idx.sum()==0:
                performance[kk, 1] = 0
            else:
                performance[kk, 1] = (perf_matched[recall_idx].sum()) / (recall_idx.sum())
            if (performance[kk, 0]==0) and (performance[kk, 1]==0):
                performance[kk, 2] = 0
            else:
                performance[kk, 2] = 2 * (performance[kk, 0] * performance[kk, 1]) / (performance[kk, 0] + performance[kk, 1])


        loss_t = loss_sum/input_ids_test.shape[0]
        unit_accuracy = matched_sum/N_target
        micro_accuracy = t_matched/N_count
        confusion = metrics.confusion_matrix(res_ner[:, 0], res_ner[:, 1])
        tsv_writer_test.writerow([e + 1,
                                  (confusion[0,0]+confusion[1,1]+confusion[2,2])/confusion[:3,:3].sum(),
                                  performance[:,0].mean(),
                                  performance[:,1].mean(),
                                  performance[:,2].mean(),
                                  performance[0,0], performance[0,1], performance[0,2],
                                  performance[1,0], performance[1,1], performance[1,2],
                                  performance[2,0], performance[2,1], performance[2,2],
                                  # performance[3, 0], performance[3, 1], performance[3, 2],
                                  # performance[4, 0], performance[4, 1], performance[4, 2],
                                  # performance[5, 0], performance[5, 1], performance[5, 2],
                                  loss_t, accuracy, micro_accuracy])
        out_file_test.flush()

    out_file_train.close()
    out_file_test.close()
