from pretrain.examples.extract_features import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from kumc_dataset import InputDataset
from pynvml.smi import nvidia_smi
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

parser = argparse.ArgumentParser(description='KUMC MedSTS')
parser.add_argument('--pretrained', type=str)
args = parser.parse_args()

pretrained_model = args.pretrained

# HYPERPARAMETERS
device = 'cpu' # gpu usage will be updated
epoch = 4
learning_rate = 3e-5
batch_size = 32
seq_len = 128
num_label = 6
is_krbert = True
savepath = "res_sts/"

if not os.path.isdir(savepath):
    os.mkdir(savepath)

filename_read = 'Path for MedSTS dataset'
reader = csv.reader(open(filename_read, 'r', encoding='cp949'))
data=[]
for line in reader:
    data.append(line)

medsts = np.array(data[1:])


if pretrained_model=='kmbert':
    tokenizer = BertTokenizer(vocab_file="./kmbert/kmbert_vocab.txt", do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained('./kmbert/', num_labels=num_label)
    savepath += "kmbert/"
if pretrained_model=='kmbert_vocab':
    tokenizer = BertTokenizer(vocab_file="./kmbert_vocab/kmbert_vocab.txt", do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained('./kmbert_vocab/', num_labels=num_label)
    savepath += "kmbert_vocab/"


loss_fct = CrossEntropyLoss(ignore_index=-1).to(device)


# DATA
token_cls = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
token_sep = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

sentence_1 = [tokenizer.convert_tokens_to_ids(x) for x in [tokenizer.tokenize(x) for x in medsts[:,4]]]
sentence_2 = [tokenizer.convert_tokens_to_ids(x) for x in [tokenizer.tokenize(x) for x in medsts[:,5]]]

sentence, seg = [], []
for i,v in enumerate(sentence_1):
    s1 = [token_cls] + sentence_1[i][:int(seq_len/2 -1)] + [token_sep]
    s2 = sentence_2[i][:int(seq_len/2 -1)] + [token_sep]
    seg_s1 = list(np.zeros(len(s1)))
    seg_s2 = list(np.ones(len(s2)))
    sentence.append(s1+s2)
    seg.append(seg_s1+seg_s2)

input_ids = pad_sequences(sentence, maxlen=seq_len, dtype="long", truncating="post", padding="post")
seg_label = pad_sequences(seg, maxlen=seq_len, dtype="long", truncating="post", padding="post")
attention_masks = np.array([[float(i > 0) for i in ii] for ii in input_ids])
labels_raw = np.array([float(x[3]) for x in medsts])
labels = np.array((labels_raw+0.001).round(), dtype="long")


idx_train = medsts[:,1]=='train'
idx_test = medsts[:,1]=='test'
nidx_test = np.array(medsts[idx_test, 0], dtype="long")

# DATASET & DATALOADER
input_ids_train = torch.tensor(input_ids[idx_train])
seg_label_train = torch.tensor(seg_label[idx_train])
attention_masks_train = torch.tensor(attention_masks[idx_train])
labels_raw_train = torch.tensor(labels_raw[idx_train])
labels_train = torch.tensor(labels[idx_train])

train_data = TensorDataset(input_ids_train, seg_label_train, attention_masks_train, labels_raw_train, labels_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


input_ids_test = torch.tensor(input_ids[idx_test])
seg_label_test = torch.tensor(seg_label[idx_test])
attention_masks_test = torch.tensor(attention_masks[idx_test])
labels_raw_test = torch.tensor(labels_raw[idx_test])
labels_test = torch.tensor(labels[idx_test])
tnidx_test = torch.tensor(nidx_test)

test_data = TensorDataset(input_ids_test, seg_label_test, attention_masks_test, labels_raw_test, labels_test, tnidx_test)
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



out_file_train = open(savepath + 'train_'+str(batch_size)+'_'+str(learning_rate)+'.tsv', 'wt', encoding='utf-8', newline='')
tsv_writer_train = csv.writer(out_file_train, delimiter='\t')
tsv_writer_train.writerow(['Epoch', 'Loss', 'RMSE', 'Pearson', 'Pearson pv', 'Spearman', 'Spearman pv'])
out_file_train.flush()

out_file_test = open(savepath + 'test_'+str(batch_size)+'_'+str(learning_rate)+'.tsv', 'wt', encoding='utf-8', newline='')
tsv_writer_test = csv.writer(out_file_test, delimiter='\t')
tsv_writer_test.writerow(['Epoch', 'Loss', 'RMSE', 'Pearson', 'Pearson pv', 'Spearman', 'Spearman pv'])
out_file_test.flush()

for e in range(epoch):
    str_code = "TRAIN"
    loss_sum = 0.0
    err_sum = 0.0
    data_iter = tqdm(enumerate(train_dataloader),
                        desc="Epoch_%s:%d" % (str_code, e+1),
                        total=len(train_dataloader),
                        bar_format="{l_bar}{r_bar}")

    pred_t = np.array([-1], dtype="float")
    true_t = np.array([-1], dtype="float")
    # Training loop
    model.train()
    for i, data in data_iter:
        pred = model(input_ids=data[0],
                     token_type_ids=data[1],
                     attention_mask=data[2])

        # CrossEntropy Loss
        loss = loss_fct(pred.view(-1, num_label), data[4].view(-1))


        _, sts_pred = torch.max(pred, 1)

        pred_t = np.append(pred_t, to_np(sts_pred).reshape(-1), axis=0)
        true_t = np.append(true_t, to_np(data[3]).reshape(-1), axis=0)

        # MSE
        err = to_np(sum((data[3]-sts_pred)**2))
        mse = err/sts_pred.shape[0]

        sts_loss = loss.item()
        loss_sum += sts_loss
        err_sum += err

        optim.zero_grad()
        loss.backward()
        optim.step()

        print("\nLoss: %.4f / MSE: %.2f" % (sts_loss, mse))

        del data

    pred_t = pred_t[1:]
    true_t = true_t[1:]

    loss_t = loss_sum/data_iter.total
    rmse = (err_sum/true_t.shape[0])**(1/2)
    pearson = scipy.stats.pearsonr(pred_t, true_t)
    spearman = scipy.stats.spearmanr(pred_t, true_t)

    tsv_writer_train.writerow([e+1, loss_t, rmse, pearson[0], pearson[1], spearman[0], spearman[1]])
    out_file_train.flush()

#########################################################

    str_code = "TEST"
    loss_sum = 0.0
    err_sum = 0.0
    data_iter = tqdm(enumerate(test_dataloader),
                     desc="Epoch_%s:%d" % (str_code, e + 1),
                     total=len(test_dataloader),
                     bar_format="{l_bar}{r_bar}")

    pred_t = np.array([-1], dtype="float")
    true_t = np.array([-1], dtype="float")
    nidx_t = np.array([-1], dtype="float")
    # Test loop
    model.eval()
    for i, data in data_iter:
        pred = model(input_ids=data[0],
                     token_type_ids=data[1],
                     attention_mask=data[2])

        # CrossEntropy Loss
        loss = loss_fct(pred.view(-1, num_label), data[4].view(-1))

        _, sts_pred = torch.max(pred, 1)

        pred_t = np.append(pred_t, to_np(sts_pred).reshape(-1), axis=0)
        true_t = np.append(true_t, to_np(data[3]).reshape(-1), axis=0)
        nidx_t = np.append(nidx_t, to_np(data[5]).reshape(-1), axis=0)

        # MSE
        err = to_np(sum((data[3] - sts_pred) ** 2))
        mse = err / sts_pred.shape[0]

        sts_loss = loss.item()
        loss_sum += sts_loss
        err_sum += err

        print("\nLoss: %.4f / MSE: %.2f" % (sts_loss, mse))

        del data

    pred_t = pred_t[1:]
    true_t = true_t[1:]
    nidx_t = nidx_t[1:]

    loss_t = loss_sum / data_iter.total
    rmse = (err_sum / true_t.shape[0]) ** (1 / 2)
    pearson = scipy.stats.pearsonr(pred_t, true_t)
    spearman = scipy.stats.spearmanr(pred_t, true_t)

    out_file_res = open(savepath + 'res_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(e+1) + '.tsv', 'wt',
                         encoding='utf-8', newline='')
    tsv_writer_res = csv.writer(out_file_res, delimiter='\t')
    tsv_writer_res.writerow(['Idx', 'Similarity', 'Pred'])
    for iii in range(nidx_t.shape[0]):
        tsv_writer_res.writerow([nidx_t[iii], true_t[iii], pred_t[iii]])
        out_file_res.flush()
    out_file_res.close()

    tsv_writer_test.writerow([e + 1, loss_t, rmse, pearson[0], pearson[1], spearman[0], spearman[1]])
    out_file_test.flush()


out_file_train.close()
out_file_test.close()
