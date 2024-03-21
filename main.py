import argparse
from collections import defaultdict
import pdb

from sklearn.model_selection import StratifiedKFold
from utils import setup_logging, build_optimizer_scheduler
from loader import MoleculeDataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from my_model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, auc
from collections import defaultdict
import pandas as pd


ngram_words_dict = defaultdict(lambda: len(ngram_words_dict) + 1)

def split_ngram_sequence(sequence, ngram=3):
    sequence = '-' + sequence + '='
    ngram_words = [ngram_words_dict[sequence[i: i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(ngram_words)

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 =  dataset[:n], dataset[n:]
    return dataset_1, dataset_2


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--gnn_num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='embedding dimensions (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.1)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help='gnn_type')
    parser.add_argument('--dataset', type=str, default = 'human', 
                        help='[human, celegans, gpcr, BindingDB]')
    parser.add_argument('--seed', type=int, default=1234, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=1234, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--ngram', type=int, default = 3, help='n-gram of split protein sequence')
    parser.add_argument('--words_dict_num', type=int, default=19000, help='the size of the words dict')
    parser.add_argument('--num_tasks', type=int, default=2, help='the num of classification')
    parser.add_argument('--logger_file_path',type=str,default='./logger/log_', help='dir of logger')
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()
    args.logger_file_path = args.logger_file_path + args.dataset + '_' + str(args.gnn_num_layer) + '_' + str(args.runseed) + '.txt'
    logger = setup_logging(args)
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    if args.dataset in ['human','celegans']:

        data = pd.read_csv('dataset/' + args.dataset + '/raw/' + args.dataset + '.csv')
        data['ngram_words'] = data['sequence'].apply(split_ngram_sequence)
        data = shuffle(data, random_state=args.seed).reset_index(drop=True)
        train_data, data_ = split_dataset(data, 0.8)
        valid_data, test_data = split_dataset(data_, 0.5)

    else:
        train_data = pd.read_csv('dataset/' + args.dataset + '/train.csv')
        valid_data = pd.read_csv('dataset/' + args.dataset + '/dev.csv')
        test_data = pd.read_csv('dataset/' + args.dataset +'/test.csv')
        train_data['ngram_words'] = train_data['sequence'].apply(split_ngram_sequence)
        valid_data['ngram_words'] = valid_data['sequence'].apply(split_ngram_sequence)
        test_data['ngram_words'] = test_data['sequence'].apply(split_ngram_sequence)

    args.words_dict_num =  len(ngram_words_dict) + 10
    train_dataset = MoleculeDataset(train_data, ngram=3)
    valid_dataset = MoleculeDataset(valid_data, ngram=3)
    test_dataset = MoleculeDataset(test_data, ngram=3)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, collate_fn=valid_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, collate_fn=test_dataset.collate)

    num_total_steps = args.epochs * len(train_dataloader)
    
    model = GNN_graphpred(args)
    model.to(device)
    trainer = Trainer(args, num_total_steps, model)
    tester = Tester(model)
    file_model = './output/model_BindingDB/' + args.dataset + '_'


    logger.info('start training!')
    logger.info(f'{args}')
    logger.info('model_10')
    best_auc_val = 0.8
    ema = EMA(model, 0.999)
    ema.register()
    early_stop = False
    early_stop_count = 0

    for epoch in range(1, args.epochs+1):
        logger.info(f'Epoch:{epoch}/{args.epochs}')
        loss_train = trainer.train(device, train_dataloader, ema)
        ema.apply_shadow()
        AUC_val, pre_val, recall_val, aupr_val, f1_val = tester.test(device, valid_dataloader)
        AUC_test, pre_test, recall_test, aupr_test, f1_test = tester.test(device, test_dataloader)
        logger.info(f"loss_train: {loss_train}")
        logger.info(f"AUC_val: {AUC_val}, pre_val: {pre_val}, recall_val: {recall_val}, aupr_val: {aupr_val}, f1_val: {f1_val}")
        logger.info(f"AUC_test: {AUC_test}, pre_test: {pre_test}, recall_test: {recall_test}, aupr_test: {aupr_test}, f1_test: {f1_test}")
        if early_stop == True:
            early_stop_count += 1
            print('early_stop_count:', early_stop_count)
        if AUC_val > best_auc_val:
            early_stop = True
            early_stop_count = 0
            print('early_stop_count:', early_stop_count)
            best_auc_val = AUC_val
            tester.save_model(model, file_model, AUC_val)
            logger.info(f"best_auc_val: {best_auc_val}")
        if early_stop_count > 20:
            break
        ema.restore()



class Trainer(object):
    def __init__(self, args, num_total_steps, model):
        self.model = model
        self.optimizer, self.scheduler = build_optimizer_scheduler(args, self.model, num_total_steps)
    def train(self, device, train_loader, ema):
        self.model.train()
        total_loss = 0
        total_num = 0
        for idx, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            input_graph = batch['graph_data'].to(device)
            input_ngram_words = batch['ngram_words'].to(device)
            input_ngram_words_mask = batch['ngram_words_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = self.model(input_graph, input_ngram_words, input_ngram_words_mask, labels, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ema.update()
            mean_loss = loss.to('cpu').data.numpy()
            num = len(batch)
            total_loss += mean_loss*num
            total_num += num
        return total_loss/total_num
    
class Tester(object):
    def __init__(self, model):
        self.model = model
    def test(self, device, test_loader):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc="Iteration")):
                input_graph = batch['graph_data'].to(device)
                input_ngram_words = batch['ngram_words'].to(device)
                input_ngram_words_mask = batch['ngram_words_mask'].to(device)
                labels = batch['labels'].to(device)

                correct_labels, predicted_labels, predicted_scores = self.model(input_graph, input_ngram_words, input_ngram_words_mask, labels, train=False)

                T.extend(correct_labels.tolist())
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)

        pr, re, _ = precision_recall_curve(T, S)
        aupr = auc(re, pr)
        F1 = f1_score(T, Y)
        return AUC, precision, recall, aupr, F1

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename, AUC_val):
        torch.save(model.state_dict(), filename + str(round(AUC_val, 5)) + '.pt')

if __name__ == "__main__":
    main()