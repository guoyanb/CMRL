# [file name]: main.py
import warnings
from data_process import data_load
from train import Train
from utils import *
import random
import os
import gc
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.sparse as sp
warnings.filterwarnings("ignore")

def main_indep():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features, in_size = data_load()

    NDCG_5, NDCG_3, NDCG_1, MRR = [], [], [], []

    train_data_pos = np.array(pd.read_csv('./Data/indepent_data/train_data_pos.csv', header=None))
    train_data_neg = np.array(pd.read_csv('./Data/indepent_data/train_data_neg.csv', header=None))
    val_data_pos = np.array(pd.read_csv('./Data/indepent_data/test_data_pos.csv', header=None))
    val_data_neg = np.array(pd.read_csv('./Data/indepent_data/test_data_neg.csv', header=None))

    hg = construct_hg(train_data_pos)

    train_data = np.vstack((train_data_pos, train_data_neg))
    np.random.shuffle(train_data)

    val_data = np.vstack((val_data_pos, val_data_neg))

    result = Train(train_data, val_data, in_size, args, hg, features, device)

    NDCG_5.append(result[0])
    NDCG_3.append(result[1])
    NDCG_1.append(result[2])
    MRR.append(result[3])

    print('---------- Independent test finished -----------')
    print(f'Independent test result: '
          f'NDCG@5: {np.mean(NDCG_5):.6f}, '
          f'NDCG@3: {np.mean(NDCG_3):.6f}, '
          f'NDCG@1: {np.mean(NDCG_1):.6f}, '
          f'MRR: {np.mean(MRR):.6f}')

    return np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)

def main_CV():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features, in_size = data_load()

    NDCG_5, NDCG_3, NDCG_1, MRR = [], [], [], []

    for fold_num in range(1, 6):
        train_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_pos.csv', header=None))
        train_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_neg.csv', header=None))
        val_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_pos.csv', header=None))
        val_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_neg.csv', header=None))

        hg = construct_hg(train_data_pos)

        train_data = np.vstack((train_data_pos, train_data_neg))
        np.random.shuffle(train_data)

        val_data = np.vstack((val_data_pos, val_data_neg))

        result = Train(train_data, val_data, in_size, args, hg, features, device)

        NDCG_5.append(result[0])
        NDCG_3.append(result[1])
        NDCG_1.append(result[2])
        MRR.append(result[3])

    print('---------- 5-fold CV finished -----------')
    print(f'5-fold CV result: '
          f'NDCG@5: {np.mean(NDCG_5):.6f}, '
          f'NDCG@3: {np.mean(NDCG_3):.6f}, '
          f'NDCG@1: {np.mean(NDCG_1):.6f}, '
          f'MRR: {np.mean(MRR):.6f}')

    return np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)

def seed_it(seed):
    gc.collect()
    torch.cuda.empty_cache()
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

if __name__ == '__main__':
    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    args = parameters_set()

    print('Starting the 5-fold CV experiment', args.lr)
    seed_it(args.seed)

    CV_NDCG5, CV_NDCG3, CV_NDCG1, CV_MRR = main_CV()

    with open('./Result/HCMGNN_CV_print.txt', 'a') as f:
        f.write(f'{CV_NDCG5}\t{CV_NDCG3}\t{CV_NDCG1}\t{CV_MRR}\n')

    print('Starting the independent test experiment')

    indep_NDCG5, indep_NDCG3, indep_NDCG1, indep_MRR = main_indep()

    with open('./Result/HCMGNN_indep_print.txt', 'a') as f:
        f.write(f'{indep_NDCG5}\t{indep_NDCG3}\t{indep_NDCG1}\t{indep_MRR}\n')
