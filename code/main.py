# [file name]: main.py
import warnings
from data_process import data_load
from train import Train
from utils import *
import random
import os
from copy import deepcopy
import scipy.sparse as sp
warnings.filterwarnings("ignore")

def main_indep():
    """Run independent test experiment"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features, in_size = data_load()
    
    # Initialize metrics lists
    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(7)]
    
    # Load independent test data
    train_data_pos = np.array(pd.read_csv('./Data/indepent_data/train_data_pos.csv', header=None))
    train_data_neg = np.array(pd.read_csv('./Data/indepent_data/train_data_neg.csv', header=None))
    val_data_pos = np.array(pd.read_csv('./Data/indepent_data/test_data_pos.csv', header=None))
    val_data_neg = np.array(pd.read_csv('./Data/indepent_data/test_data_neg.csv', header=None))
    
    # Construct heterogeneous graph from training data
    hg = construct_hg(train_data_pos)
    
    # Combine and shuffle training data
    train_data = np.vstack((train_data_pos, train_data_neg))
    np.random.shuffle(train_data)
    
    # Combine validation data
    val_data = np.vstack((val_data_pos, val_data_neg))
    
    # Train model and get results
    result = Train(train_data, val_data, in_size, args, hg, features, device)
    
    # Store results
    Hits_5.append(result[0])
    Hits_3.append(result[1])
    Hits_1.append(result[2])
    NDCG_5.append(result[3])
    NDCG_3.append(result[4])
    NDCG_1.append(result[5])
    MRR.append(result[6])
    
    print('---------- Independent test finished -----------')
    print('Independent test result: '
          f'Hits@5: {np.mean(Hits_5):.6f}, '
          f'Hits@3: {np.mean(Hits_3):.6f}, '
          f'Hits@1: {np.mean(Hits_1):.6f}, '
          f'NDCG@5: {np.mean(NDCG_5):.6f}, '
          f'NDCG@3: {np.mean(NDCG_3):.6f}, '
          f'NDCG@1: {np.mean(NDCG_1):.6f}, '
          f'MRR: {np.mean(MRR):.6f}')
    
    return np.mean(Hits_5), np.mean(Hits_3), np.mean(Hits_1), np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)

def main_CV():
    """Run 5-fold cross-validation experiment"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features, in_size = data_load()
    
    # Initialize metrics lists
    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(7)]
    
    # Run 5-fold cross-validation
    for fold_num in range(1, 6):
        # Load fold data
        train_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_pos.csv', header=None))
        train_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_neg.csv', header=None))
        val_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_pos.csv', header=None))
        val_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_neg.csv', header=None))
        
        # Construct heterogeneous graph from training data
        hg = construct_hg(train_data_pos)
        
        # Combine and shuffle training data
        train_data = np.vstack((train_data_pos, train_data_neg))
        np.random.shuffle(train_data)
        
        # Combine validation data
        val_data = np.vstack((val_data_pos, val_data_neg))
        
        # Train model and get results
        result = Train(train_data, val_data, in_size, args, hg, features, device)
        
        # Store results
        Hits_5.append(result[0])
        Hits_3.append(result[1])
        Hits_1.append(result[2])
        NDCG_5.append(result[3])
        NDCG_3.append(result[4])
        NDCG_1.append(result[5])
        MRR.append(result[6])
    
    print('---------- 5-fold CV finished -----------')
    print('5-fold CV result: '
          f'Hits@5: {np.mean(Hits_5):.6f}, '
          f'Hits@3: {np.mean(Hits_3):.6f}, '
          f'Hits@1: {np.mean(Hits_1):.6f}, '
          f'NDCG@5: {np.mean(NDCG_5):.6f}, '
          f'NDCG@3: {np.mean(NDCG_3):.6f}, '
          f'NDCG@1: {np.mean(NDCG_1):.6f}, '
          f'MRR: {np.mean(MRR):.6f}')
    
    return np.mean(Hits_5), np.mean(Hits_3), np.mean(Hits_1), np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)

def seed_it(seed):
    """Set random seeds for reproducibility"""
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
    # Create result directory if it doesn't exist
    if not os.path.exists('./Result'):
        os.makedirs('./Result')
    
    # Set parameters
    args = parameters_set()
    
    print('Starting the 5-fold CV experiment', args.lr)
    seed_it(args.seed)
    
    # Run cross-validation experiment
    CV_Hits5, CV_Hits3, CV_Hits1, CV_NDCG_5, CV_NDCG_3, CV_NDCG_1, CV_MRR_num = main_CV()
    
    # Save cross-validation results
    with open('./Result/HCMGNN_CV_print.txt', 'a') as f:
        f.write(f'{CV_Hits5}\t{CV_Hits3}\t{CV_Hits1}\t{CV_NDCG_5}\t{CV_NDCG_3}\t{CV_NDCG_1}\t{CV_MRR_num}\n')
    
    print('Starting the independent test experiment')
    
    # Run independent test experiment
    indep_Hits5, indep_Hits3, indep_Hits1, indep_NDCG_5, indep_NDCG_3, indep_NDCG_1, indep_MRR_num = main_indep()
    
    # Save independent test results
    with open('./Result/HCMGNN_indep_print.txt', 'a') as f:
        f.write(f'{indep_Hits5}\t{indep_Hits3}\t{indep_Hits1}\t{indep_NDCG_5}\t{indep_NDCG_3}\t{indep_NDCG_1}\t{indep_MRR_num}\n')