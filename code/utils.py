# [file name]: utils.py
import gc
import torch
import torch.nn as nn
import numpy as np
import random
import dgl
import pandas as pd
import argparse
import os
from copy import deepcopy
import scipy.sparse as sp

class Matrix(nn.Module):
    """Calculate Hits@n and NDCG@n metrics"""
    def __init__(self):
        super(Matrix, self).__init__()

    def hits(self, pos_index, scores_index):
        """Calculate Hits@n"""
        if pos_index[0] in scores_index:
            Hits = 1
        else:
            Hits = 0
        return Hits

    def ndcg(self, pos_index, scores_index, n):
        """Calculate NDCG@n"""
        dcg_sum = 0
        idcg_sum = 0
        
        # Calculate DCG
        for j in range(len(scores_index)):
            if scores_index[j] == pos_index[0]:
                dcg_sum += self.dcg(1, j + 1)
            else:
                dcg_sum += self.dcg(0, j + 1)
        
        # Calculate IDCG (ideal ranking)
        for m in range(n):
            if m == 0:
                idcg_sum += self.dcg(1, m + 1)
            else:
                idcg_sum += self.dcg(0, m + 1)
        
        return dcg_sum, idcg_sum

    def dcg(self, rel, index):
        """Calculate DCG component"""
        dcg = (2 ** rel - 1) / np.log2(index + 1)
        return dcg

    def forward(self, n, num, predict_val, num_pos, index):
        """Calculate Hits@n and NDCG@n for all samples"""
        sample_hit, sample_ndcg = [], []
        Hits_sum = 0
        ndcg_sum = 0
        
        # Restore original order from shuffle index
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]
        
        # Calculate metrics for each positive sample
        for i in range(num_pos):
            # Get negative scores for this positive sample
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            
            # Shuffle scores
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            
            # Sort by score
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple][:n]
            
            # Calculate Hits
            Hits = self.hits(pos_index, scores_index)
            
            # Calculate NDCG
            dcg_sum, idcg_sum = self.ndcg(pos_index, scores_index, n)
            ndcg_sum += dcg_sum / idcg_sum
            Hits_sum += Hits
            
            # Store per-sample metrics
            sample_hit.append(Hits)
            sample_ndcg.append(dcg_sum / idcg_sum)
        
        # Calculate average metrics
        Hits = Hits_sum / num_pos
        ndcg = ndcg_sum / num_pos
        
        return Hits, ndcg, sample_hit, sample_ndcg

class MRR(nn.Module):
    """Calculate Mean Reciprocal Rank"""
    def __init__(self):
        super(MRR, self).__init__()

    def forward(self, num, predict_val, num_pos, index):
        """Calculate MRR for all samples"""
        sample_mrr = []
        rank_sum = 0
        
        # Restore original order from shuffle index
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]
        
        # Calculate MRR for each positive sample
        for i in range(num_pos):
            # Get negative scores for this positive sample
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            
            # Shuffle scores
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            
            # Sort by score
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple]
            
            # Calculate reciprocal rank
            sample_mrr.append(1 / (scores_index.index(pos_index[0]) + 1))
            rank_sum += 1 / (scores_index.index(pos_index[0]) + 1)
        
        # Calculate average MRR
        mrr = rank_sum / num_pos
        
        return mrr, sample_mrr

def construct_hg(pos_data):
    """Construct heterogeneous graph from positive data"""
    g_m_edges, m_d_edges, g_d_edges = [list() for x in range(3)]
    
    # Extract edges from positive triplets
    for i in range(len(pos_data)):
        one_g_m_edge = []
        one_g_m_edge.extend(pos_data[i][0:2].tolist())
        
        one_m_d_edge = []
        one_m_d_edge.extend(pos_data[i][1:3].tolist())
        
        one_g_d_edge = []
        one_g_d_edge.extend([pos_data[i][0], pos_data[i][2]])
        
        # Add edges if not already present
        if one_g_m_edge not in g_m_edges:
            g_m_edges.append(one_g_m_edge)
        if one_m_d_edge not in m_d_edges:
            m_d_edges.append(one_m_d_edge)
        if one_g_d_edge not in g_d_edges:
            g_d_edges.append(one_g_d_edge)
    
    # Sort edges
    g_m_edges = np.array(sorted(g_m_edges, key=(lambda x: x[0])))
    m_d_edges = np.array(sorted(m_d_edges, key=(lambda x: x[0])))
    g_d_edges = np.array(sorted(g_d_edges, key=(lambda x: x[0])))
    
    # Create heterogeneous graph
    hg = dgl.heterograph({
        ('g', 'g_m', 'm'): (torch.LongTensor(g_m_edges[:, 0]), torch.LongTensor(g_m_edges[:, 1])),
        ('m', 'm_d', 'd'): (torch.LongTensor(m_d_edges[:, 0]), torch.LongTensor(m_d_edges[:, 1])),
        ('g', 'g_d', 'd'): (torch.LongTensor(g_d_edges[:, 0]), torch.LongTensor(g_d_edges[:, 1])),
        ('m', 'm_g', 'g'): (torch.LongTensor(g_m_edges[:, 1]), torch.LongTensor(g_m_edges[:, 0])),
        ('d', 'd_m', 'm'): (torch.LongTensor(m_d_edges[:, 1]), torch.LongTensor(m_d_edges[:, 0])),
        ('d', 'd_g', 'g'): (torch.LongTensor(g_d_edges[:, 1]), torch.LongTensor(g_d_edges[:, 0]))
    })
    
    return hg

class Prevent_leakage(nn.Module):
    """Prevent information leakage by excluding test data"""
    def __init__(self, test_data):
        super(Prevent_leakage, self).__init__()
        self.test_data = test_data

    def forward(self, metapath_instances):
        """Remove test data instances from meta-path instances"""
        test_pos_data = pd.DataFrame(self.test_data[:, :3], columns=['g', 'm', 'd'])
        
        # Append test data twice and drop duplicates
        metapath_instances_all = metapath_instances.append(test_pos_data)
        metapath_instances_all = metapath_instances_all.append(test_pos_data)
        exclude_metapath_instances = metapath_instances_all.drop_duplicates(subset=['g', 'm', 'd'], keep=False)
        exclude_metapath_instances = exclude_metapath_instances.reset_index(drop=True)
        
        return exclude_metapath_instances

class Separate_subgraph(nn.Module):
    """Separate subgraph for each meta-path"""
    def __init__(self):
        super(Separate_subgraph, self).__init__()

    def get_edges(self, edges1, edges2):
        """Get connecting edges between two edge sets"""
        new_edges = [[list() for j in range(2)] for i in range(2)]
        
        # Find connecting edges
        for i in range(len(edges1[0])):
            if edges1[1][i] in edges2[0]:
                new_edges[0][0].append(edges1[0][i])
                new_edges[0][1].append(edges1[1][i])
                
                index = [m for m, x in enumerate(edges2[0]) if x == edges1[1][i]]
                if edges1[1][i] not in new_edges[1][0]:
                    for j in range(len(index)):
                        new_edges[1][0].append(edges1[1][i])
                        new_edges[1][1].append(edges2[1][index[j]])
        
        return new_edges

    def forward(self, hg, metapath):
        """Extract subgraph for given meta-path"""
        new_triplets_edge = []
        
        # Get edge types for meta-path
        metapath_list = [f"{metapath[i]}_{metapath[i + 1]}" for i in range(len(metapath) - 1)]
        edges = [hg.edges(etype=metapath_list[i]) for i in range(len(metapath_list))]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        
        # Build connecting edges for meta-path
        if len(metapath_list) == 2:
            new_edges = self.get_edges(edges[0], edges[1])
        elif len(metapath_list) == 3:
            new_edges = self.get_edges(edges[0], edges[1])
            new_edges1 = self.get_edges(new_edges[1], edges[2])
            new_edges.append(new_edges1[1])
        
        # Map edge types to canonical edge types
        graph_data = {}
        for i in range(len(metapath_list)):
            for canonical_etype in hg.canonical_etypes:
                if metapath_list[i] in canonical_etype:
                    new_triplets_edge.append(canonical_etype)
                    break
        
        # Build subgraph
        for i in range(len(metapath_list)):
            graph_data[new_triplets_edge[i]] = (new_edges[i][0], new_edges[i][1])
        
        subgraph = dgl.heterograph(graph_data)
        return subgraph

def ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix, epoch_max_matrix, e, hits_1,
              hits_3, hits_5, ndcg1, ndcg3, ndcg5, MRR):
    """Early stopping mechanism based on Hits@1"""
    if hits_1 >= hits_max_matrix[0][0]:
        # Update best metrics
        hits_max_matrix[0][0] = hits_1
        hits_max_matrix[0][1] = hits_3
        hits_max_matrix[0][2] = hits_5
        
        NDCG_max_matrix[0][0] = ndcg1
        NDCG_max_matrix[0][1] = ndcg3
        NDCG_max_matrix[0][2] = ndcg5
        
        MRR_max_matrix[0][0] = MRR
        epoch_max_matrix[0][0] = e
        patience_num_matrix[0][0] = 0
    else:
        # Increase patience counter
        patience_num_matrix[0][0] += 1
    
    return patience_num_matrix

def parameters_set():
    """Set model hyperparameters"""
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--num_epochs', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bayesian_gamma', type=float, default=1e-4, help='Bayesian KL weight')
    
    # Model architecture parameters
    parser.add_argument('--hidden_size', type=int, default=384, help='model hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    
    # Graph structure parameters
    parser.add_argument('--etypes', nargs='+', default=[[0, 1], [2, 3], [4, 0], [5, 2], [3, 5], [1, 4]],
                        help='edge types in the six meta-paths')
    parser.add_argument('--metapaths', nargs='+',
                        default=[['g', 'm', 'd'], ['m', 'g', 'd'], ['d', 'g', 'm'], 
                                 ['d', 'm', 'g'], ['g', 'd', 'm'], ['m', 'd', 'g']],
                        help='meta-path types')
    
    args = parser.parse_args()
    return args