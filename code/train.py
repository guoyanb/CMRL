# [file name]: train.py
from model import *
from utils import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def loss_function(res_pre, labels, vec_mean, vec_cov, kl_loss, device, beta=1e-3, gamma=1e-4):
    """Calculate total loss including reconstruction, IB KL, and Bayesian KL"""
    m = nn.Sigmoid()
    n = torch.squeeze(torch.squeeze(m(res_pre.to(device)), 1)).to(torch.float32)
    labels = torch.squeeze(labels).to(device).to(torch.float32)
    
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(n, labels)
    
    # Information Bottleneck KL divergence
    ib_kl = 0.5 * torch.sum(vec_mean.pow(2) + vec_cov.pow(2) - 2 * vec_cov.log() - 1)
    
    # Total loss = reconstruction + IB_KL + Bayesian feature KL
    total_loss = recon_loss + beta * (ib_kl / res_pre.shape[0]) + gamma * kl_loss
    
    return n, total_loss, recon_loss, ib_kl / res_pre.shape[0], kl_loss

def Train(train_data, test_data, in_size, args, hg, features, device):
    """Train the CMRL model"""
    np.random.seed(args.seed)
    
    # Extract positive validation data
    val_data_pos = test_data[np.where(test_data[:, -1] == 1)]
    
    # Shuffle test data for evaluation
    shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    task_test_data = test_data[shuffle_index]
    
    # Initialize model
    model = CMRL(device,
        meta_paths=args.metapaths,
        test_data=val_data_pos,
        in_size=in_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        etypes=args.etypes)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    
    # Evaluation metrics
    mrr = MRR()
    matrix = Matrix()
    
    # Ensure features are on correct device
    features = {key: value.to(device) for key, value in features.items()}
    
    # Training tracking variables
    trainloss = []
    valloss = []
    result_list = []
    hits_max_matrix = np.zeros((1, 3))
    NDCG_max_matrix = np.zeros((1, 3))
    patience_num_matrix = np.zeros((1, 1))
    MRR_max_matrix = np.zeros((1, 1))
    epoch_max_matrix = np.zeros((1, 1))
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Convert training data to tensor
        train_data_tensor = torch.tensor(train_data, device=device, dtype=torch.long)
        
        # Forward pass
        score_train_predict, enc_mean, enc_std, kl_loss, mixture_weights = model(
            hg, features, train_data_tensor
        )
        
        # Get training labels
        train_label = torch.unsqueeze(torch.from_numpy(train_data[:, 3]), 1).to(device)
        
        # Calculate loss
        n, total_loss, recon_loss, ib_kl, feature_kl = loss_function(
            score_train_predict, train_label, enc_mean, enc_std, kl_loss, device, 
            beta=1e-3, gamma=getattr(args, 'bayesian_gamma', 1e-4)
        )
        
        trainloss.append(total_loss.item())
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Convert test data to tensor
            test_data_tensor = torch.tensor(task_test_data, device=device, dtype=torch.long)
            
            # Forward pass on test data
            score_val_predict, enc_mean, enc_std, kl_loss, _ = model(
                hg, features, test_data_tensor
            )
            
            # Get validation labels
            val_label = torch.unsqueeze(torch.from_numpy(task_test_data[:, 3]), 1).to(device)
            
            # Calculate validation loss
            _, val_total_loss, _, _, _ = loss_function(
                score_val_predict, val_label, enc_mean, enc_std, kl_loss, device,
                beta=1e-3, gamma=getattr(args, 'bayesian_gamma', 1e-4)
            )
            
            valloss.append(val_total_loss.item())
            
            # Convert predictions to numpy
            predict_val = np.squeeze(score_val_predict.detach().cpu().numpy())
            
            # Calculate evaluation metrics
            hits5, ndcg5, sample_hit5, sample_ndcg5 = matrix(5, 30, predict_val, len(val_data_pos), shuffle_index)
            hits3, ndcg3, sample_hit3, sample_ndcg3 = matrix(3, 30, predict_val, len(val_data_pos), shuffle_index)
            hits1, ndcg1, sample_hit1, sample_ndcg1 = matrix(1, 30, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(30, predict_val, len(val_data_pos), shuffle_index)
            
            # Store results
            result = [val_total_loss.item()] + [hits5] + [hits3] + [hits1] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}, '
                      f'Train loss: {total_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'IB_KL: {ib_kl.item():.4f}, '
                      f'Feature_KL: {feature_kl.item():.4f}, '
                      f'Val Loss: {result_list[epoch][0]:.4f}, '
                      f'Hits@5: {result_list[epoch][1]:.6f}, '
                      f'Hits@3: {result_list[epoch][2]:.6f}, '
                      f'Hits@1: {result_list[epoch][3]:.6f}, '
                      f'NDCG@5: {result_list[epoch][4]:.6f}, '
                      f'NDCG@3: {result_list[epoch][5]:.6f}, '
                      f'NDCG@1: {result_list[epoch][6]:.6f}, '
                      f'MRR: {result_list[epoch][7]:.6f}')
            
            # Early stopping check
            patience_num_matrix = ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix,
                                            epoch_max_matrix, epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num)
            
            if patience_num_matrix[0][0] >= args.patience:
                break
    
    # Get best epoch results
    max_epoch = int(epoch_max_matrix[0][0])
    print(f'Saving train result: {result_list[max_epoch][1:]}')
    print(f'The optimal epoch: {max_epoch}')
    
    return result_list[max_epoch][1:]