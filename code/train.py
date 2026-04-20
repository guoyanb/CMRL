# [file name]: train.py
from model import *
from utils import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def loss_function(res_pre, labels, vec_mean, vec_cov, kl_loss, device, beta=1e-3, gamma=1e-4):
    m = nn.Sigmoid()
    n = torch.squeeze(torch.squeeze(m(res_pre.to(device)), 1)).to(torch.float32)
    labels = torch.squeeze(labels).to(device).to(torch.float32)

    recon_loss = F.binary_cross_entropy(n, labels)

    ib_kl = 0.5 * torch.sum(vec_mean.pow(2) + vec_cov.pow(2) - 2 * vec_cov.log() - 1)

    total_loss = recon_loss + beta * (ib_kl / res_pre.shape[0]) + gamma * kl_loss

    return n, total_loss, recon_loss, ib_kl / res_pre.shape[0], kl_loss

def Train(train_data, test_data, in_size, args, hg, features, device):
    np.random.seed(args.seed)

    val_data_pos = test_data[np.where(test_data[:, -1] == 1)]

    shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    task_test_data = test_data[shuffle_index]

    model = CMRL(device,
        meta_paths=args.metapaths,
        test_data=val_data_pos,
        in_size=in_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        etypes=args.etypes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    mrr = MRR()
    matrix = Matrix()

    features = {key: value.to(device) for key, value in features.items()}

    trainloss = []
    valloss = []
    result_list = []
    hits_max_matrix = np.zeros((1, 3))
    NDCG_max_matrix = np.zeros((1, 3))
    patience_num_matrix = np.zeros((1, 1))
    MRR_max_matrix = np.zeros((1, 1))
    epoch_max_matrix = np.zeros((1, 1))

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        train_data_tensor = torch.tensor(train_data, device=device, dtype=torch.long)

        score_train_predict, enc_mean, enc_std, kl_loss, mixture_weights = model(
            hg, features, train_data_tensor
        )

        train_label = torch.unsqueeze(torch.from_numpy(train_data[:, 3]), 1).to(device)

        n, total_loss, recon_loss, ib_kl, feature_kl = loss_function(
            score_train_predict, train_label, enc_mean, enc_std, kl_loss, device,
            beta=1e-3, gamma=getattr(args, 'bayesian_gamma', 1e-4)
        )

        trainloss.append(total_loss.item())

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_data_tensor = torch.tensor(task_test_data, device=device, dtype=torch.long)

            score_val_predict, enc_mean, enc_std, kl_loss, _ = model(
                hg, features, test_data_tensor
            )

            val_label = torch.unsqueeze(torch.from_numpy(task_test_data[:, 3]), 1).to(device)

            _, val_total_loss, _, _, _ = loss_function(
                score_val_predict, val_label, enc_mean, enc_std, kl_loss, device,
                beta=1e-3, gamma=getattr(args, 'bayesian_gamma', 1e-4)
            )

            valloss.append(val_total_loss.item())

            predict_val = np.squeeze(score_val_predict.detach().cpu().numpy())

            hits5, ndcg5, sample_hit5, sample_ndcg5 = matrix(5, 30, predict_val, len(val_data_pos), shuffle_index)
            hits3, ndcg3, sample_hit3, sample_ndcg3 = matrix(3, 30, predict_val, len(val_data_pos), shuffle_index)
            hits1, ndcg1, sample_hit1, sample_ndcg1 = matrix(1, 30, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(30, predict_val, len(val_data_pos), shuffle_index)

            result = [val_total_loss.item()] + [hits5] + [hits3] + [hits1] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}, '
                      f'Train loss: {total_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'IB_KL: {ib_kl.item():.4f}, '
                      f'Feature_KL: {feature_kl.item():.4f}, '
                      f'Val Loss: {result_list[epoch][0]:.4f}, '
                      f'NDCG@5: {result_list[epoch][4]:.6f}, '
                      f'NDCG@3: {result_list[epoch][5]:.6f}, '
                      f'NDCG@1: {result_list[epoch][6]:.6f}, '
                      f'MRR: {result_list[epoch][7]:.6f}')

            patience_num_matrix = ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix,
                                            epoch_max_matrix, epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num)

            if patience_num_matrix[0][0] >= args.patience:
                break

    max_epoch = int(epoch_max_matrix[0][0])
    print(f'Saving train result: NDCG@5={result_list[max_epoch][4]:.6f}, NDCG@3={result_list[max_epoch][5]:.6f}, NDCG@1={result_list[max_epoch][6]:.6f}, MRR={result_list[max_epoch][7]:.6f}')
    print(f'The optimal epoch: {max_epoch}')

    # 只返回 NDCG@5, NDCG@3, NDCG@1, MRR
    return [result_list[max_epoch][4], result_list[max_epoch][5], result_list[max_epoch][6], result_list[max_epoch][7]]
