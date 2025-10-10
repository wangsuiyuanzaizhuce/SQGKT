import csv
import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from sqgkt import sqgkt  # 确保 sqgkt.py 中的 predict 方法移除了最后的 sigmoid
from params import *
from utils import gen_sqgkt_graph, build_adj_list, build_adj_list_uq, gen_sqgkt_graph_uq

# --- 设置 ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 有助于调试CUDA错误
time_now = datetime.now().strftime('%Y_%m_%d#%H_%M_%S')
output_path = os.path.join("output", time_now)
best_model_dir = os.path.join("model_saves", time_now)  # 为本次运行创建一个独立的模型保存目录
os.makedirs(output_path, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

output_file_path = os.path.join(output_path, "log.txt")
output_file = open(output_file_path, "w")

# --- 超参数 ---
params = {
    'max_seq_len': max_seq_len,
    'min_seq_len': min_seq_len,
    'epochs': 20,
    'lr': 0.001,  # Adam 通常使用更小的学习率
    'batch_size': 128,
    'size_q_neighbors': 4,
    'size_q_neighbors_2': 5,
    'size_s_neighbors': 10,
    'size_u_neighbors': 5,
    'agg_hops': 3,
    'emb_dim': 100,
    'hard_recap': False,
    'dropout': (0.2, 0.4),
    'rank_k': 10,
    'k_fold': 5
}

output_file.write(str(params) + '\n\n')
print(params)

# --- 数据和图的准备 ---
print("Loading data and building graphs...")
qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.long, device=DEVICE)
uq_table = torch.tensor(np.load('data/uq_table.npy'), dtype=torch.float32, device=DEVICE)

num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
num_user = torch.tensor(uq_table.shape[0], device=DEVICE)

q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_sqgkt_graph(q_neighbors_list, s_neighbors_list, params['size_q_neighbors'],
                                           params['size_s_neighbors'])
q_neighbors = torch.tensor(q_neighbors, dtype=torch.long, device=DEVICE)
s_neighbors = torch.tensor(s_neighbors, dtype=torch.long, device=DEVICE)

u_neighbors_list, q_neighbors_list_uq = build_adj_list_uq()
u_neighbors, q_neighbors_2 = gen_sqgkt_graph_uq(u_neighbors_list, q_neighbors_list_uq, params['size_u_neighbors'],
                                                params['size_q_neighbors_2'])
u_neighbors = torch.tensor(u_neighbors, dtype=torch.long, device=DEVICE)
q_neighbors_2 = torch.tensor(q_neighbors_2, dtype=torch.long, device=DEVICE)
print("Data loading finished.")

# --- K-Fold 交叉验证 ---
k_fold = KFold(n_splits=params['k_fold'], shuffle=True, random_state=42)
dataset = UserDataset()

# 用于记录每一折的最终测试结果
fold_results = []

# --- K-Fold 交叉验证 ---
k_fold = KFold(n_splits=params['k_fold'], shuffle=True, random_state=42)
dataset = UserDataset()

# 用于记录每一折的最终测试结果
fold_results = []

for fold, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
    print(f"\n{'=' * 20} FOLD {fold + 1}/{params['k_fold']} {'=' * 20}")
    output_file.write(f"\n{'=' * 20} FOLD {fold + 1}/{params['k_fold']} {'=' * 20}\n")

    # --- 为每一折重新初始化模型和优化器 ---
    model = sqgkt(
        num_question, num_skill, q_neighbors, s_neighbors, qs_table, num_user, u_neighbors, q_neighbors_2, uq_table,
        agg_hops=params['agg_hops'],
        emb_dim=params['emb_dim'],
        dropout=params['dropout'],
        hard_recap=params['hard_recap'],
        rank_k=params['rank_k']
    ).to(DEVICE)

    loss_fun = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # --- 准备 DataLoaders ---
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'])

    best_auc_in_fold = 0.0

    for epoch in range(params['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{params['epochs']} ---")

        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0

        for data in train_loader:
            optimizer.zero_grad()

            # --- 【问题修复处】 ---
            data = data.to(DEVICE)
            u = data[:, :, 0]
            x = data[:, :, 1]
            y_target = data[:, :, 2]
            mask = data[:, :, 3].bool()  # 确保 mask 是布尔类型
            # --- 【修复结束】 ---

            y_hat_logits = model(u, x, y_target, mask)

            y_hat_masked = torch.masked_select(y_hat_logits, mask)
            y_target_masked = torch.masked_select(y_target, mask)

            loss = loss_fun(y_hat_masked, y_target_masked.float())
            train_loss_sum += loss.item() * y_target_masked.size(0)

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss_sum / len(train_set)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # --- Testing/Validation Phase ---
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in test_loader:
                # --- 【问题修复处】 ---
                data = data.to(DEVICE)
                u = data[:, :, 0]
                x = data[:, :, 1]
                y_target = data[:, :, 2]
                mask = data[:, :, 3].bool()  # 确保 mask 是布尔类型
                # --- 【修复结束】 ---

                y_hat_logits = model(u, x, y_target, mask)

                y_hat_masked = torch.masked_select(y_hat_logits, mask)
                y_target_masked = torch.masked_select(y_target, mask)

                y_hat_probs = torch.sigmoid(y_hat_masked)

                all_preds.append(y_hat_probs.cpu())
                all_targets.append(y_target_masked.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # 避免在 all_targets 只有一个类别时 roc_auc_score 报错
        if len(torch.unique(all_targets)) > 1:
            test_auc = roc_auc_score(all_targets, all_preds)
        else:
            test_auc = 0.5  # 如果只有一个类别，AUC没有意义，设为0.5

        test_acc = accuracy_score(all_targets, all_preds >= 0.5)

        print(f"Test ACC: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        output_file.write(
            f"Fold {fold + 1}, Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Test ACC: {test_acc:.4f}, Test AUC: {test_auc:.4f}\n")

        if test_auc > best_auc_in_fold:
            best_auc_in_fold = test_auc
            best_model_path_in_fold = os.path.join(best_model_dir, f'fold_{fold + 1}_best_auc.pt')
            torch.save(model.state_dict(), best_model_path_in_fold)
            print(f"Best model for fold {fold + 1} saved with AUC: {best_auc_in_fold:.4f}")

    # 获取当前折叠最后一轮的acc和最好的auc
    last_acc_in_fold = test_acc if 'test_acc' in locals() else 0
    fold_results.append({'acc': last_acc_in_fold, 'auc': best_auc_in_fold})


# --- 总结所有折的结果 ---
print(f"\n{'=' * 20} K-Fold Cross-Validation Summary {'=' * 20}")
output_file.write(f"\n{'=' * 20} K-Fold Summary {'=' * 20}\n")

avg_acc = np.mean([res['acc'] for res in fold_results])
avg_auc = np.mean([res['auc'] for res in fold_results])
std_acc = np.std([res['acc'] for res in fold_results])
std_auc = np.std([res['auc'] for res in fold_results])

print(f"Average Test ACC: {avg_acc:.4f} (+/- {std_acc:.4f})")
print(f"Average Test AUC: {avg_auc:.4f} (+/- {std_auc:.4f})")
output_file.write(f"Average Test ACC: {avg_acc:.4f} (+/- {std_acc:.4f})\n")
output_file.write(f"Average Test AUC: {avg_auc:.4f} (+/- {std_auc:.4f})\n")

output_file.close()
print(f"\nTraining finished. Log saved to {output_file_path}")
print(f"Best models for each fold saved in {best_model_dir}")