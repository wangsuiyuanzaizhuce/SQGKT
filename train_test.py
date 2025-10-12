import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# 确保这些自定义模块可以被正确导入
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from sqgkt import sqgkt
from params import *
from utils import gen_sqgkt_graph, build_adj_list, build_adj_list_uq, gen_sqgkt_graph_uq

# --- 1. 初始化设置 ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
time_now = datetime.now().strftime('%Y_%m_%d#%H_%M_%S')
output_path = os.path.join("output", time_now)
os.makedirs(output_path, exist_ok=True)
output_file_path = os.path.join(output_path, "log.txt")
output_file = open(output_file_path, "w")

# --- 2. 超参数配置 ---
params = {
    'max_seq_len': max_seq_len,
    'min_seq_len': min_seq_len,
    'epochs': 250,
    'lr': 0.01,
    'lr_gamma': 0.85,
    'batch_size': 128,
    'size_q_neighbors': 4,
    'size_q_neighbors_2': 5,
    'size_s_neighbors': 10,
    'size_u_neighbors': 5,
    'num_workers': 0,
    'prefetch_factor': 4,
    'agg_hops': 3,
    'emb_dim': 100,
    'hard_recap': False,
    'dropout': (0.2, 0.4),
    'rank_k': 10,
}
output_file.write(str(params) + '\n')
print(params)

# --- 3. 数据加载与图构建 ---
qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE)
uq_table = torch.tensor(np.load('data/uq_table.npy'), dtype=torch.float32, device=DEVICE)

num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
num_user = torch.tensor(uq_table.shape[0], device=DEVICE)

q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_sqgkt_graph(q_neighbors_list, s_neighbors_list, params['size_q_neighbors'],
                                           params['size_s_neighbors'])
q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

u_neighbors_list, q_neighbors_list = build_adj_list_uq()
u_neighbors, q_neighbors_2 = gen_sqgkt_graph_uq(u_neighbors_list, q_neighbors_list, params['size_u_neighbors'],
                                                params['size_q_neighbors_2'])
u_neighbors = torch.tensor(u_neighbors, dtype=torch.int64, device=DEVICE)
q_neighbors_2 = torch.tensor(q_neighbors_2, dtype=torch.int64, device=DEVICE)

# --- 4. 模型、损失函数、优化器初始化 ---
model = sqgkt(
    num_question, num_skill, q_neighbors, s_neighbors, qs_table, num_user, u_neighbors, q_neighbors_2, uq_table,
    agg_hops=params['agg_hops'],
    emb_dim=params['emb_dim'],
    dropout=params['dropout'],
    hard_recap=params['hard_recap'],
).to(DEVICE)

loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['lr_gamma'])

# --- 5. 断点续训逻辑 ---
resume_path = "model/checkpoint_last.pt"
start_epoch = 0
max_auc = 0.0

if os.path.exists(resume_path):
    print(f"✅ Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    max_auc = checkpoint['max_auc']
    print(f"▶️ Resumed from epoch {start_epoch}, current max_auc is {max_auc:.4f}")
else:
    print("ℹ️ No checkpoint found, starting from scratch.")

dataset = UserDataset()
print('model has been built')

train_data_len = int(len(dataset) * 0.8)
test_data_len = len(dataset) - train_data_len
indices = np.arange(len(dataset))
np.random.shuffle(indices)
train_indices = indices[:train_data_len]
test_indices = indices[train_data_len:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=params['batch_size'],
    shuffle=True,
    num_workers=params["num_workers"],
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=params['batch_size'],
    shuffle=False,
    num_workers=params["num_workers"],
    drop_last=True,
)

# epoch_total_counter 已被移除，我们用 epoch 和 fold 直接显示
for epoch in range(start_epoch, params['epochs']):
    # 这些变量将在后面计算平均值
    train_loss, train_acc_aver, train_auc_aver = 0.0, 0.0, 0.0
    test_loss_aver, test_acc_aver, test_auc_aver = 0.0, 0.0, 0.0

    time0 = time.time()

    # --- Training Phase ---
    model.train()
    train_step, train_loss_total, train_right, train_total, train_auc_total = 0, 0.0, 0.0, 0.0, 0.0
    print('-------------------training------------------')
    for data in train_loader:
        optimizer.zero_grad()
        u, x, y_target, mask = (d.to(DEVICE) for d in
                                [data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3].to(torch.bool)])

        y_hat = model(u, x, y_target, mask)
        y_hat_masked = torch.masked_select(y_hat, mask)
        y_target_masked = torch.masked_select(y_target, mask)

        loss = loss_fun(y_hat_masked, y_target_masked.to(torch.float32))
        loss.backward()
        optimizer.step()

        y_pred_masked = (y_hat_masked >= 0.0).int()

        # --- 实时计算与打印 ---
        train_step += 1
        batch_loss = loss.item()
        num_valid = torch.sum(mask).item()
        batch_acc = torch.sum(y_pred_masked == y_target_masked).item() / num_valid
        try:
            batch_auc = roc_auc_score(y_target_masked.cpu().numpy(), y_hat_masked.detach().cpu().numpy())
        except ValueError:
            batch_auc = 0.5

        print(f'step: {train_step}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

        # --- 累加用于 fold 平均 ---
        train_loss_total += batch_loss
        train_right += torch.sum(y_pred_masked == y_target_masked).item()
        train_total += num_valid
        train_auc_total += batch_auc * num_valid

    # --- Testing Phase ---
    model.eval()
    test_step, test_loss_total, test_right, test_total, test_auc_total = 0, 0.0, 0.0, 0.0, 0.0
    print('-------------------testing------------------')
    with torch.no_grad():
        for data in test_loader:
            u, x, y_target, mask = (d.to(DEVICE) for d in
                                    [data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3].to(torch.bool)])

            y_hat = model(u, x, y_target, mask)
            y_hat_masked = torch.masked_select(y_hat, mask)
            y_target_masked = torch.masked_select(y_target, mask)

            loss = loss_fun(y_hat_masked, y_target_masked.to(torch.float32))

            y_pred_masked = (y_hat_masked >= 0.0).int()

            # --- 实时计算与打印 ---
            test_step += 1
            batch_loss = loss.item()
            num_valid = torch.sum(mask).item()
            batch_acc = torch.sum(y_pred_masked == y_target_masked).item() / num_valid
            try:
                batch_auc = roc_auc_score(y_target_masked.cpu().numpy(), y_hat_masked.detach().cpu().numpy())
            except ValueError:
                batch_auc = 0.5

            print(f'step: {test_step}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}, auc: {batch_auc:.4f}')

            # --- 累加用于 fold 平均 ---
            test_loss_total += batch_loss
            test_right += torch.sum(y_pred_masked == y_target_masked).item()
            test_total += num_valid
            test_auc_total += batch_auc * num_valid

    run_time = time.time() - time0
    print(LOG_B + f'Time: {run_time:.2f}s' + LOG_END)
    # 实时刷新缓冲区，确保日志不丢失
    output_file.flush()

    # 计算平均值
    train_acc_aver = train_right / train_total if train_total > 0 else 0.0
    train_auc_aver = train_auc_total / train_total if train_total > 0 else 0.0
    test_acc_aver = test_right / test_total if test_total > 0 else 0.0
    test_auc_aver = test_auc_total / test_total if test_total > 0 else 0.0

    print('>>>>>>>>>>>>>>>>>>' + LOG_Y + f" Epoch: {epoch + 1} Average Results " + LOG_END + '<<<<<<<<<<<<<<<<<<')
    print(
        LOG_G + f'Training Average: loss: {train_loss_total:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver:.4f}' + LOG_END)
    print(
        LOG_G + f'Testing Average:  loss: {test_loss_total:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver:.4f}' + LOG_END)

    output_file.write(
        f'--- epoch {epoch + 1} average | train_loss: {train_loss_total:.4f}, train_acc: {train_acc_aver:.4f}, train_auc: {train_auc_aver:.4f} | test_loss: {test_loss_total:.4f}, test_acc: {test_acc_aver:.4f}, test_auc: {test_auc_aver:.4f} ---\n')

    # 1. 检查并保存最佳模型
    if test_auc_aver > max_auc:
        max_auc = test_auc_aver
        print(f"🎉 New best model found with average AUC: {max_auc:.4f}. Saving to model/model_best_auc.pt")
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_best_auc.pt')

    # 2. 保存最新的检查点
    print(f"💾 Saving last checkpoint for epoch {epoch} to model/checkpoint_last.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'max_auc': max_auc,
    }, 'model/checkpoint_last.pt')

    # 更新学习率
    scheduler.step()

output_file.close()
print("🎉 Training finished!")