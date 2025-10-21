import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Subset
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from sqgkt import sqgkt
from params import *
from utils import gen_sqgkt_graph, build_adj_list, build_adj_list_uq, gen_sqgkt_graph_uq

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", time_now)
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "log.txt")
log_file = open(log_file_path, "w")

# 创建参数解析器
import argparse

parser = argparse.ArgumentParser(description="Train and Test SQGKT Model")
parser.add_argument(
    "--max_seq_len", type=int, default=max_seq_len, help="Maximum sequence length"
)
parser.add_argument(
    "--min_seq_len", type=int, default=min_seq_len, help="Minimum sequence length"
)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument(
    "--k_folds", type=int, default=5, help="Number of folds for K-fold cross-validation"
)
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of data loading workers"
)
parser.add_argument(
    "--agg_hops", type=int, default=3, help="Number of aggregation hops"
)
parser.add_argument("--emb_dim", type=int, default=100, help="Embedding dimension")
parser.add_argument(
    "--dropout", type=float, nargs=2, default=(0.2, 0.4), help="Dropout rates"
)
parser.add_argument("--hard_recap", action="store_true", help="Use hard recap")
parser.add_argument(
    "--rank_k", type=int, default=10, help="Rank k for low-rank approximation"
)
parser.add_argument(
    "--size_q_neighbors", type=int, default=4, help="Size of question neighbors"
)
parser.add_argument(
    "--size_q_neighbors_2",
    type=int,
    default=5,
    help="Size of second question neighbors",
)
parser.add_argument(
    "--size_s_neighbors", type=int, default=10, help="Size of skill neighbors"
)
parser.add_argument(
    "--size_u_neighbors", type=int, default=5, help="Size of user neighbors"
)
parser.add_argument(
    "--remark", type=str, default=None, help="Remark for the training session"
)

# 解析参数
args = parser.parse_args()

# 训练时的超参数
params = {
    "max_seq_len": args.max_seq_len,
    "min_seq_len": args.min_seq_len,
    "epochs": args.epochs,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "size_q_neighbors": args.size_q_neighbors,
    "size_q_neighbors_2": args.size_q_neighbors_2,
    "size_s_neighbors": args.size_s_neighbors,
    "size_u_neighbors": args.size_u_neighbors,
    "num_workers": args.num_workers,
    "agg_hops": args.agg_hops,
    "emb_dim": args.emb_dim,
    "hard_recap": args.hard_recap,
    "dropout": args.dropout,
    "rank_k": args.rank_k,
    "k_folds": args.k_folds,
}

# 打印并写超参数
if args.remark:
    log_file.write(f"Remark: {args.remark}\n")
log_file.write(str(params) + "\n")
print(params)
batch_size = params["batch_size"]

qs_table = torch.tensor(
    sparse.load_npz("data/qs_table.npz").toarray(), dtype=torch.int64, device=DEVICE
)
uq_table = torch.tensor(
    np.load("data/uq_table.npy"), dtype=torch.float32, device=DEVICE
)

num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
num_user = torch.tensor(uq_table.shape[0], device=DEVICE)

q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_sqgkt_graph(
    q_neighbors_list,
    s_neighbors_list,
    params["size_q_neighbors"],
    params["size_s_neighbors"],
)
q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

u_neighbors_list, q_neighbors_list = build_adj_list_uq()
u_neighbors, q_neighbors_2 = gen_sqgkt_graph_uq(
    u_neighbors_list,
    q_neighbors_list,
    params["size_u_neighbors"],
    params["size_q_neighbors_2"],
)
u_neighbors = torch.tensor(u_neighbors, dtype=torch.int64, device=DEVICE)
q_neighbors_2 = torch.tensor(q_neighbors_2, dtype=torch.int64, device=DEVICE)

# 数据集
dataset = UserDataset()

print(f"Dataset loaded. Starting {args.k_folds}-fold cross-validation...")

# K 折交叉验证
kf = KFold(n_splits=args.k_folds, shuffle=True)
all_fold_results = []
indices = np.arange(len(dataset))

for fold_idx, (train_indices, test_indices) in enumerate(kf.split(indices), start=1):
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params["num_workers"],
        drop_last=True,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        drop_last=True,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    train_data_len, test_data_len = len(train_dataset), len(test_dataset)

    # 每折重新初始化模型与优化器
    model = sqgkt(
        num_question,
        num_skill,
        q_neighbors,
        s_neighbors,
        qs_table,
        num_user,
        u_neighbors,
        q_neighbors_2,
        uq_table,
        agg_hops=params["agg_hops"],
        emb_dim=params["emb_dim"],
        dropout=params["dropout"],
        hard_recap=params["hard_recap"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])  # 优化器
    loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE)  # 损失函数

    epoch_total = 1
    best_auc = 0
    best_model_dict = None

    for epoch in range(params["epochs"]):
        print(
            "==================="
            + LOG_Y
            + f"fold: {fold_idx}/{params['k_folds']} | epoch: {epoch_total}/{params['epochs']}"
            + LOG_END
            + "===================="
        )
        log_file.write(
            "===================\n"
            + f"fold: {fold_idx}/{params['k_folds']} | epoch: {epoch_total}/{params['epochs']}\n"
            + "====================\n"
        )

        print("-------------------training------------------")
        log_file.write("-------------------training------------------\n")
        train_start_time = time.time()
        model.train()
        train_batch = train_loss = train_total = train_right = train_auc = 0

        train_all_targets = []
        train_all_preds = []

        for data in train_loader:
            optimizer.zero_grad()
            u, x, y_target, mask = (
                data[:, :, 0].to(DEVICE),
                data[:, :, 1].to(DEVICE),
                data[:, :, 2].to(DEVICE),
                data[:, :, 3].to(torch.bool).to(DEVICE),
            )
            y_hat = model(u, x, y_target, mask)
            y_hat = torch.masked_select(y_hat, mask)
            y_pred = torch.ge(y_hat, torch.tensor(0.5, device=DEVICE)).to(torch.int)
            y_target = torch.masked_select(y_target, mask)
            loss = loss_fun(y_hat, y_target.to(torch.float32))
            train_loss += loss.item()

            acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask).to(float)
            train_right += torch.sum(torch.eq(y_target, y_pred)).to(float)
            train_total += torch.sum(mask).to(float)

            auc = roc_auc_score(y_target.detach().cpu(), y_pred.detach().cpu())
            loss.backward()
            optimizer.step()
            train_batch += 1
            print(
                f"train batch: {train_batch}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}"
            )
            log_file.write(
                f"train batch: {train_batch}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}\n"
            )
            train_all_targets.extend(y_target.detach().cpu().numpy())
            train_all_preds.extend(y_hat.detach().cpu().numpy())
        train_end_time = time.time()

        train_loss = train_loss / max(train_batch, 1)
        train_acc = train_right / max(train_total, 1.0)
        train_auc = roc_auc_score(train_all_targets, train_all_preds)

        # 保存训练状态
        checkpoint_path = os.path.join(fold_dir, "checkpoint.pt")
        log_file.flush()
        checkpoint = {
            "epoch": epoch_total,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(
            f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) [Fold {fold_idx}] Checkpoint saved."
        )

        print("-------------------testing------------------")
        log_file.write("-------------------testing------------------\n")
        test_batch = test_loss = test_total = test_right = test_auc = 0

        test_all_targets = test_all_preds = []

        model.eval()
        test_start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                u, x, y_target, mask = (
                    data[:, :, 0].to(DEVICE),
                    data[:, :, 1].to(DEVICE),
                    data[:, :, 2].to(DEVICE),
                    data[:, :, 3].to(torch.bool).to(DEVICE),
                )
                y_hat = model(u, x, y_target, mask)

                y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
                y_pred = torch.ge(y_hat, torch.tensor(0.5, device=DEVICE)).to(torch.int)
                y_target = torch.masked_select(y_target, mask.to(torch.bool))
                loss = loss_fun(y_hat, y_target.to(torch.float32))
                test_loss += loss.item()

                acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask).to(float)
                test_right += torch.sum(torch.eq(y_target, y_pred)).to(float)
                test_total += torch.sum(mask).to(float)

                test_batch += 1
                print(
                    f"test batch: {test_batch}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}"
                )
                log_file.write(
                    f"test batch: {test_batch}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}\n"
                )
                test_all_targets.extend(y_target.detach().cpu().numpy())
                test_all_preds.extend(y_hat.detach().cpu().numpy())
            test_end_time = time.time()

        test_loss = test_loss / max(test_batch, 1)
        test_acc = test_right / max(test_total, 1.0)
        test_auc = roc_auc_score(test_all_targets, test_all_preds)

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_dict = {
                "epoch": epoch_total,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
            # 保存最佳模型权重
            torch.save(best_model_dict, os.path.join(fold_dir, "best_model.pt"))

        train_time = train_end_time - train_start_time
        test_time = test_end_time - test_start_time

        print(
            LOG_B
            + f"[Fold {fold_idx}] train time: {train_time:.2f}s, average batch time: {(train_time / max(train_batch,1)):.2f}s"
            + LOG_END
        )
        print(
            LOG_B
            + f"[Fold {fold_idx}] test time: {test_time:.2f}s, average batch time: {(test_time / max(test_batch,1)):.2f}s"
            + LOG_END
        )
        print(
            LOG_G
            + f"[Fold {fold_idx}] training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc:.4f}"
            + LOG_END
        )
        print(
            LOG_G
            + f"[Fold {fold_idx}] testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc:.4f}"
            + LOG_END
        )

        log_file.write(f"fold {fold_idx} | epoch {epoch_total}\n")
        log_file.write(
            f"training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc:.4f}\n"
        )
        log_file.write(
            f"testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc:.4f}\n"
        )
        log_file.write(
            f"train time: {train_time:.2f}s, average batch time: {(train_time / max(train_batch,1)):.2f}s\n"
        )
        log_file.write(
            f"test time: {test_time:.2f}s, average batch time: {(test_time / max(test_batch,1)):.2f}s\n"
        )
        log_file.flush()
        epoch_total += 1

    # 每一折结束后记录最佳结果
    if best_model_dict is not None:
        all_fold_results.append(
            {
                "fold": fold_idx,
                "train_acc": float(best_model_dict["train_acc"]),
                "test_acc": float(best_model_dict["test_acc"]),
                "train_auc": float(best_model_dict["train_auc"]),
                "test_auc": float(best_model_dict["test_auc"]),
                "train_loss": float(best_model_dict["train_loss"]),
                "test_loss": float(best_model_dict["test_loss"]),
            }
        )
        print(
            f"[Fold {fold_idx}] Best train_acc: {float(best_model_dict['train_acc']):.4f}, test_acc: {float(best_model_dict['test_acc']):.4f}"
        )
        log_file.write(
            f"[Fold {fold_idx}] Best -> train_acc: {float(best_model_dict['train_acc']):.4f}, test_acc: {float(best_model_dict['test_acc']):.4f}, train_auc: {float(best_model_dict['train_auc']):.4f}, test_auc: {float(best_model_dict['test_auc']):.4f}\n"
        )
        log_file.flush()

# 汇总所有折的结果
if all_fold_results:
    avg_train_acc = sum(r["train_acc"] for r in all_fold_results) / len(
        all_fold_results
    )
    avg_test_acc = sum(r["test_acc"] for r in all_fold_results) / len(all_fold_results)
    avg_train_auc = sum(r["train_auc"] for r in all_fold_results) / len(
        all_fold_results
    )
    avg_test_auc = sum(r["test_auc"] for r in all_fold_results) / len(all_fold_results)

    # 最高的准确率和AUC
    best_train_acc = max(r["train_acc"] for r in all_fold_results)
    best_test_acc = max(r["test_acc"] for r in all_fold_results)
    best_train_auc = max(r["train_auc"] for r in all_fold_results)
    best_test_auc = max(r["test_auc"] for r in all_fold_results)

    summary = (
        f"\n======{args.k_folds}-fold Validation Result======\n"
        f"Average train acc: {avg_train_acc:.4f}, Average test acc: {avg_test_acc:.4f}\n"
        f"Average train auc: {avg_train_auc:.4f}, Average test auc: {avg_test_auc:.4f}\n"
        # 输出最高准确率和AUC
        f"Best train acc: {best_train_acc:.4f}, Best test acc: {best_test_acc:.4f}\n"
        f"Best train auc: {best_train_auc:.4f}, Best test auc: {best_test_auc:.4f}\n"
    )
    print(summary)
    log_file.write(summary)
    log_file.flush()

log_file.close()
