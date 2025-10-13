import pandas as pd
import numpy as np
import os
from scipy import sparse
from scipy.stats import poisson

# --- 全局参数 ---
min_seq_len = 20
max_seq_len = 200
k = 0.3
d = 0.7
b = 10
weights = np.array([0.4, 0.4, 0.2])


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    print("Starting data processing...")
    ensure_dir('data')

    # --- 1. 加载和初步清洗数据 ---
    try:
        data = pd.read_csv(filepath_or_buffer='data/skill_builder_data_corrected_collapsed.csv',
                           encoding="ISO-8859-1", low_memory=False)
    except FileNotFoundError:
        print("Error: 'skill_builder_data_corrected_collapsed.csv' not found in 'data/' directory.")
        exit()

    # 数据清洗
    data = data.sort_values(by='user_id', ascending=True)
    data.dropna(subset=['skill_id', 'problem_id', 'user_id'], inplace=True)
    data = data[data['skill_id'] != ' ']
    data = data[data['original'] == 1]  # 只保留首次尝试

    # 筛选有效长度的用户序列
    is_valid_user = data.groupby('user_id').size() >= min_seq_len
    data = data[data['user_id'].isin(is_valid_user[is_valid_user].index)]

    data = data.loc[:, ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id',
                        'attempt_count', 'hint_count']]
    print(f"Data cleaned. Rows: {len(data)}")

    # --- 2. 计算特征因子 ---
    print("Calculating feature factors...")

    # 尝试次数因子
    question_attempt_stats = data.groupby('problem_id')['attempt_count'].mean().rename('mean_attempt')
    data = data.join(question_attempt_stats, on='problem_id')
    with np.errstate(divide='ignore', invalid='ignore'):
        attempt_factor = 1 - poisson(data['mean_attempt']).cdf(data['attempt_count'] - 1)
    data['attempt_factor_g'] = k + (1 - k) / (1 + np.exp(-d * (attempt_factor - b)))

    # 提示因子
    question_hint_stats = data.groupby('problem_id')['hint_count'].mean().rename('mean_hint')
    data = data.join(question_hint_stats, on='problem_id')
    with np.errstate(divide='ignore', invalid='ignore'):
        hint_factor = 1 - poisson(data['mean_hint']).cdf(data['hint_count'] - 1)
    data['hint_factor_g'] = k + (1 - k) / (1 + np.exp(-d * (hint_factor - b)))

    # 学生能力因子
    data['ability_factor'] = data.groupby('user_id')['correct'].transform('mean')
    data.fillna(0, inplace=True)

    print("Feature calculation complete.")

    # --- 3. 保存处理后的数据 ---
    data.to_csv('data/data_processed.csv', sep=',', index=False)
    print("data_processed.csv saved.")

    # --- 4. 创建ID到索引的映射 ---
    print("Building ID to Index mappings...")
    unique_users = data['user_id'].unique()
    unique_questions = data['problem_id'].unique()

    unique_skills = set()
    data['skill_id'].astype(str).str.split('_').apply(unique_skills.update)
    unique_skills = sorted([int(float(s)) for s in unique_skills if s])

    num_user = len(unique_users)
    num_q = len(unique_questions)
    num_s = len(unique_skills)

    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    question2idx = {qid: i + 1 for i, qid in enumerate(unique_questions)}
    question2idx[0] = 0  # Padding token
    skill2idx = {sid: i for i, sid in enumerate(unique_skills)}

    num_q_total = num_q + 1

    print(f"Total Users: {num_user}, Total Questions: {num_q} (+1 padding), Total Skills: {num_s}")

    # 保存所有映射文件
    np.save('data/question2idx.npy', question2idx)
    np.save('data/skill2idx.npy', skill2idx)
    np.save('data/user2idx.npy', user2idx)
    np.save('data/idx2question.npy', {v: k for k, v in question2idx.items()})
    np.save('data/idx2skill.npy', {v: k for k, v in skill2idx.items()})
    np.save('data/idx2user.npy', {v: k for k, v in user2idx.items()})
    print("Mapping files saved.")

    # --- 5. 生成 Question-Skill 关系表 ---
    if not os.path.exists('data/qs_table.npz'):
        print("Building Question-Skill table (qs_table)...")
        qs_table = np.zeros([num_q_total, num_s], dtype=np.int8)

        q_skill_map = data[['problem_id', 'skill_id']].drop_duplicates('problem_id')

        for _, row in q_skill_map.iterrows():
            q_idx = question2idx[row['problem_id']]
            skills = str(row['skill_id']).split('_')
            for s_str in skills:
                try:
                    s_id = int(float(s_str))
                    if s_id in skill2idx:
                        s_idx = skill2idx[s_id]
                        qs_table[q_idx, s_idx] = 1
                except ValueError:
                    continue

        # 保存 qs_table 和相关的关系表
        sparse_qs_table = sparse.csr_matrix(qs_table)
        qq_table = np.matmul(qs_table, qs_table.T)
        ss_table = np.matmul(qs_table.T, qs_table)

        sparse.save_npz('data/qs_table.npz', sparse_qs_table)
        sparse.save_npz('data/qq_table.npz', sparse.csr_matrix(qq_table))
        sparse.save_npz('data/ss_table.npz', sparse.csr_matrix(ss_table))
        print("qs_table.npz, qq_table.npz, ss_table.npz saved.")
    else:
        print("Question-Skill tables already exist.")

    # --- 6. 生成 User-Question 关系表 (3D版本) ---
    if not os.path.exists('data/uq_table.npy'):
        print("Building 3D User-Question table (uq_table.npy)...")
        uq_table_3d = np.zeros([num_user, num_q_total, 3], dtype=np.float32)

        for _, row in data.iterrows():
            u_idx = user2idx[row['user_id']]
            q_idx = question2idx[row['problem_id']]

            uq_table_3d[u_idx, q_idx, 0] = row['ability_factor']
            uq_table_3d[u_idx, q_idx, 1] = row['attempt_factor_g']
            uq_table_3d[u_idx, q_idx, 2] = row['hint_factor_g']

        np.save('data/uq_table.npy', uq_table_3d)
        print("uq_table.npy saved.")
    else:
        print("uq_table.npy already exists.")

    # --- 7. 生成 User-Question 关系表 (2D稀疏版本) ---
    if not os.path.exists('data/uq_table.npz'):
        print("Building 2D User-Question table (uq_table.npz)...")
        uq_table_2d = np.zeros([num_user, num_q_total], dtype=np.float32)

        for _, row in data.iterrows():
            u_idx = user2idx[row['user_id']]
            q_idx = question2idx[row['problem_id']]

            factors = np.array([row['attempt_factor_g'], row['hint_factor_g'], row['ability_factor']])
            factor_value = np.sum(factors * weights)
            uq_table_2d[u_idx, q_idx] = factor_value

        sparse_uq_table = sparse.csr_matrix(uq_table_2d)
        sparse.save_npz('data/uq_table.npz', sparse_uq_table)
        print("uq_table.npz saved.")
    else:
        print("uq_table.npz already exists.")

    # --- 8. 生成用户序列数据 ---
    if not os.path.exists('data/user_seq.npy'):
        print("Building user sequence data...")

        # 确保数据按用户和时间排序
        data = data.sort_values(by=['user_id', 'order_id'])

        # 初始化序列数组
        user_seq = np.zeros([num_user, max_seq_len], dtype=np.int32)
        user_res = np.zeros([num_user, max_seq_len], dtype=np.int32)
        user_mask = np.zeros([num_user, max_seq_len], dtype=np.int32)
        user_user = np.zeros([num_user, max_seq_len], dtype=np.int32)

        # 为每个用户构建序列
        for user_id in unique_users:
            u_idx = user2idx[user_id]
            user_data = data[data['user_id'] == user_id].head(max_seq_len)

            seq_len = len(user_data)
            if seq_len > 0:
                user_seq[u_idx, :seq_len] = user_data['problem_id'].map(question2idx).values
                user_res[u_idx, :seq_len] = user_data['correct'].values
                user_mask[u_idx, :seq_len] = 1
                user_user[u_idx, :seq_len] = u_idx

        # 保存所有序列文件
        np.save('data/user_seq.npy', user_seq)
        np.save('data/user_res.npy', user_res)
        np.save('data/user_mask.npy', user_mask)
        np.save('data/user_user.npy', user_user)
        print("User sequence files saved.")
    else:
        print("User sequence files already exist.")

    # --- 9. 生成统一的用户记录文件 ---
    if not os.path.exists('data/user_records.npy'):
        print("Building unified user records...")
        user_records = np.zeros([num_user, max_seq_len, 4], dtype=np.int32)

        for user_id in unique_users:
            u_idx = user2idx[user_id]
            user_data = data[data['user_id'] == user_id].head(max_seq_len)
            seq_len = len(user_data)

            if seq_len > 0:
                user_records[u_idx, :seq_len, 0] = u_idx
                user_records[u_idx, :seq_len, 1] = user_data['problem_id'].map(question2idx).values
                user_records[u_idx, :seq_len, 2] = user_data['correct'].values
                user_records[u_idx, :seq_len, 3] = 1

        np.save('data/user_records.npy', user_records)
        print("user_records.npy saved.")
    else:
        print("user_records.npy already exists.")

    print("\n=== All files processed successfully! ===")
    print("Generated files:")
    print("- data_processed.csv")
    print("- question2idx.npy, skill2idx.npy, user2idx.npy")
    print("- idx2question.npy, idx2skill.npy, idx2user.npy")
    print("- qs_table.npz, qq_table.npz, ss_table.npz")
    print("- uq_table.npy (3D), uq_table.npz (2D sparse)")
    print("- user_seq.npy, user_res.npy, user_mask.npy, user_user.npy")
    print("- user_records.npy")