import pandas as pd
import numpy as np
import os
from scipy import sparse
from scipy.stats import norm, poisson

min_seq_len = 20
max_seq_len = 200
k = 0.3
d = 0.7
b = 10

if __name__ == '__main__':
    data = pd.read_csv(filepath_or_buffer='skill_builder_data_corrected_collapsed.csv', encoding="ISO-8859-1",
                       dtype={17: str, 18: str})
    data = data.sort_values(by='user_id', ascending=True)
    data = data.drop(data[data['skill_id'] == ' '].index)
    data = data.dropna(subset=['skill_id'])
    data = data.drop(data[data['original'] == 0].index)
    is_valid_user = data.groupby('user_id').size() >= min_seq_len
    data = data[data['user_id'].isin(is_valid_user[is_valid_user].index)]
    data = data.loc[:, ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'skill_name',
                        'ms_first_response', 'answer_type', 'attempt_count', 'hint_count']]
    question_attempt_stats = data.groupby('problem_id')['attempt_count'].mean().reset_index()
    question_attempt_stats.rename(columns={'attempt_count': 'mean_attempt'}, inplace=True)
    data = pd.merge(data, question_attempt_stats, on='problem_id', suffixes=('', '_attempt'))
    data['attempt_factor'] = 1 - poisson(data['mean_attempt']).cdf(data['attempt_count'] - 1)
    data['attempt_factor_g'] = k + (1 - k) / (1 + np.exp(-d * (data['attempt_factor'] - b)))

    question_hint_stats = data.groupby('problem_id')['hint_count'].agg('mean').reset_index()
    question_hint_stats.rename(columns={'hint_count': 'mean_hint'}, inplace=True)
    data = pd.merge(data, question_hint_stats, on='problem_id')

    # data['hint_count'] = data['hint_count']
    # CDF
    data['hint_factor'] = 1 - poisson(data['mean_hint']).cdf(data['hint_count'] - 1)
    data['hint_factor_g'] = k + (1 - k) / (1 + np.exp(-d * (data['hint_factor'] - b)))
    data['ability_factor'] = data.groupby('user_id')['correct'].transform('mean')

    num_answer = data.shape[0]
    questions = set()
    skills = set()
    users = set()

    for row in data.itertuples(index=False):
        users.add(row[1])
        questions.add(row[2])
        if isinstance(row[4], (int, float)):
            skills.add(int(row[4]))
        else:
            skill_add = set(int(s) for s in row[4].split('_'))
            skills = skills.union(skill_add)
    data.to_csv('data/data_processed.csv', sep=',', index=False)

    num_q = len(questions)
    num_s = len(skills)
    num_user = len(users)
    if not os.path.exists('data/question2idx.npy'):

        questions = list(questions)
        skills = list(skills)
        users = list(users)
        question2idx = {questions[i]: i + 1 for i in range(num_q)}
        question2idx[0] = 0
        skill2idx = {skills[i]: i for i in range(num_s)}
        user2idx = {users[i]: i for i in range(num_user)}
        num_q += 1
        idx2question = {question2idx[q]: q for q in question2idx}
        idx2skill = {skill2idx[s]: s for s in skill2idx}
        idx2user = {user2idx[u]: u for u in user2idx}

        np.save('data/question2idx.npy', question2idx)
        np.save('data/skill2idx.npy', skill2idx)
        np.save('data/user2idx.npy', user2idx)
        np.save('data/idx2question.npy', idx2question)
        np.save('data/idx2skill.npy', idx2skill)
        np.save('data/idx2user.npy', idx2user)
    else:
        question2idx = np.load('data/question2idx.npy', allow_pickle=True).item()
        skill2idx = np.load('data/skill2idx.npy', allow_pickle=True).item()
        user2idx = np.load('data/user2idx.npy', allow_pickle=True).item()
        idx2question = np.load('data/idx2question.npy', allow_pickle=True).item()
        idx2skill = np.load('data/idx2skill.npy', allow_pickle=True).item()
        idx2user = np.load('data/idx2user.npy', allow_pickle=True).item()

    # row[1]:user_id, row[2]:problem_id, row[4]:skill_id
    if not os.path.exists('data/qs_table.npz'):
        qs_table = np.zeros([num_q, num_s], dtype=float)
        q_set = data['problem_id'].drop_duplicates()
        q_samples = pd.concat([data[data['problem_id'] == q_id].sample(1) for q_id in q_set])
        for row in q_samples.itertuples(index=False):
            if isinstance(row[4], (int, float)):
                qs_table[question2idx[row[2]], skill2idx[int(row[4])]] = 1
            else:
                skill_add = [int(s) for s in row[4].split('_')]
                for s in skill_add:
                    qs_table[question2idx[row[2]], skill2idx[s]] = 1
        qq_table = np.matmul(qs_table, qs_table.T)
        ss_table = np.matmul(qs_table.T, qs_table)
        qs_table = sparse.coo_matrix(qs_table)
        qq_table = sparse.coo_matrix(qq_table)
        ss_table = sparse.coo_matrix(ss_table)
        sparse.save_npz('data/qs_table.npz', qs_table)
        sparse.save_npz('data/qq_table.npz', qq_table)
        sparse.save_npz('data/ss_table.npz', ss_table)
    else:
        qs_table = sparse.load_npz('data/qs_table.npz').toarray()
        qq_table = sparse.load_npz('data/qq_table.npz').toarray()
        ss_table = sparse.load_npz('data/ss_table.npz').toarray()

    # 修正：统一使用 uq_table_3d 文件名和变量名
    if not os.path.exists('data/uq_table_3d.npy'):
        print("开始创建三维 uq_table...")
        # 创建三维数组 [num_user, num_q, 3]
        uq_table_3d = np.zeros([num_user, num_q, 3], dtype=float)
        u_set = data['user_id'].drop_duplicates()
        u_samples = pd.concat([data[data['user_id'] == u_id].sample(1) for u_id in u_set])

        processed_count = 0
        for row in u_samples.itertuples(index=False):
            if processed_count % 500 == 0:
                print(f"处理用户 {processed_count}/{len(u_samples)}")

            try:
                user_idx = user2idx[row[1]]
                question_idx = question2idx[row[2]]

                # 分别存储三个特征
                uq_table_3d[user_idx, question_idx, 0] = row.attempt_factor_g
                uq_table_3d[user_idx, question_idx, 1] = row.hint_factor_g
                uq_table_3d[user_idx, question_idx, 2] = row.ability_factor
                processed_count += 1
            except Exception as e:
                print(f"处理用户 {row[1]}, 问题 {row[2]} 时出错: {e}")
                continue

        # 保存为npy文件（三维数组不适合sparse格式）
        np.save('data/uq_table_3d.npy', uq_table_3d)
        print(f"三维 uq_table 创建完成，形状: {uq_table_3d.shape}")
    else:
        # 修正：加载正确的文件
        uq_table_3d = np.load('data/uq_table_3d.npy')
        print(f"加载三维 uq_table，形状: {uq_table_3d.shape}")

    if not os.path.exists('data/user_seq.npy'):
        user_seq = np.zeros([num_user, max_seq_len])
        user_res = np.zeros([num_user, max_seq_len])
        user_user = np.zeros([num_user, max_seq_len])
        num_seq = [0 for _ in range(num_user)]
        user_mask = np.zeros([num_user, max_seq_len])
        for row in data.itertuples(index=False):
            user_id = user2idx[row[1]]
            if num_seq[user_id] < max_seq_len - 1:
                user_seq[user_id, num_seq[user_id]] = question2idx[row[2]]
                user_res[user_id, num_seq[user_id]] = row[3]
                user_mask[user_id, num_seq[user_id]] = 1
                user_user[user_id, num_seq[user_id]] = user_id
                num_seq[user_id] += 1
        np.save('data/user_seq.npy', user_seq)
        np.save('data/user_res.npy', user_res)
        np.save('data/user_mask.npy', user_mask)
        np.save('data/user_user.npy', user_mask)
        print("用户序列文件创建完成")