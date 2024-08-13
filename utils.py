import numpy as np
from scipy import sparse

def build_adj_list():

    qs_table = sparse.load_npz('data/qs_table.npz').toarray()
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    q_neighbors_list = [[] for _ in range(num_question)]
    s_neighbors_list = [[] for _ in range(num_skill)]
    for q_id in range(num_question):
        s_ids = np.reshape(np.argwhere(qs_table[q_id] > 0), [-1]).tolist()
        q_neighbors_list[q_id] += s_ids
        for s_id in s_ids:
            s_neighbors_list[s_id].append(q_id)
    return q_neighbors_list, s_neighbors_list

def gen_sqgkt_graph(q_neighbors_list, s_neighbors_list, q_neighbor_size=4, s_neighbor_size=10):

    num_question = len(q_neighbors_list)
    num_skill = len(s_neighbors_list)
    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32)
    s_neighbors = np.zeros([num_skill, s_neighbor_size], dtype=np.int32)
    for i, neighbors in enumerate(q_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= q_neighbor_size:
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for i, neighbors in enumerate(s_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= s_neighbor_size:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=True)
    return q_neighbors, s_neighbors


def build_adj_list_uq():
    uq_table = sparse.load_npz('data/uq_table.npz').toarray()
    num_user = uq_table.shape[0]
    num_question = uq_table.shape[1]
    u_neighbors_list = [[] for _ in range(num_user)]
    q_neighbors_list = [[] for _ in range(num_question)]
    for u_id in range(num_user):

        q_ids = np.reshape(np.argwhere(uq_table[u_id] > 0), [-1]).tolist()
        u_neighbors_list[u_id] += q_ids
        for q_id in q_ids:
            q_neighbors_list[q_id].append(u_id)
    return u_neighbors_list, q_neighbors_list
# q->u  s-q
def gen_sqgkt_graph_uq(u_neighbors_list, q_neighbors_list, u_neighbor_size, q_neighbor_size_2):

    num_user = len(u_neighbors_list)
    num_question = len(q_neighbors_list)
    u_neighbors_uq = np.zeros([num_user, u_neighbor_size], dtype=np.int32)
    q_neighbors_uq = np.zeros([num_question, q_neighbor_size_2], dtype=np.int32)
    for i, neighbors in enumerate(u_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= u_neighbor_size:
            u_neighbors_uq[i] = np.random.choice(neighbors, u_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            u_neighbors_uq[i] = np.random.choice(neighbors, u_neighbor_size, replace=True)
    for i, neighbors in enumerate(q_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= q_neighbor_size_2:
            q_neighbors_uq[i] = np.random.choice(neighbors, q_neighbor_size_2, replace=False)
            continue
        if len(neighbors) > 0:
            q_neighbors_uq[i] = np.random.choice(neighbors, q_neighbor_size_2, replace=True)
    return u_neighbors_uq, q_neighbors_uq