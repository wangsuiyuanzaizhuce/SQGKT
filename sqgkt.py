import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
from params import DEVICE
import torch.nn as nn
import torch.nn.functional as F


class sqgkt(Module):
    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table, num_user, u_neighbors,
                 q_neighbors_2, uq_table, agg_hops=3, emb_dim=100,
                 dropout=(0.2, 0.4), hard_recap=False, rank_k=10):
        super(sqgkt, self).__init__()
        self.model_name = "sqgkt"
        self.num_question = num_question
        self.num_skill = num_skill
        self.q_neighbors = q_neighbors
        self.s_neighbors = s_neighbors
        self.num_user = num_user
        self.u_neighbors = u_neighbors
        self.q_neighbors_2 = q_neighbors_2
        self.agg_hops = agg_hops
        self.qs_table = qs_table
        self.uq_table = uq_table
        self.emb_dim = emb_dim
        self.hard_recap = hard_recap
        self.rank_k = rank_k

        self.emb_table_question = Embedding(num_question, emb_dim)
        self.emb_table_question_2 = Embedding(num_question, emb_dim)
        self.emb_table_skill = Embedding(num_skill, emb_dim)
        self.emb_table_user = Embedding(num_user, emb_dim)
        self.emb_table_response = Embedding(2, emb_dim)

        self.w1_q = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w2_q = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w_c = nn.Parameter(torch.tensor(0.33, requires_grad=True))
        self.w_p = nn.Parameter(torch.tensor(0.33, requires_grad=True))
        self.w_n = nn.Parameter(torch.tensor(0.33, requires_grad=True))

        self.lstm_cell = LSTMCell(input_size=emb_dim, hidden_size=emb_dim)
        self.fusion_layer = Linear(emb_dim * 2, emb_dim)

        self.mlps4agg = ModuleList(Linear(emb_dim, emb_dim) for _ in range(agg_hops))
        self.MLP_AGG_last = Linear(emb_dim, emb_dim)
        self.dropout_lstm = Dropout(dropout[0])
        self.dropout_gnn = Dropout(dropout[1])

        self.MLP_query = Linear(emb_dim, emb_dim)
        self.MLP_key = Linear(emb_dim, emb_dim)
        self.MLP_W = Linear(2 * emb_dim, 1)

    def forward(self, user, question, response, mask):
        batch_size, seq_len = question.shape
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        u_neighbor_size, q_neighbor_size_2 = self.u_neighbors.shape[1], self.q_neighbors_2.shape[1]

        state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE)
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)

        h = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
        c = torch.zeros(batch_size, self.emb_dim, device=DEVICE)

        for t in range(seq_len - 1):
            mask_t = mask[:, t].bool()

            if not mask_t.any():
                state_history[:, t] = h.clone()
                continue

            user_t, question_t, response_t = user[:, t], question[:, t], response[:, t]

            # --- Q-S Graph Aggregation ---
            node_neighbors_qs = [question_t[mask_t]]
            _batch_size = len(node_neighbors_qs[0])
            for i in range(self.agg_hops):
                nodes_current = node_neighbors_qs[-1].reshape(-1)
                neighbor_shape = [_batch_size] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in
                                                  range(i + 1)]
                if i % 2 == 0:
                    node_neighbors_qs.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                else:
                    node_neighbors_qs.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))

            emb_node_neighbor_qs = [self.emb_table_question(nodes) if i % 2 == 0 else self.emb_table_skill(nodes) for
                                    i, nodes in enumerate(node_neighbors_qs)]
            emb0_question_t = self.aggregate(emb_node_neighbor_qs)

            emb_question_t = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            emb_question_t[mask_t] = emb0_question_t
            emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t])

            # --- U-Q Graph Aggregation ---
            node_neighbors_uq = [user_t[mask_t]]
            _batch_size_uq = len(node_neighbors_uq[0])
            for i in range(self.agg_hops):
                nodes_current = node_neighbors_uq[-1].reshape(-1)
                neighbor_shape = [_batch_size_uq] + [(u_neighbor_size if j % 2 == 0 else q_neighbor_size_2) for j in
                                                     range(i + 1)]
                if i % 2 == 0:
                    node_neighbors_uq.append(self.u_neighbors[nodes_current].reshape(neighbor_shape))
                else:
                    node_neighbors_uq.append(self.q_neighbors_2[nodes_current].reshape(neighbor_shape))

            emb_node_neighbor_uq = [self.emb_table_user(nodes) if i % 2 == 0 else self.emb_table_question_2(nodes) for
                                    i, nodes in enumerate(node_neighbors_uq)]
            emb0_question_t_2 = self.aggregate_uq(emb_node_neighbor_uq, node_neighbors_uq)

            emb_question_t_2 = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            emb_question_t_2[mask_t] = emb0_question_t_2
            emb_question_t_2[~mask_t] = self.emb_table_question_2(question_t[~mask_t])

            # --- LSTM Update ---
            emb_hat_q = self.w1_q * emb_question_t + self.w2_q * emb_question_t_2
            emb_response_t = self.emb_table_response(response_t)

            interaction_emb = torch.cat((emb_hat_q, emb_response_t), dim=1)
            e_t = F.relu(self.fusion_layer(interaction_emb))

            h_prev_masked, c_prev_masked = h[mask_t], c[mask_t]
            e_t_masked = e_t[mask_t]

            h_next_masked, c_next_masked = self.lstm_cell(e_t_masked, (h_prev_masked, c_prev_masked))

            h_new, c_new = h.clone(), c.clone()
            h_new[mask_t] = h_next_masked
            c_new[mask_t] = c_next_masked
            h, c = h_new, c_new

            lstm_output = self.dropout_lstm(h)
            state_history[:, t] = lstm_output

            # --- Prediction Preparation ---
            q_next = question[:, t + 1]
            emb_q_next = self.emb_table_question(q_next)

            skills_related_list = [self.emb_table_skill(torch.nonzero(s, as_tuple=True)[0]) for s in
                                   self.qs_table[q_next]]
            max_num_skill = max(s.shape[0] for s in skills_related_list) if skills_related_list else 0

            qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim, device=DEVICE)
            for i, emb_skills in enumerate(skills_related_list):
                emb_next = emb_q_next[i].unsqueeze(0)
                if emb_skills.nelement() == 0:
                    qs_concat[i, 0, :] = emb_next
                else:
                    qs_concat[i, :(1 + emb_skills.shape[0]), :] = torch.cat((emb_next, emb_skills), dim=0)

            # --- History Recap ---
            current_state = lstm_output.unsqueeze(1)
            history_slice = state_history[:, :t].clone()

            if history_slice.shape[1] == 0:
                current_history_state = current_state
            elif t < self.rank_k:
                current_history_state = torch.cat((current_state, history_slice), dim=1)
            else:
                Q = self.emb_table_question(q_next).clone().detach().unsqueeze(-1)
                K = self.emb_table_question(question[:, :t]).clone().detach()
                product_score = torch.bmm(K, Q).squeeze(-1)

                _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                select_history = history_slice.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.emb_dim))

                current_history_state = torch.cat((current_state, select_history), dim=1)

            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state)

        return y_hat

    def aggregate(self, emb_node_neighbor):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        return torch.relu(self.MLP_AGG_last(emb_node_neighbor[0]))

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = (emb_sum_neighbor + emb_self)
        return torch.relu(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def aggregate_uq(self, emb_node_neighbor, node_neighbors_2):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate_uq(
                    emb_node_neighbor[j], emb_node_neighbor[j + 1],
                    node_neighbors_2[j], node_neighbors_2[j + 1], j
                )
        return torch.relu(self.MLP_AGG_last(emb_node_neighbor[0]))

    def sum_aggregate_uq(self, emb_self, emb_neighbor, self_nodes_ids, neighbor_nodes_ids, hop):
        if hop % 2 == 0:
            user_ids_for_lookup = self_nodes_ids
            question_ids_for_lookup = neighbor_nodes_ids
        else:
            user_ids_for_lookup = neighbor_nodes_ids
            question_ids_for_lookup = self_nodes_ids

        if user_ids_for_lookup.dim() < neighbor_nodes_ids.dim():
            expanded_user_ids = user_ids_for_lookup.unsqueeze(-1).expand_as(neighbor_nodes_ids)
        else:
            expanded_user_ids = user_ids_for_lookup

        if question_ids_for_lookup.dim() < neighbor_nodes_ids.dim():
            expanded_question_ids = question_ids_for_lookup.unsqueeze(-1).expand_as(neighbor_nodes_ids)
        else:
            expanded_question_ids = question_ids_for_lookup

        node_weights = self.uq_table[expanded_user_ids, expanded_question_ids]

        c_i = node_weights[..., 0].unsqueeze(-1)
        g_p = node_weights[..., 1].unsqueeze(-1)
        g_n = node_weights[..., 2].unsqueeze(-1)

        fusion_weights = self.w_c * c_i + self.w_p * g_p + self.w_n * g_n

        # 移除了会导致刷屏和性能问题的 print 语句

        weighted_neighbor_embs = emb_neighbor * fusion_weights
        weighted_emb_neighbor_sum = torch.mean(weighted_neighbor_embs, dim=-2)

        emb_sum = emb_self + weighted_emb_neighbor_sum
        return torch.relu(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def recap_hard(self, q_next, q_history):
        pass

    def predict(self, qs_concat, current_history_state):
        if current_history_state.shape[1] == 0:
            return torch.zeros(qs_concat.shape[0], device=qs_concat.device)

        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2))
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]

        states = current_history_state.unsqueeze(1).repeat(1, num_qs, 1, 1)
        qs_concat2 = qs_concat.unsqueeze(2).repeat(1, 1, num_state, 1)

        K = torch.tanh(self.MLP_query(states))
        Q = torch.tanh(self.MLP_key(qs_concat2))

        tmp = self.MLP_W(torch.cat((Q, K), dim=-1)).squeeze(-1)

        alpha = torch.softmax(tmp, dim=2)
        p = torch.sum(alpha * output_g, dim=(-1, -2))

        return p