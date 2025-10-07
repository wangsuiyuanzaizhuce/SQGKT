import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
from params import DEVICE
from scipy import sparse
import torch.nn as nn
import torch.nn.functional as F





class sqgkt(Module):
# num_question：问题总数, num_skill：技能总数, q_neighbors：问题技能图（问题）, s_neighbors：问题技能图（技能）, qs_table：问题技能关系权重, num_user：学生总数, u_neighbors：学生问题图（学生）,
#                  q_neighbors_2：学生问题图（问题）, uq_table：学生表现信息（包含了学习能力、尝试次数、提示次数等重要信息，用于学生问题图的权重）, agg_hops=3：GCN聚合的跳数, emb_dim=100：嵌入向量的维度,
#                  dropout=(0.2, 0.4)：第一个是LSTM中使用，第二个是GCN中使用, hard_recap=True：历史回顾策略，Ture是用硬编码（目前传入的是False）, rank_k=10：历史状态中选取最相关的10个状态
    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table, num_user, u_neighbors,
                 q_neighbors_2, uq_table, agg_hops=3, emb_dim=100,
                 dropout=(0.2, 0.4), hard_recap=True, rank_k=10):
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

        # 定义嵌入层
        # 问题——技能图嵌入
        self.emb_table_question = Embedding(num_question, emb_dim)
        # 学生——问题图嵌入
        self.emb_table_question_2 = Embedding(num_question, emb_dim)
        # 技能嵌入
        self.emb_table_skill = Embedding(num_skill, emb_dim)
        # 学生嵌入
        self.emb_table_user = Embedding(num_user, emb_dim)
        # 答题结果嵌入，输入是0/1（对错） ， 输出是对应的嵌入向量
        self.emb_table_response = Embedding(2, emb_dim)

        # requires_grad=True代表他们都是可学习的
        # 融合两个图的时候用到的参数
        self.w1_q = nn.Parameter(torch.tensor(0.5, requires_grad=True)) 
        self.w2_q = nn.Parameter(torch.tensor(0.5, requires_grad=True))  

        # 针对学生学习能力，尝试因子和提示因子的初始化
        self.w_c = nn.Parameter(torch.tensor(0.33, requires_grad=True))  
        self.w_p = nn.Parameter(torch.tensor(0.33, requires_grad=True))
        self.w_n = nn.Parameter(torch.tensor(0.33, requires_grad=True))

        # LSTM单元的定义，输入是融合后问题的嵌入和作答结果的嵌入拼接起来的，所以这里维度是emb_dim * 2 ，隐藏状态是学生的知识状态和嵌入维度相同
        self.lstm_cell = LSTMCell(input_size=emb_dim * 2, hidden_size=emb_dim)
        # 为GCN的每一个聚合都定义了一个MLP。这是GCN中的权重矩阵 (W)，用于对聚合后的信息进行线性变换。
        # 这里for _ in range(agg_hops)代表，我们不关心序号，只关心执行agg_hops次这件事，
        # 要用ModuleList，因为如果用普通的 Python 列表，PyTorch无法看到这些 Linear 层，这样才可以看到Linear层的偏置和权重，在优化器的时候会更新他们的参数，也能被移动到GPU
        self.mlps4agg = ModuleList(Linear(emb_dim, emb_dim) for _ in range(agg_hops))
        # GCN聚合之后，最后的线性变换层
        self.MLP_AGG_last = Linear(emb_dim, emb_dim)
        # 分别定义了LSTM和GCN的dropout层
        self.dropout_lstm = Dropout(dropout[0])
        self.dropout_gnn = Dropout(dropout[1])
        # 三个线性层，实现predict函数中的加性注意力机制，query和key分别被转换，之后MLP_W将他们拼接起来，投影到一个向量，作为注意力分数
        self.MLP_query = Linear(emb_dim, emb_dim)
        self.MLP_key = Linear(emb_dim, emb_dim)
        self.MLP_W = Linear(2 * emb_dim, 1)

        # todo
        # 这两个参数没有被使用，需要进一步查看是什么（和注意力相关）
        # self.attention_weights = torch.nn.Parameter(torch.randn(3, requires_grad=True))
        # self.attention_bias = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    # 接收学生ID ， 问题ID ， 作答结果和掩码（mask）作为输入。mask用于标识哪些是真实数据，哪些是为对齐序列长度而填充的padding
    def forward(self, user, question, response, mask):
        # 获得批次大小和序列长度
        batch_size, seq_len = question.shape
        # 获取先前就预定义的邻居数量
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        u_neighbor_size, q_neighbor_size_2 = self.u_neighbors.shape[1], self.q_neighbors_2.shape[1]

        # todo
        # LSTM的初始隐藏状态，但是从未被使用
        # h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        # h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))

        # 创建了一个全0的张量，用来存储每一个时间步t后，LSTM的隐藏状态h_t
        state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE)
        # 创建一个全0张量，存储模型在每一步的预测概率
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)

        # 时间步循环 ， 从0到seq_len-2
        for t in range(seq_len - 1):
            # 获取当前时间步t的批次数据
            # 这里的question：batch_size：一个批次里有多少个学生；seq_len：每个学生的答题序列有多长（通常会通过填充padding来使所有序列等长）。
            user_t = user[:, t]
            question_t = question[:, t]
            response_t = response[:, t]
            # 当前时间步的掩码，后续的很多操作只会对mask_t是Ture的进行操作，这里逻辑很简单，用eq进行比较，如果前面mask和后面tensor(1)都是1，就输出Ture，并记录
            mask_t = torch.eq(mask[:, t], torch.tensor(1))
            # Embedding 是把 0 变成一个低维、稠密的、可学习的向量，比如 [0.12, -0.45, 0.88, ...]（emb_dim维）；把 1 变成另一个 emb_dim 维的向量。
            emb_response_t = self.emb_table_response(response_t)

            # 对Q-S异构图进行图卷积操作，为当前时间步的问题question_t生成融合了多跳邻居（相关技能、相关问题）信息的增强表示。
            # 初始化了一个列表，存放各个邻居节点ID，第0层就是问题本身，只会处理没有被mask的有效数据
            node_neighbors = [question_t[mask_t]]
            # 当前时间步有效的、未被mask的学生数量
            _batch_size = len(node_neighbors[0])
            # 循环GCN的层数，应该是越多就越深
            for i in range(self.agg_hops):
                # node_neighbors是最后一行，也就是所有学生在时间t的时候回答问题的序号（排除掉了被掩码的），之后展平成一维张量
                nodes_current = node_neighbors[-1].reshape(-1)
                neighbor_shape = [_batch_size] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in
                                                  range(i + 1)]
                # 实现了在Q-S异构图上的交替游走：Q -> S -> Q -> ...
                #[
                #    tensor([101, 201]),  # 第0跳 (Q), shape: (2,)
                #    tensor([[11, 12, 13],  # 第1跳 (S), shape: (2, 3)
                #            [21, 22, 23]]),
                #    tensor([[[111, 112, 113, 114],  # 第2跳 (Q), shape: (2, 3, 4)
                #             [121, 122, 123, 124],
                #             [131, 132, 133, 134]],

                #            [[211, 212, 213, 214],
                #             [221, 222, 223, 224],
                #             [231, 232, 233, 234]]])
                #]
                # 从问题节点出发，找邻居技能节点
                if i % 2 == 0:
                    node_neighbors.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                # 从技能节点出发，找邻居问题节点
                else:
                    node_neighbors.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
            # 创建一个列表，存储嵌入向量（并非ID）
            #[
            #    tensor(...),  # 第0跳嵌入 (Q), shape: (2, emb_dim)
            #    tensor(...),  # 第1跳嵌入 (S), shape: (2, 3, emb_dim)
            #    tensor(...)  # 第2跳嵌入 (Q), shape: (2, 3, 4, emb_dim)
            #]
            # 这里是一个一维数组，每一个元素是一大坨
            emb_node_neighbor = []
            # 0，2，4跳就是嵌入问题表格，反之
            for i, nodes in enumerate(node_neighbors):
                if i % 2 == 0:
                    emb_node_neighbor.append(self.emb_table_question(nodes))
                else:
                    emb_node_neighbor.append(self.emb_table_skill(nodes))
            # 图聚合，最后就是一个第0跳嵌入 (Q), shape: (2, emb_dim)，只有第0跳，但是里面包含了后面所有的内容
            emb0_question_t = self.aggregate(emb_node_neighbor)
            # 创建一个用于存放最终结果的容器张量。有批次，形状（100）的全0张量
            emb_question_t = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            # 这里将费尽心思搞出来的emb0_question_t，上面只是假设形状是（2，100），2是有效的数量，实际上可能是上百个。匹配进emb_question_t[mask_t]去，上面的例子来看，形状可能是：
            # [[...增强嵌入_学生0...],
            # [...增强嵌入_学生1...],
            # [0., 0., 0., ..., 0.]]   第3个学生的位置仍然是0
            # 一行是一个学生这个t时间中回答的问题题号，所以这个自变量是那个emb_question_t
            emb_question_t[mask_t] = emb0_question_t
            # 为那些被填充的、无效的学生提供一个合理的嵌入值。直接让它们保持为0可能会在后续计算（如LSTM）中引入问题或被错误地解释
            # self.emb_table_question(...): 这会获取这些padding问题的原始嵌入（即没有经过图聚合的、在嵌入表里最原始的向量）
            emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t])

           
            node_neighbors_2 = [user_t[mask_t]]
            _batch_size_2 = len(node_neighbors_2[0])
            for i in range(self.agg_hops):
                nodes_current_2 = node_neighbors_2[-1].reshape(-1)
                neighbor_shape_2 = [_batch_size] + [(u_neighbor_size if j % 2 == 0 else q_neighbor_size_2) for j in
                                                    range(i + 1)]
                if i % 2 == 0:
                    node_neighbors_2.append(self.u_neighbors[nodes_current_2].reshape(neighbor_shape_2))
                else:
                    node_neighbors_2.append(self.q_neighbors_2[nodes_current_2].reshape(neighbor_shape_2))
            emb_node_neighbor_2 = []
            for i, nodes in enumerate(node_neighbors_2):
                if i % 2 == 0:
                    emb_node_neighbor_2.append(self.emb_table_user(nodes))
                else:
                    emb_node_neighbor_2.append(self.emb_table_question_2(nodes))
            emb0_question_t_2 = self.aggregate_uq(emb_node_neighbor_2)
            emb_question_t_2 = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            emb_question_t_2[mask_t] = emb0_question_t_2
            emb_question_t_2[~mask_t] = self.emb_table_question_2(question_t[~mask_t])

            
            emb_hat_q = self.w1_q * emb_question_t + self.w2_q * emb_question_t_2 

           
            lstm_input = torch.cat((emb_hat_q, emb_response_t), dim=1)  
            lstm_output = self.dropout_lstm(self.lstm_cell(lstm_input)[0])  

       
            q_next = question[:, t + 1]
            skills_related = self.qs_table[q_next]
            skills_related_list = []
            max_num_skill = 1
            for i in range(batch_size):
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                if len(skills_index.shape) == 0:
                    skills_related_list.append(torch.unsqueeze(self.emb_table_skill(skills_index), dim=0))
                else:
                    skills_related_list.append(self.emb_table_skill(skills_index))
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]

            emb_q_next = self.emb_table_question(q_next)
            qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim).to(DEVICE)
            for i, emb_skills in enumerate(skills_related_list):
                num_qs = 1 + emb_skills.shape[0]
                emb_next = torch.unsqueeze(emb_q_next[i], dim=0)
                qs_concat[i, 0: num_qs] = torch.cat((emb_next, emb_skills), dim=0)

            if t == 0:
                y_hat[:, 0] = 0.5
                y_hat[:, 1] = self.predict(qs_concat, torch.unsqueeze(lstm_output, dim=1))
                continue

            if self.hard_recap:
                history_time = self.recap_hard(q_next, question[:, 0:t])
                selected_states = []
                max_num_states = 1
                for row, selected_time in enumerate(history_time):
                    current_state = torch.unsqueeze(lstm_output[row], dim=0)
                    if len(selected_time) == 0:
                        selected_states.append(current_state)
                    else:
                        selected_state = state_history[row, torch.tensor(selected_time, dtype=torch.int64)]
                        selected_states.append(torch.cat((current_state, selected_state), dim=0))
                        if (selected_state.shape[0] + 1) > max_num_states:
                            max_num_states = selected_state.shape[0] + 1
                current_history_state = torch.zeros(batch_size, max_num_states, self.emb_dim).to(DEVICE)
                for b, c_h_state in enumerate(selected_states):
                    num_states = c_h_state.shape[0]
                    current_history_state[b, 0: num_states] = c_h_state
            else:
                current_state = lstm_output.unsqueeze(dim=1)
                if t <= self.rank_k:
                    current_history_state = torch.cat((current_state, state_history[:, 0:t]), dim=1)
                else:
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1)
                    K = self.emb_table_question(question[:, 0:t]).clone().detach()
                    product_score = torch.bmm(K, Q).squeeze(dim=-1)
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                    select_history = torch.cat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                     for i in range(batch_size)), dim=0)
                    current_history_state = torch.cat((current_state, select_history), dim=1)

            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state)
            state_history[:, t] = lstm_output
        return y_hat

    # 双层循环，第二层从后往前，例子：比如总共有3跳，现在的第0跳就包含了1，2跳的内容；第一跳就包含了第二跳的内容
    def aggregate(self, emb_node_neighbor):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        # self.MLP_AGG_last经过线性层计算后的新张量，形状通常保持不变，仍为 (_batch_size, 100)；tanh 函数 (双曲正切函数) 会将输入张量中的每一个元素值都映射到 (-1, 1) 的区间内。
        return torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0]))

    # 邻居跳之间融合用的
    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        # 比如现在是(_batch_size, 3, 4, 100)；torch.mean 会沿着倒数第二个维度计算平均值。这意味着，对于每个一跳邻居，它会把它所有（4个）二跳邻居的嵌入向量取平均，融合成一个单一的嵌入向量
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        # 直接逐元素相加。融合后的结果，形状不变，仍为 (_batch_size, 3, 100)
        emb_sum = (emb_sum_neighbor + emb_self)
        # self.mlps4agg[hop](emb_sum))：将融合后的 emb_sum 通过这个线性层，进行一次仿射变换（W*X + b），这是GCN层中的权重矩阵部分，输出形状不变，仍为 (_batch_size, 3, 100)
        # dropout_gnn正则化；
        return torch.tanh(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def aggregate_uq(self, emb_node_neighbor):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate_uq(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        return torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0]))

    def sum_aggregate_uq(self, emb_self, emb_neighbor, hop):
        num_nodes = emb_self.size(0)
        embedding_dim = emb_self.size(1)
        weighted_emb_neighbor_sum = torch.zeros_like(emb_self)

        for i in range(num_nodes):
            neighbor_embs = emb_neighbor[i]  # [neighbor_size, embedding_dim]
            node_weights = self.uq_table[i, :neighbor_embs.size(0), :]  # [neighbor_size, 3]

         
            c_i = node_weights[:, 0].unsqueeze(-1)  # [neighbor_size, 1]
            g_p = node_weights[:, 1].unsqueeze(-1)  # [neighbor_size, 1]
            g_n = node_weights[:, 2].unsqueeze(-1)  # [neighbor_size, 1]

           
            # g_ij = c_i + g_ij^p + g_ij^n
            fusion_weights = self.w_c * c_i + self.w_p * g_p + self.w_n * g_n  # [neighbor_size, 1]

   
            expanded_fusion_weights = fusion_weights.expand_as(neighbor_embs)
            weighted_neighbor_embs = neighbor_embs * expanded_fusion_weights

        
            weighted_emb_neighbor_sum[i] = torch.mean(weighted_neighbor_embs, dim=0)

    
        emb_sum = emb_self + weighted_emb_neighbor_sum
        return torch.tanh(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def recap_hard(self, q_next, q_history):
        batch_size = q_next.shape[0]
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        q_next = q_next.reshape(-1)
        skill_related = self.q_neighbors[q_next].reshape((batch_size, q_neighbor_size)).reshape(-1)
        q_related = self.s_neighbors[skill_related].reshape((batch_size, q_neighbor_size * s_neighbor_size)).tolist()
        time_select = [[] for _ in range(batch_size)]
        for row in range(batch_size):
            key = q_history[row].tolist()
            query = q_related[row]
            for t, k in enumerate(key):
                if k in query:
                    time_select[row].append(t)
        return time_select

    def predict(self, qs_concat, current_history_state):
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2))
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1).repeat(1, num_qs, 1, 1)
        qs_concat2 = torch.unsqueeze(qs_concat, dim=2).repeat(1, 1, num_state, 1)
        K = torch.tanh(self.MLP_query(states))
        Q = torch.tanh(self.MLP_key(qs_concat2))
        tmp = self.MLP_W(torch.cat((Q, K), dim=-1))
        tmp = torch.squeeze(tmp, dim=-1)
        alpha = torch.softmax(tmp, dim=2)
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)
        result = torch.sigmoid(torch.squeeze(p, dim=-1))
        return result
