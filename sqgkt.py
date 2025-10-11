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
        self.lstm_cell = LSTMCell(input_size=emb_dim, hidden_size=emb_dim)
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
        self.fusion_layer = Linear(emb_dim * 2, emb_dim)
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
        h = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
        c = torch.zeros(batch_size, self.emb_dim, device=DEVICE)

        # 创建了一个全0的张量，用来存储每一个时间步t后，LSTM的隐藏状态h_t，这里的sqe_len应该是时间序列（每个步长的历史状态都储存在这里，用一个emb_dim大小的向量表示）
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

            # 构建第二种图嵌入，如果很多做对A题的学生也做对了B题，那么B题就是A题在U-Q图上的一个重要邻居。它反映的是学生的行为关联，而不是知识点的内在关联
            node_neighbors_2 = [user_t[mask_t]]
            # 第t时间时没有被掩码的学生数目
            _batch_size_2 = len(node_neighbors_2[0])
            for i in range(self.agg_hops):
                # 在循环的每一步，我们都需要且只需要从最新加入的、最外层的邻居出发，去寻找下一层邻居。[-1] 这个索引正是用来获取这个最新加入的、最外层的邻居张量
                nodes_current_2 = node_neighbors_2[-1].reshape(-1)
                neighbor_shape_2 = [_batch_size] + [(u_neighbor_size if j % 2 == 0 else q_neighbor_size_2) for j in
                                                    range(i + 1)]
                if i % 2 == 0:
                    # 如果不reshape的话，会很乱。有了之后，比如：代码同样会先得到一个 (6, 4) 的查找结果，然后 .reshape([2, 3, 4]) 会把它重塑成一个三维张量
                    node_neighbors_2.append(self.u_neighbors[nodes_current_2].reshape(neighbor_shape_2))
                else:
                    node_neighbors_2.append(self.q_neighbors_2[nodes_current_2].reshape(neighbor_shape_2))
            # 创建一个列表，存储嵌入向量（并非ID）
            # [
            #    tensor(...),  # 第0跳嵌入 (U), shape: (2, emb_dim)
            #    tensor(...),  # 第1跳嵌入 (Q), shape: (2, 3, emb_dim)
            #    tensor(...)  # 第2跳嵌入 (U), shape: (2, 3, 4, emb_dim)
            # ]
            # 这里是一个一维数组，每一个元素是一大坨
            emb_node_neighbor_2 = []
            for i, nodes in enumerate(node_neighbors_2):
                if i % 2 == 0:
                    # 这里嵌入就像查字典一样，用用户的ID那些去查用户的嵌入向量。（用户，对应的嵌入向量（一会是用户，一会是问题的））
                    emb_node_neighbor_2.append(self.emb_table_user(nodes))
                else:
                    emb_node_neighbor_2.append(self.emb_table_question_2(nodes))
            # 汇聚信息到第0跳
            emb0_question_t_2 = self.aggregate_uq(emb_node_neighbor_2 , node_neighbors_2)
            # 创建一个全0的（B，D）来存放最终结果
            emb_question_t_2 = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            # 这里将费尽心思搞出来的emb0_question_t_2，上面只是假设形状是（2，100），2是有效的数量，实际上可能是上百个。匹配进emb_question_t_2[mask_t]去，上面的例子来看，形状可能是：
            # [[...增强嵌入_学生0...],
            # [...增强嵌入_学生1...],
            # [0., 0., 0., ..., 0.]]   第3个学生的位置仍然是0
            # 一行是一个学生这个t时间中回答的问题题号，所以这个自变量是那个emb_question_t_2
            emb_question_t_2[mask_t] = emb0_question_t_2
            # 为那些被填充的、无效的学生提供一个合理的嵌入值。直接让它们保持为0可能会在后续计算（如LSTM）中引入问题或被错误地解释
            # self.emb_table_question_2(...): 这会获取这些padding问题的原始嵌入（即没有经过图聚合的、在嵌入表里最原始的向量）
            emb_question_t_2[~mask_t] = self.emb_table_question_2(question_t[~mask_t])

            # todo 确认是否可以更新
            # 融合两种图嵌入（这里权重理论上可以更新）
            emb_hat_q = self.w1_q * emb_question_t + self.w2_q * emb_question_t_2 

            # 更新学生状态LSTM：emb_hat_q是两个图的融合， emb_response_t是学生回答的正确答案
            # cat是拼接，dim是左右拼接
            # [[ q1, q2, q3, q4,  r1, r2, r3, r4 ],   学生0的完整事件向量
            #  [ q5, q6, q7, q8,  r5, r6, r7, r8 ]]   学生1的完整事件向量
            interaction_emb = torch.cat((emb_hat_q, emb_response_t), dim=1)
            e_t = F.relu(self.fusion_layer(interaction_emb))
            h_prev_masked, c_prev_masked = h[mask_t], c[mask_t]
            e_t_masked = e_t[mask_t]

            h_next_masked, c_next_masked = self.lstm_cell(e_t_masked, (h_prev_masked, c_prev_masked))

            h_new, c_new = h.clone(), c.clone()
            h_new[mask_t] = self.dropout_lstm(h_next_masked)
            c_new[mask_t] = c_next_masked

            h, c = h_new, c_new

            lstm_output = h
            state_history[:, t] = lstm_output

            # 准备下一次预测的问题
            # 核心任务是构建一个张量，该张量完整地描述了下一个挑战是什么。这个挑战由两部分构成：下一个问题 q_next 和 它所关联的所有技能。这个最终的张量将作为预测模块的查询（Query）
            # 获取下一个时间步，所有学生分别要面对的问题ID
            q_next = question[:, t + 1]
            # 预先构建一个问题-技能的关系表，每一个问题应该对应的都是一个one-hot编码的技能表格
            skills_related = self.qs_table[q_next]
            # 处理变长技能列表 (核心难点)，但是我都用one-hot编码了，那不是都是一个张量表格吗，只是说里面的1的数量不同而已，这是没问题的，但是独热编码只是初始形状，我们还需要提取出来，到底是哪些技能，经过转换之后就不一样了（1，2，6），（2，5）这样
            skills_related_list = []
            max_num_skill = 1
            for i in range(batch_size):
                # 返回张量中所有非0元素的索引，torch.nonzero(skills_related[0]) 返回 tensor([[1], [5]])；squeeze() 之后，skills_index 变为 tensor([1, 5])。它的长度是 2
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                # 经过上面的处理，现在每一个元素都是一个一维张量，并且长度不一样（可能）
                # 这个是用来处理问题只关联单个技能这个特殊情况的，skills_related[i] 是 [0, 1, 0, 0, 0, 0, 0, 0]。
                # torch.nonzero(...) 返回一个形状为 (1, 1) 的张量：tensor([[1]])。
                # .squeeze() 之后，skills_index 会把所有大小为1的维度都移除，结果变成一个零维张量 (Scalar)：tensor(1)
                # 简单来说，这里就是一个数字/值，根本没有中括号包裹，因为squeeze的问题，导致一层包裹都没有了
                if len(skills_index.shape) == 0:
                    # 会输出一个嵌入张量（比如100维），unsqueeze会在指定的维度上增加一个大小为1的维度， dim=0代表在最前面增加一个维度，这里形状就变成了（1，100）
                    skills_related_list.append(torch.unsqueeze(self.emb_table_skill(skills_index), dim=0))
                else:
                    # 这里添加的是（N，100）张量
                    skills_related_list.append(self.emb_table_skill(skills_index))
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]

            # 上面总结下来，是获得了每个学生即将要回答对应的技能，行是学生，列是一个二维张量，第一个维度是问题关联的技能数量，第二个维度是技能的嵌入向量
            # 用来获取每个学生下一个问题的嵌入向量（技能关联图的输出）
            emb_q_next = self.emb_table_question(q_next)
            # 创建一个足够大的全0张量容器，+1是为了空出位置存放问题本身
            qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim).to(DEVICE)
            # 遍历所有的学生，获得每个学生回答下一个问题要用到的技能
            for i, emb_skills in enumerate(skills_related_list):
                # num_qs是那个（N，100）的N+1了，方便空出一个位置放原问题
                num_qs = 1 + emb_skills.shape[0]
                # emb_q_next[i] 是第 i 个学生的下一个问题嵌入，是一个一维向量，形状为 (emb_dim，)
                # 因为后面一步的时候，emb_skills是一个二维张量，为了匹配维度，这里才会拼接unsqueeze(..., dim=0) 把一维的 (emb_dim,) 变成了二维的 (1, emb_dim)
                emb_next = torch.unsqueeze(emb_q_next[i], dim=0)
                # 将这个拼接好的、代表“问题+技能”的张量，精准地“填”入我们预先创建好的容器 qs_concat 的第 i 行。后面多余的位置则保持为0（padding）
                # 这里拼接emb_next是（1，问题对应的技能），emb_skills是（N：技能数，技能的嵌入向量）
                qs_concat[i, 0: num_qs] = torch.cat((emb_next, emb_skills), dim=0)

            if t == 0:
                # 第一次交互，没有任何的历史信息，所以给一个猜测：50%
                y_hat[:, 0] = 0.5
                # 调用预测模块，将lstm_output的形状变成（batch_size ， 1 ， emb_dim）,符合预测模块的输入形状
                y_hat[:, 1] = self.predict(qs_concat, torch.unsqueeze(lstm_output, dim=1))
                continue

            if self.hard_recap:
                # q_next: 下一个问题。
                # question[:, 0:t]: 历史上所有做过的问题序列
                # 返回的是一个（batch,[这个学生他之前做问题和下一次要回答的问题有关联的问题是：]）
                history_time = self.recap_hard(q_next, question[:, 0:t])
                # 用来存放每个学生最终被选中的当前状态 + 历史状态 组合
                selected_states = []
                max_num_states = 1
                for row, selected_time in enumerate(history_time):
                    # lstm_output[row]是学生的最新知识状态（形状是emb_dim），增加维度变成（1，emb_dim），方便和后面的二位张量进行拼接
                    current_state = torch.unsqueeze(lstm_output[row], dim=0)
                    # 如果没有找到任何相关的时间点（这道题第一次见类似的）
                    if len(selected_time) == 0:
                        # selected_states中只添加这个学生的当前状态
                        selected_states.append(current_state)
                    else:
                        # 如果找到了
                        # torch.tensor(selected_time, dtype=torch.int64)先从python索引列表转换成索引张量[2, 5] -> tensor([2, 5])
                        # 然后从state_history中抽出时间点是2，5的历史状态，形状是(len(selected_time), emb_dim)，比如这里就是（2(2和5)，100）
                        selected_state = state_history[row, torch.tensor(selected_time, dtype=torch.int64)]
                        # 将当前状态和之前抽出的历史状态沿着0维（行）拼接起来
                        selected_states.append(torch.cat((current_state, selected_state), dim=0))
                        # 如果当前历史状态的数量加上当前状态（1）是最大的，就更新一下
                        if (selected_state.shape[0] + 1) > max_num_states:
                            max_num_states = selected_state.shape[0] + 1
                # 创建并填充统一的张量
                current_history_state = torch.zeros(batch_size, max_num_states, self.emb_dim).to(DEVICE)
                # 遍历每个学生拼接好的、长短不一的状态张量
                for b, c_h_state in enumerate(selected_states):
                    # 每一个学生状态张量的实际长度
                    num_states = c_h_state.shape[0]
                    # 将这个长短不一的状态张量 c_h_state，精确地“填”入大容器 current_history_state 的第 b 行。后面多余的位置则保持为0（padding）
                    current_history_state[b, 0: num_states] = c_h_state
            # 第二种策略，软性回顾
            else:
                # 原本lstm_output形状是（batch_size,emb_dim），变成了（batch_size,1,emb_dim）
                current_state = lstm_output.unsqueeze(dim=1)
                # 历史记录太短了，比要筛选的数量还短
                if t <= self.rank_k:
                    # 直接拼接起来（batch_size,1,emb_dim）和(batch_size, max_seq_len, emb_dim)，这里的拼接就是除了要拼接的维度之外，其它维度都保持一致
                    current_history_state = torch.cat((current_state, state_history[:, 0:t]), dim=1)
                # 当历史记录足够长时，执行相似度筛选。
                else:
                    # 获取 q_next 的嵌入作为查询 (Query)。.clone().detach() 表示计算相似度时不需要计算梯度。unsqueeze 将形状从 (batch_size , emb_dim) 变为  (batch_size , emb_dim , 1)以满足 bmm 的要求。
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1)
                    # 获取所有历史问题的嵌入作为键 (Keys)，形状为  (batch_size , t(这个是动态的，如果是做到第十题，那么这里能看到每个学生0到9次回答的所有问题) , emb_dim（对应问题的嵌入向量）)
                    K = self.emb_table_question(question[:, 0:t]).clone().detach()
                    # 通过批量矩阵乘法计算 Q 和 K 的点积，得到 q_next 与每个历史问题的相似度分数。
                    # 该学生的 K 矩阵形状: (B，t, emb_dim)
                    # 该学生的 Q 矩阵形状: (B，emb_dim, 1)
                    # 所以乘法之后的形状为 (B, t ，1)
                    # 含义：相乘之后是一个下一个问题和历史回答问题状态的乘积，最后直接变成了一个数字（标量）：正是 下一个问题 和 第j个历史问题 之间的点积相似度分数，因为每一个时间点的问题都会和下一个时间的问题计算一次，所以一个学生有t个这样的相似度分数
                    product_score = torch.bmm(K, Q).squeeze(dim=-1)
                    # 针对每个学生的分数（dim如果是0就是学生之间比较，如果是1就是学生自己的t个时间节点的分数），选出最高的rank_k个，并返回他们的索引（indices）
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                    # 用i遍历每一个学生
                    # 获取第i个学生学生的全部历史状态（lstm的隐状态），形状是（max_sqe_len,emb_dim）
                    # 获取第i个学生的最相关历史时刻索引，一个包含了k个索引的一维张量，例如 tensor([2, 5, 4])
                    # 之后用这个最相关的历史索引去这个学生的历史状态中直接抽取出来当时的状态，比如抽出第2，5，4行，并按照或者顺序合成一个新的张量
                    # 最后的形状是(B，k, emb_dim)，其中 k 是 self.rank_k
                    select_history = torch.cat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                     for i in range(batch_size)), dim=0)
                    # 合并操作，当前状态加上历史状态，沿着第1个维度，最后也就是（B，K+1，emb_dim）
                    current_history_state = torch.cat((current_state, select_history), dim=1)
            # 开始预测，并且储存结果
            # qs_concat（当前问题），（batch_size, max_num_skill + 1, self.emb_dim），第1个维度是技能的嵌入向量+问题本身
            # current_history_state是（B，K+1，emb_dim），第1个维度是当前状态+历史状态（两种方法，第一种是用所有相关的，第二种是最相关的几个）
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

    # 总指挥，简单来说，这里依旧是将所有的信息汇聚在第0跳
    def aggregate_uq(self, emb_node_neighbor , node_neighbors_2):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                if j % 2 == 0:
                    emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
                else:
                    emb_node_neighbor[j] = self.sum_aggregate_uq(emb_node_neighbor[j], emb_node_neighbor[j + 1], node_neighbors_2[j], node_neighbors_2[j + 1], j)
        return torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0]))



    # 核心计算部分


    # 核心计算部分
    def sum_aggregate_uq(self, emb_self, emb_neighbor, self_nodes_ids, neighbor_nodes_ids, hop):
        """
        ... (docstring 不变) ...
        """
        # 为了清晰起见，重命名输入
        if hop % 2 == 0:  # 聚合方向: User -> Question
            user_ids_for_lookup = self_nodes_ids
            question_ids_for_lookup = neighbor_nodes_ids
        else:  # 聚合方向: Question -> User
            user_ids_for_lookup = neighbor_nodes_ids
            question_ids_for_lookup = self_nodes_ids

        # --- 【核心修复区域】 ---
        # 目标：让两个ID张量的形状完全一样，都是 (B, M, N)

        # 无论聚合方向如何，我们都需要将维度较少的那个张量扩展到和维度较多的张量一样的形状。
        # 始终将 user_ids_for_lookup 和 question_ids_for_lookup 扩展到 neighbor_nodes_ids 的形状
        # 因为 neighbor_nodes_ids 总是比 self_nodes_ids 多一个维度。

        # 扩展 'user' ID 张量
        if user_ids_for_lookup.dim() < neighbor_nodes_ids.dim():
            expanded_user_ids = user_ids_for_lookup.unsqueeze(-1).expand_as(neighbor_nodes_ids)
        else:
            expanded_user_ids = user_ids_for_lookup

        # 扩展 'question' ID 张量
        if question_ids_for_lookup.dim() < neighbor_nodes_ids.dim():
            expanded_question_ids = question_ids_for_lookup.unsqueeze(-1).expand_as(neighbor_nodes_ids)
        else:
            expanded_question_ids = question_ids_for_lookup

        # --- 【修复结束】 ---

        # 2. 从 uq_table 中批量获取权重
        # uq_table 的维度是 (num_users, num_questions, 3)
        # 现在两个索引张量的形状都是 (B, M, N)，可以安全索引
        node_weights = self.uq_table[expanded_user_ids, expanded_question_ids]  # 输出形状: (B, M, N, 3)

        # ... (方法的其余部分保持不变) ...
        c_i = node_weights[..., 0].unsqueeze(-1)
        g_p = node_weights[..., 1].unsqueeze(-1)
        g_n = node_weights[..., 2].unsqueeze(-1)

        fusion_weights = self.w_c * c_i + self.w_p * g_p + self.w_n * g_n

        weighted_neighbor_embs = emb_neighbor * fusion_weights

        weighted_emb_neighbor_sum = torch.mean(weighted_neighbor_embs, dim=-2)

        emb_sum = emb_self + weighted_emb_neighbor_sum
        return torch.tanh(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def recap_hard(self, q_next, q_history):
        # 批次数目
        batch_size = q_next.shape[0]
        # 这里注释的N和M与其它MN的含义不同，不要搞混（这里并不是N+1那个）
        # self.q_neighbors (问题-技能 关联矩阵)：这是一个形状为 (总问题数, N) 的矩阵，self.q_neighbors[i] 存储了与第 i 个问题相关的 N 个技能的ID。
        # self.s_neighbors (技能-问题 关联矩阵)：这是一个形状为 (总技能数, M) 的矩阵。self.s_neighbors[j] 存储了与第 j 个技能相关的 M 个问题的ID。
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        # q_next的形状统一为一维向量(batch_size,)
        q_next = q_next.reshape(-1)

        # 两级查找
        # 这里使用问题邻居查询技能，得到之后reshape成（batch_size,q_neighbor_size)，之后再拉平
        skill_related = self.q_neighbors[q_next].reshape((batch_size, q_neighbor_size)).reshape(-1)
        # 一样的操作，但这里形状是(batch_size, q_neighbor_size * s_neighbor_size)，原本是技能按照问题划分，现在又再次回到了技能根据学生划分
        # .tolist(): 将整个PyTorch张量转换为Python的嵌套列表，方便后续使用 in 操作符进行查找。q_related 的形状是 (batch_size, M*N) 的列表。
        q_related = self.s_neighbors[skill_related].reshape((batch_size, q_neighbor_size * s_neighbor_size)).tolist()
        # 初始化结果，为每一个学生都初始化一个空的列表，用来存放找到的时间索引
        # 列表可以配合 in 操作符进行高效的成员资格检查
        time_select = [[] for _ in range(batch_size)]
        # 开始遍历
        for row in range(batch_size):
            # 第row个同学的历史作答序列（问题ID）
            #  q_history是question[:, 0:t]
            # .tolist()是转换成python列表
            key = q_history[row].tolist()
            # 技能按照学生来划分的那个列表
            query = q_related[row]
            # 遍历第row个同学的答题记录
            for t, k in enumerate(key):
                # query为下一道题的相关问题池
                if k in query:
                    time_select[row].append(t)
        return time_select

    # 终极目的：计算这两组向量之间的匹配度，得出一个最终的答对概率
    # qs_concat预测的东西：(batch_size, num_qs（N+1）, emb_dim)，current_history_state包含了当前的学生知识状态和一组精选出的相关历史知识状态：(batch_size, num_state（历史状态的数量）, emb_dim)
    def predict(self, qs_concat, current_history_state):
        # torch.transpose(current_history_state, 1, 2): 这是矩阵转置操作：输入的 current_history_state 形状是 (B, N, D) (B:-batch, N:-num_state, D:-emb_dim)，转置后，形状变为 (B, D, N)
        # torch.bmm(...): 批量矩阵乘法：qs_concat是(B, M, D)，current_history_state变成了(B, D, N)吗，乘法之后的形状变成了(B, M（是N+1）, N)
        # B是batch_size
        # qs_concat(需要预测的问题和他涉及的技能集):
        # [ [q1, q2, q3, q4],    <-- 问题自身的向量
        #   [sA1, sA2, sA3, sA4],  <-- 技能A的向量
        #   [sB1, sB2, sB3, sB4] ]  <-- 技能B的向量
        # current_history_state (知识状态集)：
        # 假设我们回顾了 5个相关的历史知识状态，那么 N = 5，嵌入维度 D 当然也是 4。
        # 所以 current_history_state 是一个 (5, 4) 的矩阵：
        # [ [h11, h12, h13, h14],  <-- 知识状态1的向量
        #   [h21, h22, h23, h24],  <-- 知识状态2的向量
        #   [h31, h32, h33, h34],  <-- 知识状态3的向量
        #   [h41, h42, h43, h44],  <-- 知识状态4的向量
        #   [h51, h52, h53, h54] ]  <-- 知识状态5的向量
        # 我们需要考虑到第i个问题/技能和第j个知识之间的匹配程度：a · b = a1*b1 + a2*b2 + a3*b3 + a4*b4.......所以这里转置之后的相乘因为是前面的行乘以后面的列，所以可以完美实现这个计算
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2))
        # shape[1]就是第二个元素，这里的第二个元素一个是N+1，一个是N
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]

        # 精细化注意力分数的计算 (alpha)
        # 这里是准备进行广播（广播机制，就直接理解成从最后一个维度往前看，可以缺少，不能不同，缺少的直接补充上然后就可以计算，这就是广播机制）
        # 在第一个维增加了一个维度，比如从（B,5，4）变成了（B,1,5,4），之后repeat就是沿着每个维度进行复制，接收的参数传入对应的维度：
        # 维度0(Batch): 复制1次(即不复制)。
        # 维度1(我们新加的维度): 复制3次(num_qs次)。
        # 维度2(State): 复制1次(不复制)。
        # 维度3(Embedding): 复制1次(不复制)。
        # 最后是（B，num_qs，5，4）= (B,N+1,M,emb_dim)
        states = torch.unsqueeze(current_history_state, dim=1).repeat(1, num_qs, 1, 1)
        # 这里同理，从（B，N+1 ，emb_dim）变成了(B,N+1,M,emb_dim)
        qs_concat2 = torch.unsqueeze(qs_concat, dim=2).repeat(1, 1, num_state, 1)
        # 这是两个独立的小型神经网络（多层感知机），这里分别经过一层激活函数（非线性），最后形状不变
        K = torch.tanh(self.MLP_query(states))
        Q = torch.tanh(self.MLP_key(qs_concat2))
        # 拼接过后，MLP_W 的输出形状是 (B, M, N, 1）(这里不会自动拿掉，可能是因为线性层计算出来自带的，所以后面还要手动处理一下，把那个只有一个元素的维度转换成正常的数据)
        # self.MLP_W = Linear(2 * emb_dim, 1)
        tmp = self.MLP_W(torch.cat((Q, K), dim=-1))
        # 再次经过squeeze 把最后一个多余的维度去掉，得到形状为 (B, M, N) 的注意力分数矩阵 tmp
        tmp = torch.squeeze(tmp, dim=-1)
        # 归一化，softmax。
        # 这里tmp的形状是（B,M,N）
        # 第二个维度是N（历史状态的维度）进行归一化，也就是说，每一个M上的所有N的和是1
        alpha = torch.softmax(tmp, dim=2)

        # 加权求和与最终预测：alpha (注意力权重) 和 output_g (原始分数矩阵)
        # alpha形状是（B，M，N），output_g的形状也是（B，M，N）
        # 第一层sum是对M（问题-技能维度上求和），第二个sum是对N（知识状态的维度上求和）
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)
        # 输出最终的概率：Sigmoid 函数可以将任意的实数分数 p 压缩到 (0, 1)
        result = torch.sigmoid(torch.squeeze(p, dim=-1))
        return result
