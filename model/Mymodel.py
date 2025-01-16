import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
class Aggregator(nn.Module):
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users
 
    def forward(self, entity_emb, user_emb, edge_index, edge_type, interact_mat, weight):
        # 获取实体数量
        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        # 提取关系嵌入，将关系类型映射到权重矩阵上
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]

        # ------------计算注意力权重 ---------------
        # 调用calculate_sim_hrt函数计算头尾实体之间的相似性权重
        neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail], weight[edge_type - 1])
        # 扩展相似性权重维度以便与邻居实体嵌入相乘
        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
                                                                     neigh_relation_emb.shape[1])
        # 对相似性权重进行归一化，得到注意力权
        neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
        # 利用注意力权重对邻居实体嵌入进行加权求和
        neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)
        # 对实体嵌入进行聚合
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        # 用户聚合，利用交互矩阵和权重矩阵
        user_agg = torch.sparse.mm(interact_mat, entity_emb)
        score = torch.mm(user_emb, weight.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        # 计算头尾实体之间的相似性的函数，使用给定的权重
        # 可以是点积或者基于问题需求的任何其他相似性度量
        # 在提供的代码中未给出，应根据具体用例进行实现
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users, n_relations, interact_mat, ind, node_dropout_rate=0.5, mess_dropout_rate = 0.1):
        super(GraphConv, self).__init__()

        # 模块列表，用于存储多个聚合器（Aggregator）模块
        self.convs = nn.ModuleList()

        # 交互矩阵、关系数量、用户数量等参数
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users

        # 节点和消息的dropout率
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        # 节点采样的索引和Top-K值
        self.ind = ind
        self.topk = 10

        # 参数初始化
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = torch.device("cuda:" + str(0))
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        # 构建多层的Aggregator模块
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        # 消息的dropout层
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # 随机采样边的索引和边类型
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        # 随机选择一定比例的边索引，replace=False表示不放回采样
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        # 根据随机索引获取采样后的边索引和边类型
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        # 对稀疏矩阵进行dropout操作
        # 获取稀疏矩阵非零元素的数量
        noise_shape = x._nnz()

        # 生成随机张量，加上随机噪声
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)

        # 转换为布尔类型的掩码
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        # 获取输入矩阵的行索引和值
        i = x._indices()
        v = x._values()

        # 根据掩码进行dropout，保留非零元素
        i = i[:, dropout_mask]
        v = v[dropout_mask]

        # 构造dropout后的稀疏矩阵
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        # 根据dropout率进行缩放
        return out * (1. / (1 - rate))

    def _convert_sp_mat_to_sp_tensor(self, X):
        # 将输入的稀疏矩阵转换为COO格式
        coo = X.tocoo()

        # 获取行索引、列索引和值，并转换为PyTorch张量
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()

        # 构造稀疏张量
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        # ----------------build item-item graph-------------------
        origin_item_adj = self.build_adj(entity_emb, self.topk)


        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        # update item-item graph
        item_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb,
                   self.topk) + self.lambda_coeff * origin_item_adj


        return entity_res_emb, user_res_emb, item_adj

    def build_adj(self, context, topk):
        # construct similarity adj matrix
        # 计算上下文嵌入向量的相似性
        n_entity = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))

        # 获取每个节点的最近邻节点索引和相似性值
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)

        # 将相似性值和索引转移到GPU
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

        # 构建稀疏张量的索引和值
        y = knn_ind.reshape(-1)
        x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)
        x = x.expand(n_entity, topk).reshape(-1)
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)

        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entity)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
        return L_norm


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.node_dropout_rate = 0.9
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.mess_dropout_rate = 0.9
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        # 初始化图卷积网络和邻接矩阵
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()
        self.lightgcn_layer = 4
        self.n_item_layer = 4
        self.cl_weight = 1
        self.β = float(0)
        self.w = float(0.7)


        # 定义全连接层
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_

        # 初始化所有嵌入向量
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

        # 将邻接矩阵转换为稀疏张量，并移动到指定设备
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        # 该函数用于初始化图卷积网络模型，返回一个初始化后的GraphConv实例。
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        # 创建稀疏张量
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        # 获取行索引和列索引并转换为LongTensor
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        # 将图的边转换为PyTorch张量
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]

        # 提取边的索引和类型
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]

        # 将边索引和类型转换为LongTensor，并移动到指定设备
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None,):
        # 提取用户、物品和标签信息
        user = batch['users']
        item = batch['items']
        labels = batch['labels']

        # 获取用户和物品的嵌入
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]


        # GCN部分
        entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(user_emb,
                                                          item_emb,
                                                          self.edge_index,
                                                          self.edge_type,
                                                          self.interact_mat,
                                                          mess_dropout=self.mess_dropout,
                                                          node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        i_e = entity_gcn_emb[item]
        # Item部分
        i_h = item_emb
        for i in range(self.n_item_layer):
            i_h = torch.sparse.mm(item_adj, i_h)
        i_h = F.normalize(i_h, p=2, dim=1)
        i_e_1 = i_h[item]

        # 更新交互矩阵
        interact_graph = self.permutation_matrix()

        # LightGCN部分
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph, False)
        u_e_1 = user_lightgcn_emb[user]
        i_e_2 = item_lightgcn_emb[item]

        # 计算损失
        loss_contrast = self.calculate_loss(i_e_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_1(i_e, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_2(u_e, u_e_1)

        # 拼接嵌入
        u_e = torch.cat((u_e, u_e_1, u_e_1), dim=-1)
        i_e = torch.cat((i_e, i_e_1, i_e_2), dim=-1)

        # 计算最终的BPR损失
        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast)

    def permutation_matrix(self):

        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)

        return torch.sparse.FloatTensor(indice_new, value_new, torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities]))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # 计算余弦相似度
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        # first calculate the sim rec
        # default = 0.8
        tau = 0.6
        f = lambda x: torch.exp(x / tau)

        # 对嵌入张量进行线性变换
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)

        # 计算同类样本的相似度矩阵和不同类样本的相似度矩阵
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))


        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        ret = loss_1
        ret = ret.mean()
        return ret

    def calculate_loss_1(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6 # default = 0.8
        f = lambda x: torch.exp(x / tau)

        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)


        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calculate_loss_2(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6# default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)

        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def light_gcn(self, user_emb, item_emb, interact_graph, optimize=False):
        # 将用户嵌入和物品嵌入拼接成一个整体的嵌入
        ego_emb = torch.cat((user_emb, item_emb), dim=0)
        all_emb = [ego_emb]

        # 执行多层的图卷积
        for i in range(self.lightgcn_layer):
            ego_emb = torch.sparse.mm(interact_graph, ego_emb)
            if optimize:
                # 如果需要添加扰动，生成随机噪音并添加到嵌入中
                random_noise = torch.rand_like(ego_emb).cuda()
                ego_emb += torch.sign(ego_emb) * F.normalize(random_noise, dim=-1) * self.β
            all_emb.append(ego_emb)
        final_emb = torch.stack(all_emb, dim=1)
        final_emb = torch.mean(final_emb, dim=1)
        # 将最终嵌入分割为用户和物品嵌入
        user_lightgcn_emb, item_lightgcn_emb = torch.split(final_emb,
                                                               [self.n_users, self.n_entities])

        #返回包含用户嵌入和物品嵌入的两个张量
        return user_lightgcn_emb, item_lightgcn_emb

    def create_bpr_loss(self, users, items, labels, loss_contrast):
        # 计算预测分数
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)

        # 计算BCE损失
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())

        # cul regularizer
        # 计算嵌入正则化损失
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # cor_loss = self.sim_decay * cor
        # 计算总损失，包括BCE损失、嵌入正则化损失和对比损失
        return bce_loss + emb_loss + self.cl_weight * loss_contrast, scores, bce_loss, emb_loss
