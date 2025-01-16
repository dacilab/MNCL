import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import os
import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)

def read_cf_new():
    # reading rating file
    rating_file = 'data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    # print(rating_np)
    #print(n_ratings) = 42346
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    # print(eval_indices)  从 0 到 n_ratings-1 的索引中随机抽取一定数量的索引来生成的，这样就得到了测试集的索引。
    left = set(range(n_ratings)) - set(eval_indices)
    # test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False).不允许重复抽取
    train_indices = list(left)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    # test_data = rating_np[test_indices]

    train_rating = rating_np[train_indices]
    # n_ratings = train_rating.shape[0]
    # print(n_ratings) = 33877
    ui_adj = generate_ui_adj(rating_np, train_rating)
    return train_data, eval_data, ui_adj

def generate_ui_adj(rating, train_rating):
    """
    生成用户-物品邻接矩阵。

    Args:
    - rating (numpy.ndarray): 包含所有评分信息的数组，每行包含用户、物品和评分。
    - train_rating (numpy.ndarray): 训练集的评分信息，每行包含用户、物品和评分。

    Returns:
    - ui_adj (scipy.sparse.dok_matrix): 表示用户-物品邻接矩阵的稀疏矩阵。
    """
    #ui_adj = sp.dok_matrix((n_user, n_item), dtype=np.float32)
    n_user, n_item = len(set(rating[:, 0])), len(set(rating[:, 1]))
    # print(n_user, n_item) 1872 3846个唯一元素

    # 使用 train_rating 中的信息创建初始的用户-物品邻接矩阵
    ui_adj_orign = sp.coo_matrix(
        (train_rating[:, 2], (train_rating[:, 0], train_rating[:, 1])), shape=(n_user, n_item)).todok()
    # print(train_rating[:, 0].shape)
    # print(train_rating[:, 1].shape)
    # print(ui_adj_orign)
    # print(ui_adj_orign.shape)  (1872, 3846)
    """  (0, 20)	1
         (0, 21)	1
         (0, 22)	1
         ...
         (1871, 3252)	0
         (1871, 3363)	0
         (1871, 3494)	1
    """

    # 创建一个块矩阵，左上角和右下角块分别为 None，左下角块为原始用户-物品邻接矩阵的转置，右上角块为原始用户-物品邻接矩阵
    ui_adj = sp.bmat([[None, ui_adj_orign],
                    [ui_adj_orign.T, None]], dtype=np.float32)
    # print(ui_adj)
    ui_adj = ui_adj.todok()
    # print(ui_adj)
    # user-item adjacency matrix (5718, 5718)
    print('already create user-item adjacency matrix', ui_adj.shape)
    return ui_adj

def remap_item(train_data, eval_data):
    global n_users, n_items
    # 计算训练数据和评估数据中用户和物品的最大ID
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1])) + 1

    # 从eval_data中提取标签（评分），这里假设评分为1表示用户与物品有交互
    eval_data_label = eval_data.take([2], axis=1)
    # 找到评分为1的索引
    indix_click = np.where(eval_data_label == 1)
    # print(indix_click)
    # 从eval_data中提取评分为1的行
    eval_data = eval_data.take(indix_click[0], axis=0)

    # 仅从train_data和eval_data中提取用户和物品列
    eval_data = eval_data.take([0, 1], axis=1)
    train_data = train_data.take([0, 1], axis=1)

    # 用训练数据填充train_user_set，将用户ID作为键，对应物品ID的列表作为值
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))

    # 用评估数据填充test_user_set，将用户ID作为键，对应物品ID的列表作为值
    for u_id, i_id in eval_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    # 从文件中加载三元组数据
    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)

    # 去除重复的三元组
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # 获取具有反向方向的三元组，如 <entity, is-aspect-of, item>
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1

        # 考虑两个额外的关系---'interact'和'be interacted'
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1

        # 获取知识图谱的完整版本
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)#沿着第一个轴（行方向）连接

    else:
        # 考虑两个额外的关系---'interact'
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    # print(max(triplets[:, 0])) 9365
    # print(max(triplets[:, 2])) 9365
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    # 创建一个用于存储协同过滤和知识图谱数据的有向多图
    ckg_graph = nx.MultiDiGraph()
    # defaultdict 用于按关系 ID 存储三元组
    rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    # 将交互三元组添加到 rd defaultdict 中，并显示进度
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    # 将知识图谱三元组添加到 ckg_graph 和 rd defaultdict 中，并显示进度
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])


    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
    # 定义计算对称归一化拉普拉斯矩阵的函数
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        # 计算 D^{-1/2}
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        # 计算对称归一化拉普拉斯矩阵
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
    # 定义计算单侧归一化拉普拉斯矩阵的函数
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        # 计算 D^{-1}
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # 计算单侧归一化拉普拉斯矩阵
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    # 存储关系矩阵的列表
    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            # 处理交互关系（用户-物品），将物品部分的节点编号加上用户数
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            # 创建交互关系矩阵
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            # 处理其他关系，创建相应的关系矩阵
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        # 将关系矩阵添加到列表中
        adj_mat_list.append(adj)

    # 计算对称归一化拉普拉斯矩阵和单侧归一化拉普拉斯矩阵的列表
    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]

    # 对于交互关系，仅保留用户部分，截取相应的子矩阵
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # norm_mat_list[0] = norm_mat_list[0].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'


    print('reading train and test user-item set ...')
    # train_cf = read_cf(directory + 'ratings_final.txt')
    # test_cf = read_cf(directory + 'test.txt')
    train_cf, eval_cf, ui_adj = read_cf_new()
    remap_item(train_cf, eval_cf)   # 重新映射用户和物品，创建用户集合.

    print('combinating train_cf and kg data ...')
    # 读取知识图谱数据（三元组）文件，存储在 triplets 变量中
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    # 构建图结构和关系字典，包括用户-物品交互关系和知识图谱中的关系
    # graph.add_edge(h_id, t_id, key=r_id)
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    # 构建稀疏关系图的邻接矩阵列表、规范化矩阵列表和平均矩阵列表
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    # 存储参数数量的字典
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
              }
    # 用户字典，包括训练集和测试集的用户集合
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }
    return train_cf, eval_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]

