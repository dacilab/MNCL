# For any questions, don't hesitate to contact me: Ding Zou (m202173662@hust.edu.cn)
import random
import torch
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
from utils.data_loader import load_data
from model.Mymodel import Recommender
from utils import evaluation
from utils.evaluate import test
from utils.helper import early_stopping

import logging
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end):
    """
    构建用于模型输入的 feed_dict。

    Args:
    - train_entity_pairs (list): 训练实体对的列表，每个元素是形如 [user, item, label] 的三元组。
    - start (int): 起始索引。
    - end (int): 终止索引。

    Returns:
    - feed_dict (dict): 包含模型输入信息的字典。
    """
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict


def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)

def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:

        batch = get_feed_dict(data, start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        _, scores, _, _ = model(batch)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def topk_eval(model, train_data, data):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall

def early_stopping(log_value, best_value, test_f1,early_test_f1, stopping_step, expected_order='acc', flag_step=10):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    early_test_f1 = early_test_f1

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
        early_test_f1 = test_f1
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, early_test_f1, stopping_step, should_stop

# def test():
#     def process_bar(num, total):
#         rate = float(num) / total
#         ratenum = int(50 * rate)
#         r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum * 2)
#         sys.stdout.write(r)
#         sys.stdout.flush()
#
#     # predict
#     rec_list = {}
#     user_count = len(test_set)
#     for i, user in enumerate(test_set):
#         candidates = self.predict(user)
#         rated_list, li = self.data.user_rated(user)
#         for item in rated_list:
#             candidates[self.data.item[item]] = -10e8
#         ids, scores = self.find_k_largest(self.max_N, candidates)
#         item_names = [self.data.id2item[iid] for iid in ids]
#         rec_list[user] = list(zip(item_names, scores))
#         if i % 1000 == 0:
#             process_bar(i, user_count)
#     process_bar(user_count, user_count)
#     print('')
#     return rec_list

# def fast_evaluation(epoch):
#     print('Evaluating the model...')
#     rec_list = test()
#     measure = ranking_evaluation(test_set, rec_list, [max_N])
#     if len(self.bestPerformance) > 0:
#         count = 0
#         performance = {}
#         for m in measure[1:]:
#             k, v = m.strip().split(':')
#             performance[k] = float(v)
#         for k in self.bestPerformance[1]:
#             if self.bestPerformance[1][k] > performance[k]:
#                 count += 1
#             else:
#                 count -= 1
#         if count < 0:
#             self.bestPerformance[1] = performance
#             self.bestPerformance[0] = epoch + 1
#             self.save()
#     else:
#         self.bestPerformance.append(epoch + 1)
#         performance = {}
#         for m in measure[1:]:
#             k, v = m.strip().split(':')
#             performance[k] = float(v)
#         self.bestPerformance.append(performance)
#         self.save()
#     print('-' * 120)
#     print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
#     measure = [m.strip() for m in measure[1:]]
#     print('*Current Performance*')
#     print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
#     bp = ''
#     bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
#     bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
#     bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
#     bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
#     print('*Best Performance* ')
#     print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
#     print('-' * 120)
#     return measure

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    print(graph)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    early_stop_step = 15
    early_test_f1 = 0
    test_f1 = 0
    epoch = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf, s, s + args.batch_size)
            batch_loss, _, _, _ = model(batch)
            # batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            # cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()

        if 1:
            """testing"""
            test_s_t = time()

            test_auc, test_f1 = ctr_eval(model, test_cf_pairs)
            # test_recall = topk_eval(model, train_cf_pairs, test_cf_pairs)

            test_e_t = time()

            train_res = PrettyTable()

            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "test auc", "test f1"]
            train_res.add_row([epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_auc, test_f1])
            print(train_res)

        cur_best_pre_0, early_test_f1, stopping_step, should_stop = early_stopping(test_auc, cur_best_pre_0, test_f1,
                                                                                   early_test_f1, stopping_step,
                                                                                   expected_order='acc',
                                                                                   flag_step=early_stop_step)

        # train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "test recall"]
        # train_res.add_row([epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_recall])
        # print(train_res)

        if stopping_step == 0:
            print("###find better!")
            print('best test auc:%.4f' % (cur_best_pre_0))
            print('test f1:%.4f' % (early_test_f1))
        elif should_stop:
            break

    print('early stopping at %d, test_auc:%.4f, test_f1:%.4f' % (epoch-15, cur_best_pre_0, early_test_f1))
