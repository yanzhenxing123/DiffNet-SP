from __future__ import division
from collections import defaultdict
import numpy as np
from time import time
import random
import tensorflow as tf

"""
self.social_neighbors: {user_id0:{user_id1, user_id2}} 类似于 hash_data




"""


class DataModule():
    def __init__(self, conf, filename, social_filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.social_filename = social_filename
        self.index = 0
        self.social_index = 0

    #######  Initalize Procedures #######
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            # self.arrangePositiveData()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
            data_dict['CONSUMED_ITEMS_VALUES_WEIGHT_AVG_INPUT'] = self.consumed_items_values_weight_avg_list
            data_dict['CONSUMED_ITEMS_NUM_INPUT'] = self.consumed_item_num_list  # [len(item_ids)]
            data_dict['CONSUMED_ITEMS_NUM_DICT_INPUT'] = self.user_item_num_dict
            data_dict['USER_ITEM_SPARSITY_DICT'] = self.user_item_sparsity_dict
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.generateSocialNeighborsSparseMatrix()
            data_dict[
                'SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list  # [(user_id1, user_id2)]，二者为朋友 # 259014个 已排序
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list  # [1.0 / len(friends)]
            data_dict['SOCIAL_NEIGHBORS_VALUES_WEIGHT_AVG_INPUT'] = self.social_neighbors_values_weight_avg_list
            data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'] = self.social_neighbor_num_list  # [len(friends)]
            data_dict['SOCIAL_NEIGHBORS_NUM_DICT_INPUT'] = self.social_neighbors_num_dict
            data_dict['USER_USER_SPARSITY_DICT'] = self.user_user_sparsity_dict
        if 'ITEM_CUSTOMER_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrixForItemUser()
            data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_list
            data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_list
            data_dict['ITEM_CUSTOMER_VALUES_WEIGHT_AVG_INPUT'] = self.item_customer_values_weight_avg_list
            data_dict['ITEM_CUSTOMER_NUM_INPUT'] = self.item_customer_num_list
            data_dict['ITEM_USER_NUM_DICT_INPUT'] = self.item_user_num_dict
        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.readSocialNeighbors()
        self.arrangePositiveData()
        self.arrangePositiveDataForItemUser()
        self.generateTrainNegative()
        self.generateSocialTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.readSocialNeighbors()
        self.arrangePositiveData()
        self.arrangePositiveDataForItemUser()
        self.generateTrainNegative()
        self.generateSocialTrainNegative()

    def initalizeRankingEva(self):
        self.readData()
        self.readSocialNeighbors()
        self.getEvaPositiveBatch()
        self.generateEvaNegative()
        self.getSocialEvaPositiveBatch()
        self.generateSocialEvaNegative()

    def linkedMap(self):
        """
        关联数据字典
        :return:
        """
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['LABEL_LIST'] = self.labels_list

        self.data_dict['SOCIAL_USER_LIST'] = self.social_user_list
        self.data_dict['SOCIAL_FRIEND_LIST'] = self.social_friend_list
        self.data_dict['SOCIAL_LABEL_LIST'] = self.social_labels_list

    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

    def linkedSocialRankingEvaMap(self):
        self.data_dict['SOCIAL_EVA_USER_LIST'] = self.social_eva_user_list
        self.data_dict['SOCIAL_EVA_FRIEND_LIST'] = self.social_eva_friend_list

    #######  Data Loading #######
    def readData(self):
        """
        读取user-item数据
        :return:
        """
        f = open(self.filename)
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))

        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data

    # ----------------------
    # Read social network information
    def readSocialNeighbors(self, friends_flag=1):
        """
        读取user-user数据
        :param friends_flag:
        :return:
        """
        social_neighbors = defaultdict(set)
        social_neighbors_num_dict = defaultdict(set)
        social_total_data = set()
        social_total_user_list = set()

        # links_file = open(self.conf.links_filename)
        links_file = open(self.social_filename)
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            social_total_data.add((u1, u2))
            social_total_user_list.add(u1)
            if friends_flag == 1:
                social_neighbors[u2].add(u1)
                social_total_data.add((u2, u1))
                social_total_user_list.add(u2)

        user_list = sorted(list(social_neighbors.keys()))
        for u in range(self.conf.num_users):
            social_neighbors_num_dict[u] = len(social_neighbors[u]) + 1

        self.social_neighbors_num_dict = social_neighbors_num_dict
        self.social_neighbors = social_neighbors
        self.social_total_data = social_total_data
        self.social_total_user_list = list(social_total_user_list)

    def arrangePositiveDataForItemUser(self):
        positive_data_for_item_user = defaultdict(set)
        item_user_num_dict = defaultdict(set)

        total_data_for_item_user = set()
        hash_data_for_item_user = self.hash_data
        for (u, i) in hash_data_for_item_user:
            total_data_for_item_user.add((i, u))
            positive_data_for_item_user[i].add(u)

        item_list = sorted(list(positive_data_for_item_user.keys()))

        for i in range(self.conf.num_items):
            item_user_num_dict[i] = len(positive_data_for_item_user[i]) + 1

        self.item_user_num_dict = item_user_num_dict
        self.positive_data_for_item_user = positive_data_for_item_user
        self.total_data_for_item_user = len(total_data_for_item_user)

    # ----------------------
    # This function designes for generating train/val/test negative  
    def generateTrainNegative(self):
        """
        generateTrainNegative
        生成user-item负面样本
        :return:
        """
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1

    def generateSocialTrainNegative(self):
        """
        generateSocialTrainNegative
        生成user-user负面样本
        :return:
        """
        num_items = self.conf.num_users
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        for (u1, u2) in self.social_total_data:
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u1, j) in self.social_total_data:
                    j = np.random.randint(num_items)
                negative_data[u1].add(j)
        self.social_negative_data = negative_data
        self.terminal_flag = 1

    # ----------------------
    # This function designes for val/test set, compute loss
    def getVTRankingOneBatch(self):
        """
        validation and test
        :return:
        """
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list

        user_list = []
        item_list = []
        labels_list = []

        social_user_list = []
        social_friend_list = []
        social_labels_list = []

        for u in total_user_list:
            # 1. user-item
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))

            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))

            # 2. user-user
            social_user_list.extend([u] * len(self.social_neighbors[u]))
            social_friend_list.extend(list(self.social_neighbors[u]))
            social_labels_list.extend([1] * len(self.social_neighbors[u]))

            social_user_list.extend([u] * len(self.social_negative_data[u]))
            social_friend_list.extend(list(self.social_negative_data[u]))
            social_labels_list.extend([0] * len(self.social_negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])

        self.social_user_list = np.reshape(social_user_list, [-1, 1])
        self.social_friend_list = np.reshape(social_friend_list, [-1, 1])
        self.social_labels_list = np.reshape(social_labels_list, [-1, 1])

    # ----------------------
    # This function designes for the training process
    def getTrainRankingBatch(self):
        """
        train
        :return:
        """
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.conf.training_batch_size

        user_list = []  # [0...0, 1, 1, 1, user_id3, user_id3]
        item_list = []  # [item_id1, item_id2]
        labels_list = []  # [1,1,1,1]

        social_user_list = []
        social_friend_list = []
        social_labels_list = []

        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index + batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
            self.terminal_flag = 0

        for u in target_user_list:
            # 1. user-item
            # positive
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))

            # negative
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))

            # 2. user-user
            # positive
            social_user_list.extend([u] * len(self.social_neighbors[u]))
            social_friend_list.extend(list(self.social_neighbors[u]))
            social_labels_list.extend([1] * len(self.social_neighbors[u]))

            # negative
            social_user_list.extend([u] * len(self.social_negative_data[u]))
            social_friend_list.extend(list(self.social_negative_data[u]))
            social_labels_list.extend([0] * len(self.social_negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])

        self.social_user_list = np.reshape(social_user_list, [-1, 1])
        self.social_friend_list = np.reshape(social_friend_list, [-1, 1])
        self.social_labels_list = np.reshape(social_labels_list, [-1, 1])

    # ----------------------
    # This function is designed for the positive data
    def getEvaPositiveBatch(self):
        """
        eva就是用测试集做评价集
        得到 user-item positive
        :return:
        """
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict

    def getSocialEvaPositiveBatch(self):
        """
        eva就是用测试集做评价集
        得到 user-user positive
        :return:
        """
        social_total_data = self.social_total_data
        user_list = []
        friend_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in social_total_data:
            user_list.append(u)
            friend_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.social_eva_user_list = np.reshape(user_list, [-1, 1])
        self.social_eva_friend_list = np.reshape(friend_list, [-1, 1])
        self.social_eva_index_dict = index_dict

    # ----------------------
    # This function is designed for generating negative data
    def generateEvaNegative(self):
        """
        生成eva的user-item负面数据
        :return:
        """
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_items
        eva_negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                eva_negative_data[u].append(j)
        self.eva_negative_data = eva_negative_data

    def generateSocialEvaNegative(self):
        """
        生成eva的user-user负面数据
        :return:
        """
        social_total_data = self.social_total_data
        social_total_user_list = self.social_total_user_list
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_users
        social_eva_negative_data = defaultdict(list)

        for u in social_total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in social_total_data:
                    j = np.random.randint(num_items)
                social_eva_negative_data[u].append(j)
        self.social_eva_negative_data = social_eva_negative_data

    # ----------------------
    # This function designs for generating negative batch in rating evaluation,
    def getEvaRankingBatch(self):
        """
        user-item getEvaRankingBatch
        :return:
        """
        batch_size = self.conf.evaluate_batch_size  # 2560
        num_evaluate = self.conf.num_evaluate  # 1000
        eva_negative_data = self.eva_negative_data  # 负面数据
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index + batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0  # 停止
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u] * num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag


    def getSocialEvaRankingBatch(self):
        """
        user-user getSocialEvaRankingBatch
        :return:
        """
        batch_size = self.conf.evaluate_batch_size  # 2560
        num_evaluate = self.conf.num_evaluate  # 1000
        social_eva_negative_data = self.social_eva_negative_data  # 负面数据
        social_total_user_list = self.social_total_user_list

        index = self.social_index
        terminal_flag = 1
        total_users = len(social_total_user_list)
        user_list = []
        friend_list = []
        if index + batch_size < total_users:
            batch_user_list = social_total_user_list[index:index + batch_size]
            self.social_index = index + batch_size
        else:
            terminal_flag = 0  # 停止
            batch_user_list = social_total_user_list[index:total_users]
            self.social_index = 0
        for u in batch_user_list:
            user_list.extend([u] * num_evaluate)
            friend_list.extend(social_eva_negative_data[u])
        self.social_eva_user_list = np.reshape(user_list, [-1, 1])
        self.social_eva_friend_list = np.reshape(friend_list, [-1, 1])
        return batch_user_list, terminal_flag

    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        user_item_num_dict = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)

        user_list = sorted(list(positive_data.keys()))
        for u in range(self.conf.num_users):
            user_item_num_dict[u] = len(positive_data[u]) + 1

        self.positive_data = positive_data
        self.user_item_num_dict = user_item_num_dict
        self.total_data = len(total_data)

    # ----------------------
    # Generate Social Neighbors Sparse Matrix Indices and Values
    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors
        social_neighbors_num_dict = self.social_neighbors_num_dict  # weight avg

        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_values_weight_avg_list = []
        social_neighbor_num_list = []  # [len(friends)]
        social_neighbors_dict = defaultdict(list)  # {user_id:[friend_id1, friend_id2]} value进行了排序，本质还是 social_neighbors

        user_user_num_for_sparsity_dict = defaultdict(set)
        user_user_sparsity_dict = {}

        user_user_sparsity_dict['0-4'] = []
        user_user_sparsity_dict['4-8'] = []
        user_user_sparsity_dict['8-16'] = []
        user_user_sparsity_dict['16-32'] = []
        user_user_sparsity_dict['32-64'] = []
        user_user_sparsity_dict['64-'] = []

        for u in range(self.conf.num_users):
            user_user_num_for_sparsity_dict[u] = len(social_neighbors[u])

        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])

        user_list = sorted(list(social_neighbors.keys()))

        # node att
        for user in range(self.conf.num_users):
            if user in social_neighbors_dict:
                social_neighbor_num_list.append(len(social_neighbors_dict[user]))
            else:
                social_neighbor_num_list.append(1)

        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(1.0 / len(social_neighbors_dict[user]))

                # social_neighbors_values_weight_avg_list.append(
                #     1.0 / (np.sqrt(social_neighbors_num_dict[user]) * np.sqrt(
                #         social_neighbors_num_dict[friend]))
                # )  # weight avg

        for u in range(self.conf.num_users):
            cur_user_neighbors_num = user_user_num_for_sparsity_dict[u]
            if ((cur_user_neighbors_num >= 0) & (cur_user_neighbors_num < 4)):
                user_user_sparsity_dict['0-4'].append(u)
            elif ((cur_user_neighbors_num >= 4) & (cur_user_neighbors_num < 8)):
                user_user_sparsity_dict['4-8'].append(u)
            elif ((cur_user_neighbors_num >= 8) & (cur_user_neighbors_num < 16)):
                user_user_sparsity_dict['8-16'].append(u)
            elif ((cur_user_neighbors_num >= 16) & (cur_user_neighbors_num < 32)):
                user_user_sparsity_dict['16-32'].append(u)
            elif ((cur_user_neighbors_num >= 32) & (cur_user_neighbors_num < 64)):
                user_user_sparsity_dict['32-64'].append(u)
            elif (cur_user_neighbors_num >= 64):
                user_user_sparsity_dict['64-'].append(u)

        self.user_user_sparsity_dict = user_user_sparsity_dict
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)
        self.social_neighbors_values_weight_avg_list = np.array(social_neighbors_values_weight_avg_list).astype(
            np.float32)  # weight avg
        self.social_neighbor_num_list = np.array(social_neighbor_num_list).astype(np.int64)
        # self.social_neighbors_values_list = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_list)], stddev=0.01))

    # ----------------------
    # Generate Consumed Items Sparse Matrix Indices and Values
    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_values_weight_avg_list = []
        consumed_item_num_list = []
        consumed_items_dict = defaultdict(list)
        user_item_num_for_sparsity_dict = defaultdict(set)
        user_item_sparsity_dict = {}

        user_item_sparsity_dict['0-4'] = []
        user_item_sparsity_dict['4-8'] = []
        user_item_sparsity_dict['8-16'] = []
        user_item_sparsity_dict['16-32'] = []
        user_item_sparsity_dict['32-64'] = []
        user_item_sparsity_dict['64-'] = []

        consumed_items_num_dict = self.user_item_num_dict  # weight avg
        # social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  # weight avg

        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])

        user_list = sorted(list(positive_data.keys()))

        for u in range(self.conf.num_users):
            user_item_num_for_sparsity_dict[u] = len(positive_data[u])

        for user in range(self.conf.num_users):
            if user in consumed_items_dict:
                consumed_item_num_list.append(len(consumed_items_dict[user]))
            else:
                consumed_item_num_list.append(1)

        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0 / len(consumed_items_dict[u]))
                consumed_items_values_weight_avg_list.append(
                    1.0 / (np.sqrt(consumed_items_num_dict[u]) * np.sqrt(item_user_num_dict[i])))  # weight avg

        for u in range(self.conf.num_users):
            cur_user_consumed_item_num = user_item_num_for_sparsity_dict[u]
            if ((cur_user_consumed_item_num >= 0) & (cur_user_consumed_item_num < 4)):
                user_item_sparsity_dict['0-4'].append(u)
            elif ((cur_user_consumed_item_num >= 4) & (cur_user_consumed_item_num < 8)):
                user_item_sparsity_dict['4-8'].append(u)
            elif ((cur_user_consumed_item_num >= 8) & (cur_user_consumed_item_num < 16)):
                user_item_sparsity_dict['8-16'].append(u)
            elif ((cur_user_consumed_item_num >= 16) & (cur_user_consumed_item_num < 32)):
                user_item_sparsity_dict['16-32'].append(u)
            elif ((cur_user_consumed_item_num >= 32) & (cur_user_consumed_item_num < 64)):
                user_item_sparsity_dict['32-64'].append(u)
            elif (cur_user_consumed_item_num >= 64):
                user_item_sparsity_dict['64-'].append(u)

        self.user_item_sparsity_dict = user_item_sparsity_dict
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)
        self.consumed_items_values_weight_avg_list = np.array(consumed_items_values_weight_avg_list).astype(
            np.float32)  # weight avg
        self.consumed_item_num_list = np.array(consumed_item_num_list).astype(np.int64)

    def generateConsumedItemsSparseMatrixForItemUser(self):
        positive_data_for_item_user = self.positive_data_for_item_user
        item_customer_indices_list = []
        item_customer_values_list = []
        item_customer_values_weight_avg_list = []
        item_customer_num_list = []
        item_customer_dict = defaultdict(list)

        consumed_items_num_dict = self.user_item_num_dict  # weight avg
        # social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  # weight avg

        for i in positive_data_for_item_user:
            item_customer_dict[i] = sorted(positive_data_for_item_user[i])
        item_list = sorted(list(positive_data_for_item_user.keys()))

        for item in range(self.conf.num_items):
            if item in item_customer_dict:
                item_customer_num_list.append(len(item_customer_dict[item]))
            else:
                item_customer_num_list.append(1)

        for i in item_list:
            for u in item_customer_dict[i]:
                item_customer_indices_list.append([i, u])
                item_customer_values_list.append(1.0 / len(item_customer_dict[i]))
                item_customer_values_weight_avg_list.append(
                    1.0 / (np.sqrt(consumed_items_num_dict[u]) * np.sqrt(item_user_num_dict[i])))

        self.item_customer_indices_list = np.array(item_customer_indices_list).astype(np.int64)
        self.item_customer_values_list = np.array(item_customer_values_list).astype(np.float32)
        self.item_customer_num_list = np.array(item_customer_num_list).astype(np.int64)
        self.item_customer_values_weight_avg_list = np.array(item_customer_values_weight_avg_list).astype(np.float32)
