from __future__ import division
import tensorflow as tf
import numpy as np
from ipdb import set_trace


class diffnetplus():
    """
    图神经网络，diffnet++
    """

    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',  # user_user 稀疏矩阵
            'CONSUMED_ITEMS_SPARSE_MATRIX',  # user_item 稀疏矩阵
            'ITEM_CUSTOMER_SPARSE_MATRIX'  # item_user 稀疏矩阵
        )

    def startConstructGraph(self):
        """
        创建训练模型
        :return:
        """
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def inputSupply(self, data_dict):
        """
        提供输入数据
        :param data_dict:
        :return:
        """
        low_att_std = 1.0

        #  一、Node Attention initialization  Attention节点初始化 构建图神经网路
        # ----------------------
        # 1. user-user social network node attention initialization # user-user 社交网络节点注意力初始化
        self.first_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(units=1,  # 输出神经元的数量 原来为1 还是为1
                                                                               activation=tf.nn.sigmoid,  # sigmoid激活函数
                                                                               name='first_low_att_SN_layer1')

        self.first_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(units=1,  # 输出神经元的数量
                                                                               activation=tf.nn.leaky_relu,  # relu激活函数
                                                                               name='first_low_att_SN_layer2')

        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']  # 259014
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']  # 259014

        self.social_neighbors_values_input1 = tf.reduce_sum(
            tf.math.exp(
                self.first_low_att_layer_for_social_neighbors_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal([len(self.social_neighbors_indices_input)], stddev=low_att_std)
                        ), [-1, 1]
                    )  # 每一层为一个向量
                )
            ), 1
        )  # shape=(259014,) 得到一维的向量

        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1,
                                                                        axes=0)  # 均值和方差
        self.first_user_user_low_att = [first_mean_social_influ, first_var_social_influ]  # 均值和方差

        # second_low attention # 和上边一样
        self.second_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(units=1,  # 输出神经元的数量 原来为1 还是为1
                                                                                activation=tf.nn.sigmoid,  # sigmoid激活函数
                                                                                name='second_low_att_SN_layer1')
        self.second_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(units=1,  # 输出神经元的数量 原来为1 还是为1
                                                                                activation=tf.nn.leaky_relu,
                                                                                name='second_low_att_SN_layer2')

        self.social_neighbors_values_input2 = tf.reduce_sum(
            tf.math.exp(
                self.second_low_att_layer_for_social_neighbors_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)
                        ), [-1, 1]
                    )
                )
            ), 1
        )

        self.social_neighbors_values_input3 = tf.Variable(
            tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01)
        )  # 正态分布 259014
        self.social_neighbors_num_input = 1.0 / np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],
                                                           [-1, 1])  # user' friends 17237

        # ----------------------
        # 2. user-item interest graph node attention initialization # user-item 图节点attention初始化
        self.first_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1,
                                                                        activation=tf.nn.sigmoid,
                                                                        name='first_low_att_UI_layer1')
        self.first_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1,
                                                                        activation=tf.nn.leaky_relu,
                                                                        name='first_low_att_UI_layer2')

        self.user_item_sparsity_dict = data_dict['USER_ITEM_SPARSITY_DICT']
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']
        self.consumed_items_values_input1 = tf.reduce_sum(
            tf.math.exp(
                self.first_low_att_layer_for_user_item_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal(
                                [len(self.consumed_items_indices_input)], stddev=low_att_std
                            )
                        ), [-1, 1]
                    )
                )
            ), 1
        )  # 185869

        # 均值、方差
        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.consumed_items_values_input1, axes=0)
        self.first_user_item_low_att = [first_mean_social_influ, first_var_social_influ]

        self.second_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1,
                                                                         activation=tf.nn.sigmoid,
                                                                         name='second_low_att_UI_layer1')
        self.second_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1,
                                                                         activation=tf.nn.leaky_relu,
                                                                         name='second_low_att_UI_layer2')
        self.consumed_items_values_input2 = tf.reduce_sum(
            tf.math.exp(
                self.second_low_att_layer_for_user_item_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal(
                                [len(self.consumed_items_indices_input)], stddev=1.0
                            )
                        ), [-1, 1]
                    )
                )
            ), 1
        )

        self.consumed_items_values_input3 = tf.Variable(
            tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01)
        )  # 正态分布 185869
        self.consumed_items_num_input = 1.0 / np.reshape(data_dict['CONSUMED_ITEMS_NUM_INPUT'], [-1, 1])  # 17237个用户

        # ----------------------
        # 3. item-user graph node attention initialization # item-user 图节点注意力初始化
        self.first_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1,
                                                                        activation=tf.nn.sigmoid,
                                                                        name='first_low_att_IU_layer1')
        self.first_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1,
                                                                        activation=tf.nn.leaky_relu,
                                                                        name='first_low_att_IU_layer2')

        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']
        self.item_customer_values_input1 = tf.reduce_sum(
            tf.math.exp(
                self.first_low_att_layer_for_item_user_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal(
                                [len(self.item_customer_indices_input)], stddev=low_att_std)
                        ), [-1, 1]
                    )
                )
            ), 1
        )

        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.item_customer_values_input1, axes=0)
        self.first_item_user_low_att = [first_mean_social_influ, first_var_social_influ]

        self.second_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1,
                                                                         activation=tf.nn.sigmoid,
                                                                         name='second_low_att_IU_layer1')
        self.second_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1,
                                                                         activation=tf.nn.leaky_relu,
                                                                         name='second_low_att_IU_layer2')
        self.item_customer_values_input2 = tf.reduce_sum(
            tf.math.exp(
                self.second_low_att_layer_for_item_user_layer1(
                    tf.reshape(
                        tf.Variable(
                            tf.random_normal(
                                [len(self.item_customer_indices_input)], stddev=0.01
                            )
                        ), [-1, 1]
                    )
                )
            ), 1)

        self.item_customer_values_input3 = tf.Variable(
            tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)
        )  # 正态分布 185869
        self.item_customer_num_input = 1.0 / np.reshape(data_dict['ITEM_CUSTOMER_NUM_INPUT'], [-1, 1])

        # ----------------------
        # 4. prepare the shape of sparse matrice # 准备稀疏矩阵的形状
        self.social_neighbors_dense_shape = np.array([self.conf.num_users,
                                                      self.conf.num_users]).astype(np.int64)  # [17237, 17237]
        self.consumed_items_dense_shape = np.array([self.conf.num_users,
                                                    self.conf.num_items]).astype(np.int64)  # [17237, 38342]
        self.item_customer_dense_shape = np.array([self.conf.num_items,
                                                   self.conf.num_users]).astype(np.int64)  # [38342, 17237]

        ########  二、Rough Graph Attention initialization ######## # 粗糙图attention初始化
        """
        delete all
        """
        # ----------------------
        # 1. User part

        # ----------------------
        # 2. Item part


        ######## 三、Generate Sparse Matrices with/without attention # with/without 生成稀疏矩阵 #########
        # ----------------------
        # Frist Layer
        # (1) user-user
        self.social_neighbors_sparse_matrix_avg = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input,  # [1.0 / len(friends)]
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input1,  # shape=(259014,) 得到一维的向量
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )
        # (2) user-item
        self.consumed_items_sparse_matrix_avg = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input,  # [1.0 / len(consumed_items_dict[u]]  185869
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input1,  # shape=(185869,) 得到一维的向量
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )
        # (3) item-user
        self.item_customer_sparse_matrix_avg = tf.SparseTensor(
            indices=self.item_customer_indices_input,  # [(item_id, user_id)] shape=(185869, 2)
            values=self.item_customer_values_input,  # [1.0 / len(item_customer_dict[i]] 185869
            dense_shape=self.item_customer_dense_shape  # [38342, 17237]
        )
        self.first_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices=self.item_customer_indices_input,  # [(item_id, user_id)] shape=(185869, 2)
            values=self.item_customer_values_input1,  # shape=(185869,) 得到一维的向量
            dense_shape=self.item_customer_dense_shape  # [38342, 17237]
        )

        # 对第一层稀疏矩阵进行softmax
        self.first_social_neighbors_low_level_att_matrix = tf.sparse.softmax(
            self.first_layer_social_neighbors_sparse_matrix
        )
        self.first_consumed_items_low_level_att_matrix = tf.sparse.softmax(
            self.first_layer_consumed_items_sparse_matrix
        )
        self.first_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(
            self.first_layer_item_customer_sparse_matrix
        )

        # ----------------------
        # Second layer
        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input2,  # shape=(259014,) 得到一维的向量
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )
        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input2,  # shape=(185869,) 得到一维的向量
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )
        self.second_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices=self.item_customer_indices_input,  # [(item_id, user_id)] shape=(185869, 2)
            values=self.item_customer_values_input2,  # shape=(185869,) 得到一维的向量
            dense_shape=self.item_customer_dense_shape  # [38342, 17237]
        )
        # 对第二层稀疏矩阵进行softmax
        self.second_social_neighbors_low_level_att_matrix = tf.sparse.softmax(
            self.second_layer_social_neighbors_sparse_matrix
        )
        self.second_consumed_items_low_level_att_matrix = tf.sparse.softmax(
            self.second_layer_consumed_items_sparse_matrix
        )
        self.second_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(
            self.second_layer_item_customer_sparse_matrix
        )

    def convertDistribution(self, x):
        """
        转换分布, 将数据标准化到一个较小的范围内, 所以乘以 0.1
        :param x:
        :return:
        """
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y

    # ----------------------
    # Operations for Diffusion # 扩散操作
    """
    Notes:
    - SocialNeighbors: user-user
    - ConsumedItems: user-item
    - Customer: item-user
    """

    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):
        """
        从 user-user 生成 user_embedding

        self.first_social_neighbors_low_level_att_matrix：softmax过后的社交关系稀疏矩阵 {
            index=[(user_id1, user_id2)]
            value=reduce_sum() # shape=(259014,) 得到一维的向量
            shape=(17237, 17237)
        }


        :param current_user_embedding: 用户原始 embedding # shape = (17237, 64)
        :return:
        """
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.first_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):
        """
        从 user-item 生成 user_embedding

        self.first_consumed_items_low_level_att_matrix：softmax 过后的user-item稀疏矩阵 {
            index=[(user_id, item_id)]
            value=reduce_sum() # shape=(185869,) 得到一维的向量
            shape=(17237, 38342)
        }

        :param current_item_embedding: 物品原始 embedding shape=(38342, 64)
        :return:
        """
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.first_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer1(self, current_user_embedding):
        """
        从 item-user 生成 item_embedding

        self.first_items_users_neighborslow_level_att_matrix：过后的item-user稀疏矩阵 {
            index=[(item_id, user_id)]
            value=reduce_sum() 185869
            shape=(38342, 17237)
        }
        :param current_user_embedding:
        :return:
        """
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.first_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):
        """
        从 user-user 生成 user_embedding

        :param current_user_embedding:
        :return:
        """
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.second_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):
        """
        从 user-items 生成 user_embedding

        :param current_item_embedding:
        :return:
        """
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.second_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):
        """
        从 item-user 生成 item_embedding

        :param current_item_embedding:
        :return:
        """
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.second_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    def initializeNodes(self):
        """
        初始化图节点，定义好每个层
        :return:
        """
        self.item_input = tf.placeholder("int32", [None, 1])  # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1])  # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01),
            name='user_embedding'
        )  # shape = [17237, 64]

        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01),
            name='item_embedding'
        )  # [38342, 64]

        # 从本地npy读出来
        self.user_review_vector_matrix = tf.constant(np.load(self.conf.user_review_vector_matrix),
                                                     dtype=tf.float32)  # shape=(17237, 150)
        self.item_review_vector_matrix = tf.constant(np.load(self.conf.item_review_vector_matrix),
                                                     dtype=tf.float32)  # shape=(38342, 150)
        self.reduce_dimension_layer = tf.layers.Dense(  # 降维层
            self.conf.dimension,
            activation=tf.nn.sigmoid,
            name='reduce_dimension_layer'
        )

        ########  Fine-grained Graph Attention initialization  # 细粒度Graph Attention初始化 ########
        # ----------------------
        # User part
        # ----------------------
        # First diffusion layer 扩散层
        self.first_user_part_social_graph_att_layer1 = tf.layers.Dense(1,
                                                                       activation=tf.nn.tanh,
                                                                       name='firstGCN_UU_user_MLP_first_layer')
        self.first_user_part_social_graph_att_layer2 = tf.layers.Dense(1,
                                                                       activation=tf.nn.leaky_relu,
                                                                       name='firstGCN_UU_user_MLP_sencond_layer')
        self.first_user_part_interest_graph_att_layer1 = tf.layers.Dense(1,
                                                                         activation=tf.nn.tanh,
                                                                         name='firstGCN_UI_user_MLP_first_layer')
        self.first_user_part_interest_graph_att_layer2 = tf.layers.Dense(1,
                                                                         activation=tf.nn.leaky_relu,
                                                                         name='firstGCN_UI_user_MLP_second_layer')

        # ----------------------
        # Second diffusion layer 扩散
        self.second_user_part_social_graph_att_layer1 = tf.layers.Dense(1,
                                                                        activation=tf.nn.tanh,
                                                                        name='secondGCN_UU_user_MLP_first_layer')

        self.second_user_part_social_graph_att_layer2 = tf.layers.Dense(1,
                                                                        activation=tf.nn.leaky_relu,
                                                                        name='secondGCN_UU_user_MLP_second_layer')

        self.second_user_part_interest_graph_att_layer1 = tf.layers.Dense(1,
                                                                          activation=tf.nn.tanh,
                                                                          name='secondGCN_UI_user_MLP_first_layer')

        self.second_user_part_interest_graph_att_layer2 = tf.layers.Dense(1,
                                                                          activation=tf.nn.leaky_relu,
                                                                          name='secondGCN_UI_user_MLP_second_layer')

        # ----------------------
        # Item part
        self.first_item_part_itself_graph_att_layer1 = tf.layers.Dense(1,
                                                                       activation=tf.nn.tanh,
                                                                       name='firstGCN_IU_itemself_MLP_first_layer')
        self.first_item_part_itself_graph_att_layer2 = tf.layers.Dense(1,
                                                                       activation=tf.nn.leaky_relu,
                                                                       name='firstGCN_IU_itemself_MLP_second_layer')
        self.first_item_part_user_graph_att_layer1 = tf.layers.Dense(1,
                                                                     activation=tf.nn.tanh,
                                                                     name='firstGCN_IU_customer_MLP_first_layer')
        self.first_item_part_user_graph_att_layer2 = tf.layers.Dense(1,
                                                                     activation=tf.nn.leaky_relu,
                                                                     name='firstGCN_IU_customer_MLP_second_layer')
        self.second_item_part_itself_graph_att_layer1 = tf.layers.Dense(1,
                                                                        activation=tf.nn.tanh,
                                                                        name='secondGCN_IU_itemself_MLP_first_layer')
        self.second_item_part_itself_graph_att_layer2 = tf.layers.Dense(1,
                                                                        activation=tf.nn.leaky_relu,
                                                                        name='secondGCN_IU_itemself_MLP_second_layer')
        self.second_item_part_user_graph_att_layer1 = tf.layers.Dense(1,
                                                                      activation=tf.nn.tanh,
                                                                      name='secondGCN_IU_customer_MLP_first_layer')
        self.second_item_part_user_graph_att_layer2 = tf.layers.Dense(1,
                                                                      activation=tf.nn.leaky_relu,
                                                                      name='secondGCN_IU_customer_MLP_second_layer')

    def constructTrainGraph(self):
        """
        创建训练图 ★
        :return:
        """
        ########  Fusion Layer # 融合层 相加操作 ########

        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)  # 转换分布
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)  # 转换分布

        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(
            first_user_review_vector_matrix
        )  # 降维 shape=(17237, 64) 激活函数：sigmoid
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(
            first_item_review_vector_matrix
        )  # 降维 shape=(38342, 64) 激活函数：sigmoid

        second_user_review_vector_matrix = self.convertDistribution(
            self.user_reduce_dim_vector_matrix
        )  # 转换分布 shape=(17237, 64)
        second_item_review_vector_matrix = self.convertDistribution(
            self.item_reduce_dim_vector_matrix
        )  # 转换分布 shape=(38342, 64)

        self.fusion_item_embedding = self.item_embedding + second_item_review_vector_matrix
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix

        ######## Influence and Interest Diffusion Layer 影响力和兴趣扩散层 相乘操作 ########

        # ----------------------
        # First Layer

        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems1(
            self.fusion_item_embedding
        )  # shape=(17237, 64)
        user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors1(
            self.fusion_user_embedding
        )  # shape=(17237, 64)

        # 降维
        consumed_items_attention = tf.math.exp(
            self.first_user_part_interest_graph_att_layer2(  # leaky_relu
                self.first_user_part_interest_graph_att_layer1(  # tanh 128 -> 1
                    tf.concat(
                        [self.fusion_user_embedding, user_embedding_from_consumed_items], 1
                    )  # shape=(17237, 128)
                )
            )
        ) + 0.7  # shape=(17237, 1)

        # 降维
        social_neighbors_attention = tf.math.exp(
            self.first_user_part_social_graph_att_layer2(  # leaky_relu
                self.first_user_part_social_graph_att_layer1(  # tanh 128 -> 1
                    tf.concat(
                        [self.fusion_user_embedding, user_embedding_from_social_neighbors], 1
                    )  # shape=(17237, 128)
                )
            )
        ) + 0.3  # shape=(17237, 1)

        sum_attention = consumed_items_attention + social_neighbors_attention  # shape=(17237, 1)
        self.consumed_items_attention_1 = consumed_items_attention / sum_attention  # shape=(17237, 1)
        self.social_neighbors_attention_1 = social_neighbors_attention / sum_attention  # shape=(17237, 1)

        first_gcn_user_embedding = 1 / 2 * self.fusion_user_embedding + 1 / 2 * (
                self.consumed_items_attention_1 * user_embedding_from_consumed_items +
                self.social_neighbors_attention_1 * user_embedding_from_social_neighbors
        )  # shape=(17237, 64)

        item_itself_att = tf.math.exp(
            self.first_item_part_itself_graph_att_layer2(  # leaky_relu
                self.first_item_part_itself_graph_att_layer1(  # tanh 64 -> 1
                    self.fusion_item_embedding  # shape=(38342, 64)
                )
            )
        ) + 1.0  # shape=(38342, 1)

        item_customer_attenton = tf.math.exp(
            self.first_item_part_user_graph_att_layer2(  # leaky_relu
                self.first_item_part_user_graph_att_layer1(  # tanh
                    self.generateItemEmebddingFromCustomer1(  # shape=(38342, 64)
                        self.fusion_user_embedding  # shape=(17237, 64)
                    )
                )
            )
        ) + 1.0  # shape=(38342, 1)

        item_sum_attention = item_itself_att + item_customer_attenton  # shape=(38342, 1)

        self.item_itself_att1 = item_itself_att / item_sum_attention  # shape=(38342, 1)
        self.item_customer_attenton1 = item_customer_attenton / item_sum_attention  # shape=(38342, 1)

        first_gcn_item_embedding = self.item_itself_att1 * self.fusion_item_embedding + \
                                   self.item_customer_attenton1 * self.generateItemEmebddingFromCustomer1(
            self.fusion_user_embedding)  # shape=(38342, 64)

        # ----------------------
        # Second Layer

        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems2(
            first_gcn_item_embedding
        )  # shape=(17237, 64)
        user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors2(
            first_gcn_user_embedding
        )  # shape=(17237, 64)

        consumed_items_attention = tf.math.exp(
            self.second_user_part_interest_graph_att_layer2(  # leaky_relu
                self.second_user_part_interest_graph_att_layer1(  # tanh 128 -> 1
                    tf.concat([first_gcn_user_embedding, user_embedding_from_consumed_items], 1)
                )
            )
        ) + 0.7  # shape=(17237, 1)

        social_neighbors_attention = tf.math.exp(
            self.second_user_part_social_graph_att_layer2(  # leaky_relu
                self.second_user_part_social_graph_att_layer1(  # tanh 128 -> 1
                    tf.concat([first_gcn_user_embedding, user_embedding_from_social_neighbors], 1)
                )
            )
        ) + 0.3  # shape=(17237, 1)

        sum_attention = consumed_items_attention + social_neighbors_attention  # shape=(17237, 1)
        self.consumed_items_attention_2 = consumed_items_attention / sum_attention  # shape=(17237, 1)
        self.social_neighbors_attention_2 = social_neighbors_attention / sum_attention  # shape=(17237, 1)

        second_gcn_user_embedding = 1 / 2 * first_gcn_user_embedding + 1 / 2 * (
                self.consumed_items_attention_2 * user_embedding_from_consumed_items
                + self.social_neighbors_attention_2 * user_embedding_from_social_neighbors
        )  # shape=(17237, 64)

        item_itself_att = tf.math.exp(
            self.second_item_part_itself_graph_att_layer2(  # leaky_relu
                self.second_item_part_itself_graph_att_layer1(  # tanh 64 -> 1
                    first_gcn_item_embedding  # shape=(38342, 64)
                )
            )
        ) + 1.0  # shape=(38342, 1)

        item_customer_attenton = tf.math.exp(
            self.second_item_part_user_graph_att_layer2(  # leaky_relu
                self.second_item_part_user_graph_att_layer1(  # tanh 64 -> 1
                    self.generateItemEmebddingFromCustomer2(
                        first_gcn_user_embedding  # shape=(17237, 64)
                    )
                )
            )
        ) + 1.0  # shape=(38342, 1)

        item_sum_attention = item_itself_att + item_customer_attenton  # shape=(38342, 1)

        self.item_itself_att2 = item_itself_att / item_sum_attention  # shape=(38342, 1)
        self.item_customer_attenton2 = item_customer_attenton / item_sum_attention  # shape=(38342, 1)

        second_gcn_item_embedding = self.item_itself_att2 * first_gcn_item_embedding + \
                                    self.item_customer_attenton2 * \
                                    self.generateItemEmebddingFromCustomer2(
                                        first_gcn_user_embedding  # (17237, 64)
                                    )  # shape=(38342, 64)

        ######## Prediction Layer # 预测层 ########

        self.final_user_embedding = tf.concat([first_gcn_user_embedding,
                                               second_gcn_user_embedding,
                                               self.user_embedding,
                                               second_user_review_vector_matrix], 1)
        self.final_item_embedding = tf.concat([first_gcn_item_embedding,
                                               second_gcn_item_embedding,
                                               self.item_embedding,
                                               second_item_review_vector_matrix], 1)

        latest_user_latent = tf.gather_nd(self.final_user_embedding, self.user_input)  # shape=(?, 256)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, self.item_input)  # shape=(?, 256)

        self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)  # shape=(?, 256)
        self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))  # shape=(?, 1)

        # ----------------------
        # Optimazation 训练优化器

        self.loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        """
        保存变量
        :return:
        """
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v

        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################

    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['val'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST',
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction,
        }

        self.map_dict = map_dict
