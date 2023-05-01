from __future__ import division
import tensorflow as tf
import numpy as np
from ipdb import set_trace


class MGNN():
    """
    互惠神经网络
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
        # 1. user-user social network node attention initialization
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']  # 259014 [(u_id1, u_id2)]

        self.first_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,  # sigmoid激活函数
            name='first_low_att_SN_layer1'
        )
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

        # second_low attention
        self.second_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,  # sigmoid激活函数
            name='second_low_att_SN_layer1'
        )
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
        )  # shape=(259014,)

        # ----------------------
        # 2. user-item interest graph node attention initialization
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']  # 185869 [(user_id, item_id)]

        self.first_low_att_layer_for_user_item_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            name='first_low_att_UI_layer1'
        )
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
        )  # shape=(185869,)

        self.second_low_att_layer_for_user_item_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            name='second_low_att_UI_layer1'
        )
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
        )  # shape=(185869,)

        # ----------------------
        # 3. item-user graph node attention initialization
        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']  # 185869 [(item_id, user_id)]

        self.first_low_att_layer_for_item_user_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            name='first_low_att_IU_layer1'
        )
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

        self.second_low_att_layer_for_item_user_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            name='second_low_att_IU_layer1'
        )
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
            ), 1
        )

        # ----------------------
        # 4. prepare the shape of sparse matrice # 准备稀疏矩阵的形状
        self.social_neighbors_dense_shape = np.array(
            [self.conf.num_users, self.conf.num_users]
        ).astype(np.int64)  # [17237, 17237]

        self.consumed_items_dense_shape = np.array(
            [self.conf.num_users, self.conf.num_items]
        ).astype(np.int64)  # [17237, 38342]

        self.item_customer_dense_shape = np.array(
            [self.conf.num_items, self.conf.num_users]
        ).astype(np.int64)  # [38342, 17237]

        ######## 三、Generate Sparse Matrices with/without attention # with/without 生成稀疏矩阵 #########
        # ----------------------
        # Frist Layer
        # (1) user-user
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input1,  # shape=(259014,) 得到一维的向量
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )

        # (2) user-item
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input1,  # shape=(185869,) 得到一维的向量
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )
        # (3) item-user
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
        # (1) user-user
        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input2,  # shape=(259014,) 得到一维的向量
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )

        # (2) user-item
        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input2,  # shape=(185869,) 得到一维的向量
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )

        # (3) item-user
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

        # MGNN
        # (1) user-user
        self.user_user_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,  # [(user_id1, user_id2)]，二者为朋友 shape=(259014, 2)
            values=self.social_neighbors_values_input1,  # shape=(259014,) 得到一维的向量
            dense_shape=self.social_neighbors_dense_shape  # [17237, 17237]
        )

        # (2) user-item
        self.user_item_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,  # [(user_id, item_id)] shape=(185869, 2)
            values=self.consumed_items_values_input1,  # shape=(185869,) 得到一维的向量
            dense_shape=self.consumed_items_dense_shape  # [17237, 38342]
        )
        # (3) item-user
        self.item_user_sparse_matrix = tf.SparseTensor(
            indices=self.item_customer_indices_input,  # [(item_id, user_id)] shape=(185869, 2)
            values=self.item_customer_values_input1,  # shape=(185869,) 得到一维的向量
            dense_shape=self.item_customer_dense_shape  # [38342, 17237]
        )

        # 对第一层稀疏矩阵进行softmax
        self.user_user_attention_matrix = tf.sparse.softmax(
            self.user_user_sparse_matrix
        )
        self.user_item_attention_matrix = tf.sparse.softmax(
            self.user_item_sparse_matrix
        )
        self.item_user_attention_matrix = tf.sparse.softmax(
            self.item_user_sparse_matrix
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

    def get_item_influence_embedding(self, current_user_embedding):
        """
        1.1 从 user-item x item-user 生成 user-embedding
        :param current_user_embedding:
        :return:
        """
        self.layer1_1 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        self.layer1_2 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        item_influence_embedding = self.layer1_2(
            tf.sparse_tensor_dense_matmul(
                self.user_item_attention_matrix, self.layer1_1(
                    tf.sparse_tensor_dense_matmul(
                        self.item_user_attention_matrix, current_user_embedding
                    )
                )
            )
        )
        return item_influence_embedding

    def get_social_item_embedding(self, current_item_embedding):
        """
        1.2 从 user-user x user-item 生成 user-embedding
        :param current_item_embedding:
        :return:
        """

        self.layer1_3 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        self.layer1_4 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        social_item_embedding = self.layer1_4(
            tf.sparse_tensor_dense_matmul(
                self.user_user_attention_matrix, self.layer1_3(
                    tf.sparse_tensor_dense_matmul(
                        self.user_item_attention_matrix, current_item_embedding
                    )
                )
            )
        )

        return social_item_embedding

    def get_consumption_preference_embedding(self, item_influence_embedding, social_item_embedding):
        """
        1.3 获得 consumption_preference_embedding 用户消费喜好
        :param item_influence_embedding:
        :param social_item_embedding:
        :return:
        """
        self.layer1_5 = tf.layers.Dense(
            units=self.conf.dimension,  # 128 -> 64
            activation=tf.nn.leaky_relu,
            name='reduce_dimension_layer'
        )

        consumption_preference_embedding = self.layer1_5(
            tf.concat(
                [item_influence_embedding, social_item_embedding], 1
            )
        )
        return consumption_preference_embedding

    def get_social_preference_embedding(self, current_user_embedding):
        """
        2.1 GCN 从 user-user 生成 user_embedding
        :param current_user_embedding:
        :return:
        """

        self.layer2_1 = tf.layers.Dense(
            units=self.conf.dimension,
            activation=tf.nn.leaky_relu,
        )

        social_preference_embedding = self.layer2_1(
            tf.sparse_tensor_dense_matmul(
                self.user_user_attention_matrix, current_user_embedding
            )
        )
        return social_preference_embedding

    def get_preference_embedding(self, consumption_preference_embedding, current_user_embedding):
        """
        3.1 获取 preference_embedding 作为互惠层的输入
        :param consumption_preference_embedding:
        :param current_user_embedding:
        :return:
        """

        self.layer3_1 = tf.layers.Dense(
            units=self.conf.dimension * 2,  # 128
            activation=tf.nn.leaky_relu,
        )
        preference_embedding = self.layer3_1(
            tf.concat(
                [consumption_preference_embedding, current_user_embedding], 1
            )
        )
        return preference_embedding

    def get_social_embedding(self, social_preference_embedding, current_user_embedding):
        """
        3.2 获取 social_embedding 作为互惠层的输入
        :param social_preference_embedding:
        :param current_user_embedding:
        :return:
        """

        self.layer3_2 = tf.layers.Dense(
            units=self.conf.dimension * 2,  # 128
            activation=tf.nn.leaky_relu,
        )
        social_embedding = self.layer3_2(
            tf.concat(
                [social_preference_embedding, current_user_embedding], 1
            )
        )
        return social_embedding

    def get_mutual_embedding(self, preference_embedding, social_embedding):
        """
        3.3 preference_embedding 和 social_embedding 点积(Dot)获得 mutual_embedding
        :param preference_embedding:
        :param social_preference_embedding:
        :return:
        """

        mutual_embedding = tf.multiply(preference_embedding, social_embedding)  # 128
        return mutual_embedding

    def get_mutual_preference_embedding(self, preference_embedding, mutual_embedding):
        """
        3.4 preference_embedding 和 mutual_embedding concat 获得 mutual_preference_embedding
        :param preference_embedding:
        :param mutual_embedding:
        :return:
        """
        self.layer3_3 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        mutual_preference_embedding = self.layer3_3(
            tf.concat(
                [preference_embedding, mutual_embedding], 1
            )  # 192
        )
        return mutual_preference_embedding

    def get_mutual_social_embedding(self, social_embedding, mutual_embedding):
        """
        3.5 preference_embedding 和 mutual_embedding concat 获得 mutual_social_embedding
        :param preference_embedding:
        :param mutual_embedding:
        :return:
        """
        self.layer3_4 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )
        mutual_social_embedding = self.layer3_4(
            tf.concat(
                [social_embedding, mutual_embedding], 1
            )  # 192
        )
        return mutual_social_embedding

    def initializeNodes(self):
        """
        初始化图节点，定义好每个层
        :return:
        """
        self.item_input = tf.placeholder("int32", [None, 1])  # item_list: [item_id, ....]
        self.user_input = tf.placeholder("int32", [None, 1])  # user_list: [0...0, 2, 2, 3, 3]
        self.labels_input = tf.placeholder("float32", [None, 1])  # labels_list: [0 or 1]


        self.social_user_input = tf.placeholder("int32", [None, 1])  # item_list: [item_id, ....]
        self.social_friend_input = tf.placeholder("int32", [None, 1])  # user_list: [0...0, 2, 2, 3, 3]
        self.social_labels_input = tf.placeholder("float32", [None, 1])  # labels_list: [0 or 1]



        ######## 1. 嵌入层 ########
        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01),
            name='user_embedding'
        )  # shape = [17237, 64]

        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01),
            name='item_embedding'
        )  # [38342, 64]

        # 从本地npy读出来
        self.user_review_vector_matrix = tf.constant(
            np.load(self.conf.user_review_vector_matrix), dtype=tf.float32
        )  # shape=(17237, 150)

        self.item_review_vector_matrix = tf.constant(
            np.load(self.conf.item_review_vector_matrix), dtype=tf.float32
        )  # shape=(38342, 150)

        self.reduce_dimension_layer = tf.layers.Dense(  # 降维层 -> 64
            units=self.conf.dimension,  # 64
            activation=tf.nn.sigmoid,
            name='reduce_dimension_layer'
        )

        ########  Fine-grained Graph Attention initialization  # 细粒度Graph Attention初始化 ########
        # ----------------------
        # User part
        # ----------------------
        # First diffusion layer 扩散层
        self.first_user_part_social_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='firstGCN_UU_user_MLP_first_layer'
        )
        self.first_user_part_social_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='firstGCN_UU_user_MLP_sencond_layer'
        )
        self.first_user_part_interest_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='firstGCN_UI_user_MLP_first_layer'
        )
        self.first_user_part_interest_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='firstGCN_UI_user_MLP_second_layer'
        )

        # ----------------------
        # Second diffusion layer 扩散
        self.second_user_part_social_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='secondGCN_UU_user_MLP_first_layer'
        )

        self.second_user_part_social_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='secondGCN_UU_user_MLP_second_layer'
        )

        self.second_user_part_interest_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='secondGCN_UI_user_MLP_first_layer'
        )

        self.second_user_part_interest_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='secondGCN_UI_user_MLP_second_layer'
        )

        # ----------------------
        # Item part
        self.first_item_part_itself_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='firstGCN_IU_itemself_MLP_first_layer'
        )
        self.first_item_part_itself_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='firstGCN_IU_itemself_MLP_second_layer'
        )
        self.first_item_part_user_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='firstGCN_IU_customer_MLP_first_layer'
        )
        self.first_item_part_user_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='firstGCN_IU_customer_MLP_second_layer'
        )
        self.second_item_part_itself_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='secondGCN_IU_itemself_MLP_first_layer'
        )
        self.second_item_part_itself_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='secondGCN_IU_itemself_MLP_second_layer'
        )
        self.second_item_part_user_graph_att_layer1 = tf.layers.Dense(
            units=1,
            activation=tf.nn.tanh,
            name='secondGCN_IU_customer_MLP_first_layer'
        )
        self.second_item_part_user_graph_att_layer2 = tf.layers.Dense(
            units=1,
            activation=tf.nn.leaky_relu,
            name='secondGCN_IU_customer_MLP_second_layer'
        )

        # MGNN
        self.reduce_dimension_layer = tf.layers.Dense(  # 降维层 -> 64
            units=self.conf.dimension,  # 64
            activation=tf.nn.sigmoid,
            name='reduce_dimension_layer'
        )

    def get_user_and_item_embedding(self):
        """
        融合层，先拿到 user_embedding 和 item_embedding
        :return:
        """
        # 转换分布 -> 降维 -> 转换分布
        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)  # shape=(17237, 150)
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)  # shape=(38342, 150)

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

        # 加法融合 item_embedding 和 user_embedding 都是正态分布的随机数
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix  # shape=(17237, 64)
        self.fusion_item_embedding = self.item_embedding + second_item_review_vector_matrix  # shape=(38342, 64)

        return self.fusion_user_embedding, self.fusion_item_embedding

    def constructTrainGraph(self):
        """
        创建训练图 ★
        :return:
        """

        self.current_user_embedding, self.current_item_embedding = self.get_user_and_item_embedding()
        # 1. 空间层 5层神经网络
        self.item_influence_embedding = self.get_item_influence_embedding(self.current_user_embedding)  # (17237, 64)
        self.social_item_embedding = self.get_social_item_embedding(self.current_item_embedding)  # (17237, 64)
        self.consumption_preference_embedding = self.get_consumption_preference_embedding(
            self.item_influence_embedding, self.social_item_embedding
        )  # shape=(17237, 64)

        # 2. 光谱层 GCN 一层神经网络
        self.social_preference_embedding = self.get_social_preference_embedding(
            self.current_user_embedding
        )  # (17237, 64)

        # 3. 互惠层
        self.prefenrence_embedding = self.get_preference_embedding(
            self.consumption_preference_embedding, self.current_user_embedding
        )  # concat 128
        self.social_embedding = self.get_social_embedding(
            self.social_preference_embedding, self.current_user_embedding
        )  # concat 128

        self.mutual_embedding = self.get_mutual_embedding(
            self.prefenrence_embedding, self.social_embedding
        )  # dot 128

        self.mutual_preference_embedding = self.get_mutual_preference_embedding(
            self.prefenrence_embedding, self.mutual_embedding
        )  # concat 64
        self.mutual_social_embedding = self.get_mutual_social_embedding(
            self.social_embedding, self.mutual_embedding
        )  # concat 64

        # 4. 预测层

        self.layer4_1 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )  # mutual_preference_embedding

        self.layer4_2 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )  # mutual_social_embedding

        self.layer4_3 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )  # current_user_embedding

        self.layer4_4 = tf.layers.Dense(
            units=self.conf.dimension,  # 64
            activation=tf.nn.leaky_relu,
        )  # current_item_embedding

        self.mutual_preference_embedding = self.layer4_1(self.mutual_preference_embedding)
        self.current_item_embedding = self.layer4_4(self.current_item_embedding)

        self.mutual_social_embedding = self.layer4_2(self.mutual_social_embedding)
        self.current_user_embedding = self.layer4_3(self.current_user_embedding)

        latest_user_latent = tf.gather_nd(
            self.mutual_preference_embedding, self.user_input
        )  # shape=(?, 64) user_input'shape=(?, 1)
        latest_item_latent = tf.gather_nd(
            self.current_item_embedding, self.item_input
        )  # shape=(?, 64) item_input'shape=(?, 1)

        latest_user_latent1 = tf.gather_nd(
            self.mutual_social_embedding, self.social_user_input
        )  # shape=(?, 64) user_input'shape=(?, 1)

        latest_user_latent2 = tf.gather_nd(
            self.current_user_embedding, self.social_friend_input
        )  # shape=(?, 64) item_input'shape=(?, 1)

        self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)  # shape=(?, 64)
        self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))  # shape=(?, 1)

        self.predict_social_vector = tf.multiply(latest_user_latent1, latest_user_latent2)  # shape=(?, 64)
        self.social_prediction = tf.sigmoid(tf.reduce_sum(self.predict_social_vector, 1, keepdims=True))  # shape=(?, 1)

        # ----------------------
        # Optimazation 训练优化器

        self.loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.social_loss = tf.nn.l2_loss(self.social_labels_input - self.social_prediction)

        self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss + self.social_loss)
        # self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)

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
            self.labels_input: 'LABEL_LIST',
            self.social_user_input: 'SOCIAL_USER_LIST',
            self.social_friend_input: 'SOCIAL_FRIEND_LIST',
            self.social_labels_input: 'SOCIAL_LABEL_LIST',
        }

        map_dict['val'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST',
            self.social_user_input: 'SOCIAL_USER_LIST',
            self.social_friend_input: 'SOCIAL_FRIEND_LIST',
            self.social_labels_input: 'SOCIAL_LABEL_LIST',
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST',
            self.social_user_input: 'SOCIAL_USER_LIST',
            self.social_friend_input: 'SOCIAL_FRIEND_LIST',
            self.social_labels_input: 'SOCIAL_LABEL_LIST',
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
