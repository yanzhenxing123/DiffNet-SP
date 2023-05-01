"""
@Author: yanzx
@Date: 2023/5/1 15:37
@Description: 分割links数据集
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = "../data/yelp/yelp.links"

data = pd.read_csv(filepath_or_buffer=path,
                   header=None,
                   names=["user_id", "item_id", "rating_value"],
                   delimiter=r"\s+")

# 先分出20%的数据存入测试集
data_train, data_test = train_test_split(data, test_size=0.1, random_state=1234)
# 剩下的数据10%存入验证集
data_test, data_validate = train_test_split(data_test, test_size=0.2, random_state=1234)

data_train.to_csv("../data/yelp/yelp.train.links", sep='\t', index=False, header=False, mode='w')
data_test.to_csv("../data/yelp/yelp.test.links", sep='\t', index=False, header=False, mode='w')
data_validate.to_csv("../data/yelp/yelp.val.links", sep='\t', index=False, header=False, mode='w')
