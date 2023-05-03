# DiffNet
基于社交影响力与用户兴趣扩散的图神经网络推荐算法设计与实现

## 数据集
### yelp
#### rating
评分
+ all: 207945
+ train: 185869 # 90%
+ test: 18579 # 8%
+ val: 3497 # 2%

#### links
社交
+ all: 143765
+ train: 129388 # 90%
+ test: 11501 # 8%
+ val: 2876 # 2%

## 运行
`python=3.6 tensorflow=1.12.0`

+ 安装依赖：`pip install -r requirements.txt`
+ 执行：
    + `python entry.py --data_name=flickr --model_name=diffnetplus` # 模型：diffnetplus 数据集：flickr
    + `python entry.py --data_name=yelp --model_name=MGNN` # 模型：mgnn 数据集：yelp
