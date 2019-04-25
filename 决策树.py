import numpy as np
# 决策树构建之前需要做好特征选择
# 特征选择在于选取对训练数据具有分类能力的特征
# 信息增益  在划分数据集之后信息发生的变化称为信息增益  获取信息增益最高的特征就是最好的选择
# 在可以评测哪个数据划分方式是最好的数据划分之前，我们必须学习如何计算信息增益。
# 集合信息的度量方式成为香农熵或者简称为熵(entropy)，这个名字来源于信息论之父克劳德·香农
# 熵定义为信息的期望值在信息论和概率统计中 熵是表示随机变量不确定性的度量
from math import log
def createDateSet():
    dataset = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']  # 分类属性
    return dataset,labels        # 返回数据集和分类属性

def calcShannonEnt(dataset):
    numEntires = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntires
        shannonEnt-=prob * log(prob,2)
    return shannonEnt
if __name__ == '__main__':
    dataset,features = createDateSet()
    print(dataset)
    print(calcShannonEnt(dataset))
b = open('1232','a+')
print('a',file=b,flush=True)