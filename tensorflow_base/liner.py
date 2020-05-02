# -*- coding: utf-8 -*-

"""
@Author     : Liang Jianming
@Date       : 2020/4/30
@File       : liner.py
@SoftWare   : PyCharm
"""

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def test_gpu():
    """
    测试tensorflow的版本信息和是否可用
    :return: null
    """
    print(tf.test.is_gpu_available())
    print('TensorFlow Version：{}'.format(tf.__version__))


def prepare_data():
    """
    读取txt文件内容写到csv文件
    :return: null
    """
    head_list = list()  # 截取表头内容
    data_list = list()  # 准备好数据
    with open(file='./sample_data.txt', mode='rb') as file:
        lines = file.readlines()
        for (i, line) in enumerate(lines):
            if i == 0:
                head_list.append(str(line, encoding='utf-8').split(',')[0])
                head_list.append(str(line, encoding='utf-8').split(',')[1])
                head_list.append(str(line, encoding='utf-8').split(',')[2].strip())
            else:
                temp_dic = {head_list[0]: str(line, encoding='utf-8').split(',')[0],
                            head_list[1]: str(line, encoding='utf-8').split(',')[1],
                            head_list[2]: str(line, encoding='utf-8').strip().split(',')[2].strip()}
                data_list.append(temp_dic)
        file.close()
    # 写数据到csv文件
    with open('./sample_data.csv', 'w', encoding='utf-8') as file:
        df = pd.DataFrame(data=data_list, columns=head_list)
        df.to_csv('./sample_data.csv', index=0)
        file.close()


def liner_regression(data_path):
    """
    线性回归问题
    :param data_path: 数据集路径
    :return: null
    """
    # 以下是一个典型的线性回归问题，其主要的是应用到机器学习的梯度下降算法
    data = pd.read_csv(data_path)
    x = list(data.Education)
    y = data.Income
    print(data.Education)
    graph = plt.scatter(x=data.Education, y=data.Income)
    plt.draw()
    print(graph)
    model = tf.keras.Sequential()  # 建立模型
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))  # 建立y=ax+b函数，往模型添加层,a和b是需要训练的两个参数
    model.summary()  # 训练参数
    model.compile(optimizer='adam', loss='mse')  # 决定优化器和损失函数
    model.fit(data.Education, data.Income, epochs=5000)  # 开始训练
    print(model.predict(data.Education))  # 对原样本进行预测
    print(model.predict(pd.Series([10])))  # 对10进行预测
    plt.plot(list(data.Education), list(data.Income))
    plt.show()  # 远程回调远端linux进行绘图，但是目还不能绘图,原因我的代码在linux服务器上边跑的


def multilayer_perceptron(data_path):
    """
    多层感知器
    :param data_path: 加载广告数据集的路径
    :return: null
    """
    data = pd.read_csv(data_path)
    x = data.iloc[:, 1:-1]  # 取第一列到倒数第二列
    y = data.iloc[:, -1]  # 取倒数第一列
    print(x)
    print(y)
    plt.plot(data.TV, data.sales)
    plt.show()
    test_data = data.iloc[:10, 1:-1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()  # 查看训练参数
    model.compile(optimizer='adam', loss='mse')  # 确定损失函数和优化器
    model.fit(x, y, epochs=1000)  # 迭代训练1000次
    print(model.predict(test_data))  # 预测值
    print(data.iloc[:10, -1])  # 实际值


def logistic_regression(data_path):
    """
    二分类问题
    :param data_path: 数据集的路径
    :return: null
    """
    data = pd.read_csv(data_path, header=None)
    # data_head = data.head()
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1].replace(-1, 0)
    print(data.iloc[:, -1].value_counts())  # 统计一下最后一列，相当于groupby
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 激活函数使用sigmoid层，把输出结果映射到0-1的概率值
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x, y, epochs=100)
    print(history.history.keys())
    plt.plot(history.epoch, history.history['loss'])
    plt.show()
    plt.plot(history.epoch, history.history['acc'])
    plt.show()


def image_classification(data_path=None):
    """
    图像多分类问题
    :param data_path: 数据集的路径
    :return: null
    """
    msg = '训练完毕'
    try:
        fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # 第一次运行会非常慢，从国外的网站下载数据
        (train_image, train_label), (test_image, train_label) = fashion_mnist
        print(train_image.shape)
        print(msg)
        return msg
    except ImportError:
        return '程序运行有误，导致模型训练有误'


if __name__ == '__main__':
    # test_gpu()
    # prepare_data()
    # liner_regression(data_path='./dataset/sample_data.csv')
    # multilayer_perceptron(data_path='./dataset/Advertising.csv')
    # logistic_regression(data_path='./dataset/credit-a.csv')
    image_classification()
