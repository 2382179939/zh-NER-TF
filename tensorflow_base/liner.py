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
import keras


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
    当label的编码方式为顺序编码，图像多分类问题
    :param data_path: 数据集的路径
    :return: null
    """
    msg = '训练完毕'
    try:
        fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # 第一次运行会非常慢，从国外的网站下载数据
        (train_image, train_label), (test_image, test_label) = fashion_mnist
        # first = train_image[0]
        # print(np.max(train_image[0]))
        # print(np.min(train_image[0]))
        # plt.imshow(train_image[0])  # 画图像
        # plt.show()
        train_image = train_image / 255  # 把所有的数字搞到0-1之间
        test_image = test_image / 255
        model = tf.keras.Sequential()  # 数据准备好以后开始建模型
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 使得二维矩阵变成一个一维向量
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        history = model.fit(train_image, train_label, epochs=5)
        evaluate_result = model.evaluate(test_image, test_label)  # 测试数据集来评估我们的模型
        print(evaluate_result)
        plt.plot(history.epoch, history.history['loss'])  # 训练次数和损失值的函数曲线图
        plt.show()
        plt.plot(history.epoch, history.history['acc'])  # 训练次数和精确率的函数去曲线图
        plt.show()
        return msg
    except ImportError:
        return '程序运行有误，导致模型训练有误'


def one_hot_encode():
    """
    当label的方式为独热编码时的训练方式，图像多分类问题
    :return: null
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # 第一次运行会非常慢，从国外的网站下载数据
    (train_image, train_label), (test_image, test_label) = fashion_mnist
    train_image = train_image / 255  # 把所有的数字搞到0-1之间
    test_image = test_image / 255
    # label采取one编码，损失函数需要改变为categorical_crossentropy
    train_label_one_hot = tf.keras.utils.to_categorical(train_label)
    test_label_one_hot = tf.keras.utils.to_categorical(test_label)
    model = tf.keras.Sequential()  # 数据准备好以后开始建模型
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 使得二维矩阵变成一个一维向量
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # 这里可以添加多层，是的神经网络的拟合能力更强
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    # 可以实例化一个Adam优化器，指定学习速率指定为0.01
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_image, train_label_one_hot, epochs=100)
    plt.plot(history.epoch, history.history.get('loss'))
    plt.show()
    plt.plot(history.epoch, history.history.get('acc'))
    plt.show()
    predict = model.predict(test_image)
    if np.argmax(predict[0]) == np.argmax(test_label_one_hot[0]):
        print('预测正确，并且该图片的分类是：%d' % np.argmax(predict[0]))
    else:
        print('预测的结果为：%d' % np.argmax(predict[0]))
        print('正确的结果为：%d' % np.argmax(test_label_one_hot[0]))


def over_fit():
    """
    过拟合问题
    :return: 0
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # 第一次运行会非常慢，从国外的网站下载数据
    (train_image, train_label), (test_image, test_label) = fashion_mnist
    train_image = train_image / 255  # 把所有的数字搞到0-1之间
    test_image = test_image / 255
    # label采取one编码，损失函数需要改变为categorical_crossentropy
    train_label_one_hot = tf.keras.utils.to_categorical(train_label)
    test_label_one_hot = tf.keras.utils.to_categorical(test_label)
    model = tf.keras.Sequential()  # 数据准备好以后开始建模型
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 使得二维矩阵变成一个一维向量
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # 这里可以添加多层，是的神经网络的拟合能力更强
    model.add(tf.keras.layers.Dropout(rate=0.5))  # 丢弃百分之五十的单元数
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_image, train_label_one_hot, epochs=5, validation_data=(test_image, test_label_one_hot))
    # 如何看待过拟合和欠拟合的现象
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.show()
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.show()


def reduce_network_capacity():
    """
    减少网络的容量，
    :return: 0
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()  # 第一次运行会非常慢，从国外的网站下载数据
    (train_image, train_label), (test_image, test_label) = fashion_mnist
    train_image = train_image / 255  # 把所有的数字搞到0-1之间
    test_image = test_image / 255
    # label采取one编码，损失函数需要改变为categorical_crossentropy
    train_label_one_hot = tf.keras.utils.to_categorical(train_label)
    test_label_one_hot = tf.keras.utils.to_categorical(test_label)
    model = tf.keras.Sequential()  # 数据准备好以后开始建模型
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 使得二维矩阵变成一个一维向量
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_image, train_label_one_hot, epochs=10, validation_data=(test_image, test_label_one_hot))
    plt.plot(history.epoch, history.history['acc'], label='acc')
    plt.plot(history.epoch, history.history['val_acc'], label='acc')
    plt.show()
    return 0


def function_API():
    """
    函数式API
    :return: 0
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.00
    input = keras.Input(shape=(28, 28))
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['acc'], label='accuracy')
    plt.show()


if __name__ == '__main__':
    # test_gpu()
    # prepare_data()
    # liner_regression(data_path='./dataset/sample_data.csv')
    # multilayer_perceptron(data_path='./dataset/Advertising.csv')
    # logistic_regression(data_path='./dataset/credit-a.csv')
    # image_classification()
    # one_hot_encode()
    # over_fit()
    # reduce_network_capacity()
    function_API()
