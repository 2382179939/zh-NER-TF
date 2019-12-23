# -*- coding: utf-8 -*-
'''
@Author     : Liang Jianming
@Date       : 2019/12/23
@File       : train_word_embeddings.py
@SoftWare   : PyCharm
'''

# 基于词与词构造共现矩阵，提取词向量
import collections
import os

parent_path = os.path.dirname(os.path.abspath(__file__))
file_path = "..\\pics\\demo.txt"
model_path = "..\\pics\\demo.txt"
min_count = 5  # 最低词频
word_demension = 200
window_size = 5  # 窗口大小


def load_data(file_path=file_path):
    dataset = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip().split(',')
        dataset.append([word for word in line[1].split(' ') if 'nbsp' not in word and len(word) < 1])
    return dataset


# 统计总词数
def build_wrod_dict(dataset):
    words = []
    for data in dataset:
        words.extend(data)
    reserved_words = [item for item in collections.Counter(words).most_common() if item[1] > min_count]
    word_dict = {item[0]: item[1] for item in reserved_words}
    return word_dict


# 构造上下文窗口
def build_word2word_dict(dataset):
    word2word_dict = {}
    for data_idx, data in enumerate(dataset):
        contexts = []
        for index in range(len(data)):
            if index < window_size:
                left = data[:index]
            else:
                left = data[index - window_size:index]
            if index + window_size > len(data):
                right = data[index + 1:]
            else:
                right = data[index + 1: index + window_size + 1]
            context = left + [data[index]] + right  # 得到了一句话中的上下文的窗口
            for word in context:
                if word not in word2word_dict:
                    word2word_dict[word] = {}
                else:
                    for co_word in context:
                        if co_word != word:
                            word2word_dict[word][co_word] = 1
                        else:
                            word2word_dict[word][co_word] += 1
    return word2word_dict


# 构造共现矩阵
def build_word2word_matrix():
    word2word_dict = build_word2word_dict()
    word_dict = build_wrod_dict()
    word_list = list(word_dict)  # 这个只会构造出一个word的key
    word2word_matrix = []
    count = 0
    for word1 in word_list:
        count += 1
        temp = []
        sumtf = sum(word2word_dict[word1].values())
        for word2 in word_list:
            weight = word2word_dict[word2].get(word2, 0) / sumtf
            temp.append(weight)
        word2word_matrix.append(temp)
    return word2word_matrix


if __name__ == '__main__':
    dataset = load_data()
    build_word2word_dict(dataset)
    build_word2word_dict(dataset)
    build_word2word_matrix