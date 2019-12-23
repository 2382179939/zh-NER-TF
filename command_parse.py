# -*- coding: utf-8 -*-
import argparse
import pickle


def parse_tool():
    """
    演示argparse的使用
    :return:null
    """
    parse = argparse.ArgumentParser(description='demo of argsparse')
    parse.add_argument('-n', '--name', default='li', help='姓名')
    parse.add_argument('-y', '--year', default='20', help='年龄')
    args = parse.parse_args()
    print(args)
    name = args.name
    year = args.year
    print('hello {} {}'.format(name, year))


def read_pkl_file():
    with open('./data_path/word2id.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)
        f.close()


if __name__ == '__main__':
    # parse_tool()
    read_pkl_file()
