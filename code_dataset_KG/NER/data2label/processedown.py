import random
import os
import re


def split_dataset(input_file, train_ratio):
    # 读取文本文件中的数据
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()

        # 去除空行和第一列为符号的行
    # data = [line for line in data if line.strip() and not re.match(r'^\W', line)]
        # 对读入的数据进行处理
        for i in range(len(data)):
            line = data[i].strip()  # 去除行首行尾的空格和换行符
            if line == '':
                # 在空行的第三列添加内容'O'
                data[i] = '   O\n'
            else:
                # 将非空行按照原样保留
                data[i] = line + '\n'



    # 计算划分的索引
    train_size = int(len(data) * train_ratio)

    # 划分训练集和测试集
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data


def save_dataset(data, output_file):
    # 将数据保存到文件中
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(data)


if __name__ == "__main__":
    # 输入文件路径
    input_file = '/app/data2label/营养成分BIO标注结果.txt'
    # 划分比例
    train_ratio = 0.8

    # 划分数据集
    train_data, test_data = split_dataset(input_file, train_ratio)

    # 保存训练集和测试集
    save_dataset(train_data, '/app/data2label/data2labeltrain.txt')
    save_dataset(test_data, '/app/data2label/data2labeltest.txt')