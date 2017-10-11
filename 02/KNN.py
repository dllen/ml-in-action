# -*- coding: utf-8 -*-

from numpy import *
import operator


def create_date_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def print_split_line():
    print '*' * 20


def classify(in_x, data_set, labels, k):
    print in_x
    print_split_line()
    # 获取数据集行数
    data_set_size = data_set.shape[0]
    print data_set_size
    print_split_line()
    # 计算距离
    # 根据给出数据生成矩阵
    print tile(in_x, (data_set_size, 1))
    print_split_line()
    # 给定元素和数据每个元素对应坐标求差
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    print diff_mat
    print_split_line()
    # 计算距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_distances = distances.argsort()
    class_count = {}
    # 选择距离最小的K个点
    for i in range(k):
        vote_i_label = labels[sorted_distances[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 排序
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file_to_matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    num_of_lines = len(array_of_lines)
    return_mat = zeros((num_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def test():
    ho_ratio = 0.10
    data_mat, data_labels = file_to_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :], data_labels[num_test_vecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, data_labels[i])
        if (classifier_result != data_labels[i]): error_count += 1.0

    print "the total error rate is: %f" % (error_count / float(num_test_vecs))


test()

# group, labels = create_date_set()
# print_split_line()
# print classify([0, 0.2], group, labels, 2)
