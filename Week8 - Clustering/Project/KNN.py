import getopt
import sys
import heapq
import time
import math
import numpy as np
import scipy.spatial.distance as sp
from functions import *


def knn(train_path, test_path, name, k):
    '''
    Text recognition using the KNN algorithm
    param train_path: path where is the truth.dsv file
    param test_path: path to folder with the test images
    param name: name of the ouput table with the path as well
    param k: int containing the number of nearest neighbors to consider
    '''
    train_table = import_train_table(train_path + '/truth.dsv') # if results in error, check that you are in this folder when running the script
    train_data = import_train_images(train_path, train_table)
    test_table = import_test_table(test_path)
    test_data = import_test_images(test_path, test_table)

    results = list()
    index = 0
    if not int(k) > 0:
        k = 1  # for k=1 are the best results
    for test_image in test_data:
        heap = list()
        classes = defaultdict(int)
        classification = list()
        for symbol, values in train_data.items():
            for train_image in values:
                distance = sp.minkowski(test_image, train_image)
                heapq.heappush(heap, [distance, symbol])
        [classification.append(heap.pop(0)[1]) for x in range(0, k)]
        symbol = max(set(classification), key=classification.count)
        results.append(test_table[index] + ':' + symbol)
        index += 1
    save_results(results, test_path, name)
    accuracy = check_results(name, test_path+'/truth.dsv')
    print('Accuracy of the test data: ', accuracy)


# PROJECT USING SKIPIT

# USING LOCAL FILES
knn("train_1000_10/train", "train_1000_10/test", "result_table", 0)
