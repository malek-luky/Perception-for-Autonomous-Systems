import getopt
import sys
import heapq
import time
import math
import numpy as np
import scipy.spatial.distance as sp
from functions import *


def create_validation_images(train_data):
    '''
    Separate the train images to train images and validation data
    Validation data were used to train the algorithms while debugging
    param train_data: dictionary in format {result: 3D list of images}
    return: separated train_data to train and validation data
    '''
    ret = dict()
    if RATIO:
        for symbol, image in train_data.items():
            index = math.ceil(len(train_data[symbol])/RATIO)
            ret[symbol] = train_data[symbol][0:index]
            train_data[symbol] = train_data[symbol][index::]
    return ret, train_data

def learning(train_data, brightnest_color):
    '''
    Sum up the number of color intensities on each pixel for all images
    param train_data: dict in format {result: 3D list of image data}
    param brightnest_color: the color with the highest color intensity
    return: dictionary in format {symbol: [pixel1:[shade 0, shade 1...],...]}
    '''
    ret = defaultdict(list)
    size = len(list(train_data.values())[0][0])
    for color_intensity in range(0, brightnest_color):
        mask = color_intensity * np.ones(shape=(size))
        for symbol, train_images in train_data.items():
            success_matrix = np.sum(mask == train_images, axis=0)
            ret[symbol].append(success_matrix)
    return ret


def compare_images(learned_images, image):
    '''
    Compare the image we are recognizing with tha matrix from the learning func
    param learned_images: dict in format {symbol: [pixel1:[shade 0, shade 1...],...]}
    param image: 2D list with image
    return: 2D list with number of equal color intensities as train_data
    '''
    ret = list()
    for pixel in range(0, len(image)):
        pixel_color = image[pixel]
        count = learned_images[pixel_color][pixel]
        ret.append(count)
    return ret

def image_recognition(image_data, learned_data, size, train_data):
    '''
    Compare each pixel to color intensity from train dataset
    param image_data: dict in format {result: 3D list of image data} we are recognizing
    param learned_data: dict in format {symbol: [pixel1:[shade 0, shade 1...],...]}
    param height: int with the image height
    param width: int with the image width
    param train_data: dict in format {result: 3D list of image data} we learned from
    return: 1D list with the recognized symbol for each image
    '''
    ret = list()
    total_sum = sum([len(x) for x in train_data.values()])
    for image in image_data:
        best_match = float('-inf')
        for symbol, learned_images in learned_data.items():
            success_matrix = np.array(compare_images(
                learned_images, image)) + np.ones(shape=(size))
            apriori = math.log(len(train_data[symbol]) / total_sum)
            probability = np.sum(
                np.log(np.divide(success_matrix, len(train_data[symbol]))))
            if probability > best_match:
                final_letter = symbol
                best_match = probability
        ret.append(final_letter)
    return ret

def naive_bayes(train_path, test_path, name):
    '''
    Text recognition using the Naive Bayes algorithm
    param train_path: path where is the truth.dsv file
    param test_path: path to folder with the test images
    param name: name of the ouput table with the path as well
    '''
    train_table = import_train_table(train_path + '/truth.dsv') # if results in error, check that you are in this folder when running the script
    train_data = import_train_images(train_path, train_table)
    test_table = import_test_table(test_path)
    test_data = import_test_images(test_path, test_table)
    validation_data, train_data = create_validation_images(train_data)

    # LEARNING FROM TRAIN DATA
    learned_data = learning(train_data, NO_OF_INTENSITIES)

    # VALIDATION DATA
    if RATIO:  # RATIO equal zero deactivate the validation dataset
        accuracy = 0
        img = Image.open(train_path + '/' + os.listdir(train_path)[0])
        size = len(img.getdata())
        for result, validation_images in validation_data.items():
            classification = image_recognition(
                validation_images, learned_data, size, train_data)
            for symbol in classification:
                if result == symbol:
                    accuracy += 1
        total_sum = sum([len(x) for x in validation_data.values()])
        print('Accuracy of the validation data: ', accuracy/total_sum)

    # TEST DATA
    results = list()
    index = 0
    img = Image.open(train_path + '/' + os.listdir(train_path)[0])
    size = len(img.getdata())
    classification = image_recognition(
        test_data, learned_data, size, train_data)
    for symbol in classification:
        results.append(test_table[index] + ':' + symbol)
        index += 1
    save_results(results, test_path, name)

    accuracy = check_results(name, test_path+'/truth.dsv')
    print('Accuracy of the test data: ',accuracy)

naive_bayes("train_1000_10/train", "train_1000_10/test", "result_table")
