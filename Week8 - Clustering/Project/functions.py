import os
import csv
from collections import defaultdict
from PIL import Image
from PIL import ImageFilter
import numpy as np

# number of color intensities, edited in image_edit function, values 1-256
NO_OF_INTENSITIES = 16
RATIO = 0  # percentage of the train_data used for validation, 0 deactivate the validation data, values 0-99
SHARPENING = 4  # image sharpening, lower value, lower sharpening, 0 turn it off, only positive integers

def import_train_table(path):
    '''
    Import values from truth.dsv table
    param path: string with the path to the table
    return: dictionary with format {image_name:result}
    '''
    ret = dict()
    with open(path, 'r') as table:  # TODO: udelat obecne
        for row in table:
            separator = row.index(':')
            ret[row[0:separator]] = row[separator + 1]
    return ret


def import_test_table(path):
    '''
    Return the list of the image names in the folder
    param path: string with the path to the omage folder
    return: list with the names for all images
    '''
    ret = list()
    for item in os.listdir(path):
        if item.endswith(".png"):
            ret.append(item)
    return ret


def import_train_images(path, train_table):
    '''
    Import the images listed in truth.dsv and return as dictionary
    param path: string with path to the folder with images and truth.dsv table
    param train_table: dictionary of the filenames and resutls
    return: dictionary in format {result: 3D list of images}
    '''
    ret = defaultdict(list)
    for image, symbol in train_table.items():
        img = image_edit(path, image)
        ret[symbol].append(img)
    return ret


def import_test_images(path, test_table):
    '''
    Import the images from the specified path and return them in table
    param path: string with path to the folder with images
    param test_table: 1D list of the filenames
    return: 3D list with the 2D image at each index
    '''
    ret = list()   # table with probabilities from training data
    for image in test_table:
        if image.endswith(".png"):  # check only the images in the folder
            img = image_edit(path, image)
            ret.append(img)
    return ret


def image_edit(path, image):
    '''
    Sharpen iamges and reduce the number of bits for more accurate recognition
    param path: string with path to the folder with images
    param iamge: 2D list of the pixel values for the image
    '''
    img = Image.open(path + '/' + image)
    img = img.convert("L")
    for i in range(0, SHARPENING):
        img = img.filter(ImageFilter.SHARPEN)
    img = np.array(img.getdata())
    img = img // int(256/NO_OF_INTENSITIES)
    return img


def save_results(table, path, name):
    '''
    Save the results in the table in format {image_name: result}
    param path: string with path to the folder with images
    param table: 1D list of the filenames
    param name: name of the ouput table with the path as well
    '''
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow([row for row in table])


def check_results(output, results_table):
    '''
    Returns the validation/test data accuracy if we know the correct results
    param ouput: 1D table with the results of the image recognition
    param results_table: 1D table with the correct results
    return: int in range 0 to 1 of the accuracy
    '''
    guesses = dict()
    results = dict()
    accuracy = 0
    total_images = 0
    with open(output, 'r') as table:
        for row in table:
            separator = row.index(':')
            guesses[row[0:separator]] = row[separator + 1]
    with open(results_table, 'r') as table:  # TODO: udelat obecne
        for row in table:
            separator = row.index(':')
            results[row[0:separator]] = row[separator + 1]
    for name, symbol in results.items():
        if guesses[name] == symbol:
            accuracy += 1
        total_images += 1
    return accuracy/total_images
