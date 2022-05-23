from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
# ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def from_raw_to_CNN(cups, boxes, books, cupsize, boxsize, booksize, resize=True):
    # Reshape
    if resize:
        real_cups = resize_and_flatten(cups, padding=False, max_size=cupsize, output_size=128, flatten=False)
        real_boxes = resize_and_flatten(boxes, padding=False, max_size=boxsize, output_size=128, flatten=False)
        real_books = resize_and_flatten(books, padding=False, max_size=booksize, output_size=128, flatten=False)
    else:
        real_cups =cups.copy()
        real_boxes=boxes.copy()
        real_books=books.copy()
    # Scale
    real_cups = real_cups / 255
    real_boxes = real_boxes / 255
    real_books = real_books / 255
    # Shape for KERAS
    a, b, c = real_cups.shape
    real_cups = real_cups.reshape((a,b,c,1))
    a, b, c = real_boxes.shape
    real_boxes = real_boxes.reshape((a,b,c,1))
    a, b, c = real_books.shape
    real_books = real_books.reshape((a,b,c,1))
    # Targets cups, boxes = [0, 1]
    cup_target = np.zeros((len(real_cups),))
    box_target = np.ones((len(real_boxes),))
    book_target = np.ones((len(real_books),))*2

    # Combine the data
    test_data = np.concatenate((real_cups, real_boxes, real_books))
    test_targets = np.concatenate((cup_target, box_target, book_target))

    print(test_data.shape, test_targets.shape)
    return test_data, test_targets


def import_images(path):
    """
    Extracts all images from path and adds to list.
    Returns list of images as well as max-scale of any image.
    """
    imdir = path
    ext = ['png', 'jpg', 'gif']
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]
    max_size = 0
    for _, img in enumerate(images):
        w, h,_ = img.shape
        if w>max_size:
            max_size=w
        if h>max_size:
            max_size=h
    try:
        return np.asarray(images)[:, :, :, 0], max_size
    except IndexError:
        return images, max_size

def flatten(images):
    """
    Takes a list of images, and returns the flattened images (1, size),
    ready for machine learning.
    """
    # Flatten 
    flat_images = []
    for _,matrix in enumerate(images):
        flat_images.append(matrix.flatten())
    flat_images=np.asarray(flat_images)
    return flat_images


def resize_and_flatten(images, padding=False, max_size=1024, output_size=128, flatten=True):
    """
    Takes a list of images, and returns flattened vectors after resizing.
    """
    # Resize
    resized_images = []
    for _, img in enumerate(images): # Resize
        # resize image
        w, h,_ = img.shape
        # Grayscale image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if padding:
            # Add padding
            top=(max_size-h)//2
            bottom=(max_size-h)//2
            left=(max_size-w)//2
            right=(max_size-w)//2
            borderType=cv2.BORDER_CONSTANT # cv2.BORDER_WRAP
            value=[255, 255, 255] # White borders
            img_gray = cv2.copyMakeBorder(img_gray, top, bottom, left, right, borderType, value=value)
        # Finally We want a fixed-size image for input to ML
        resized_images.append(cv2.resize(img_gray, (output_size, output_size), interpolation = cv2.INTER_AREA))
    if flatten:
    # Flatten 
        flat_images = []
        for _,matrix in enumerate(resized_images):
            flat_images.append(matrix.flatten())
        flat_images=np.asarray(flat_images)

        return flat_images, resized_images
    else:
        return np.asarray(resized_images)

def normalize(all_data, scaler_type="minmax"):
    """
    Input is a nxm array (n images flattened to size m)
    or a list of such images: [nxm, nxm, ...]
    returns a scaler fit on all the data, as well as scaled data.
    """
    stacked_data = np.concatenate((all_data), axis=0)
    if scaler_type=="minmax":
        scaler = MinMaxScaler().fit(stacked_data)
    elif scaler_type=="std":
        scaler = StandardScaler().fit(stacked_data)
    else:
        raise ValueError("No such scaler. Use 'minmax', or 'std'")

    scaled = []
    for data in all_data:
        scaled.append(scaler.transform(data))
    
    return scaled, scaler

def normalize_2D(all_data, scaler_type="minmax"):
    """
    Input is a nxmxm array, not flattened!
    or a list of such images: [nxm, nxm, ...]
    returns a scaler fit on all the data, as well as scaled data.
    """

    size = len(all_data[0][0]) # For reconstruction
    
    flat_data = []
    for data in all_data:
        flat_data.append(flatten(data))

    stacked_data = np.concatenate((flat_data), axis=0)
    if scaler_type=="minmax":
        scaler = MinMaxScaler().fit(stacked_data)
    elif scaler_type=="std":
        scaler = StandardScaler().fit(stacked_data)
    else:
        raise ValueError("No such scaler. Use 'minmax', or 'std'")

    # Scale and reconstruct the data
    scaled = []
    for data in flat_data:
        scaled_row = []
        scaled_data = scaler.transform(data)
        for row in scaled_data:
            scaled_row.append(row.reshape((size, size)))
        scaled.append(scaled_row)

    return scaled, scaler

def simple_normalize(data, scaler):
    return scaler.transform(data)

def reshape(images, vec_type="row"):
    """
    Reshape data to row or column vectors.
    """
    r, c = images.shape
    if vec_type=="row":
        return images.reshape((r, 1, c))
    if vec_type=="col":
        return images.reshape((r, c, 1))
    else:
        raise ValueError("no such vec_type. Choose between ¨row¨ or ¨col¨")

def from_numpy_to_pd(data, labels):
    """
    Input is a list [data1, data2, ...] of row-data's,
    and labels [label1, label2, ...] mathcing the list.append

    returns a pandas dataframe containing all data and labels.
    """
    frames = []
    for i,d in enumerate(data):
        df = pd.DataFrame({'data':list(d), 'label': labels[i]})
        frames.append(df)
    
    return pd.concat(frames)

def flat_to_PCA(all_data, n_components=81, verbose=False):
    """
    "all_data" is a list of all data [data1, data2, ...].
    n_components is dimensions of transform.

    Should happen after flatten but before normalize
    """
    pca = PCA(n_components=n_components)
    # Stack data for fitting PCA
    stacked_data = np.concatenate((all_data), axis=0)
    if stacked_data.shape[0] < n_components:
        raise ValueError("n_components cannot be more than total datapoints.")
    pca.fit(stacked_data)
    transformed = []
    for data in all_data:
        transformed.append(pca.transform(data))

    if verbose:
        print("Explained by PCA:", sum(pca.explained_variance_ratio_))
    return transformed, pca

def simple_pca(data, pca):
    return pca.transform(data)

#############################
######## ANN HELPERS ########
#############################

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=50) # 50
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def plot_history_loss(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Val Error')
    plt.legend()

def plot_history_accuracy(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.yscale("log")
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Acc')
    plt.plot(hist['epoch'], hist['val_accuracy'], label = 'Val Acc')
    plt.legend()


def show_final_score(history, loss=True, accuracy=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    display(hist.tail(1))
    if loss:
        plot_history_loss(hist)
    if accuracy:
        plot_history_accuracy(hist)



################################
##### From computer vision #####
################################

def gaussianSmoothing(im, sigma): # From week 6
    """
    Returns the gaussian smoothed image I, and the image derivatives Ix and Iy.
    """
    # 1 obtain the kernels for gaussian and for differentiation.
    g, gx, _ = gaussian1DKernel(sigma=sigma)
    # 2 Filter the image in both directions and diff in both directions
    I = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T) # smooth I = g * g.T * I
    # 3 Differentiate - d/dx I = g * gx.T * I 
    Ix = cv2.filter2D(cv2.filter2D(im, -1, gx.T), -1, g)
    Iy = cv2.filter2D(cv2.filter2D(im, -1, g.T), -1, gx)
    return I, Ix, Iy
    
def gaussian1DKernel(sigma, rule=5, eps=0):
    """
    Returns 1D filter kernel g, and its derivative gx.
    """
    if eps:
        filter_size=eps
    else:
        filter_size = np.ceil(sigma*rule)
    x = np.arange(-filter_size, filter_size+1) # filter
    # Make kernel
    g = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-x**2 / (2*sigma**2))
    #g /= g.sum() # Normalize filter to 1. No need with normalization factor
    g = g.reshape(-1, 1) # Make it into a col vector
    # Make the derivate of g.
    # NB! Need the normalization term of the gaussian
    gx = -(-x**2)/(sigma**2) * g[:,0]
    gx = gx.reshape(-1, 1) # Make it into a col vector
    return g, gx, x

def scaleSpaced(imgs, sigma, n):
    im_scales = []
    for _, im in enumerate(imgs):
        for i in range(1,n+1):
            sig = sigma*2**i
            blur, _, _ = gaussianSmoothing(im=im, sigma=sig)
            im_scales.append(blur)
    return im_scales

def flipper(imgs, vertical=True, horizontal=False):
    flipped = []
    for _, im in enumerate(imgs):
        if vertical and horizontal:
            flipped.append(cv2.flip(im, 0))
        if vertical:
            flipped.append(cv2.flip(im, -1))
        if horizontal:
            flipped.append(cv2.flip(im, 1))
    return flipped

def rotate_imgs(imgs, angle):
    rotated = []
    for _, im in enumerate(imgs):
        im = np.asarray(im)
        image_center = tuple(np.array(im.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated.append(cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR))
    return rotated

def store_images(imgs, path, verbose=True):
    array = np.asarray(imgs)
    if verbose:
        print(array.shape)
    for i, img in enumerate(array):
        named_path = path+"augmented"+str(i)+".jpg"
        cv2.imwrite(named_path, img)