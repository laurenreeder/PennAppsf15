from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from os import listdir
from os.path import isfile, join
from functools import partial
import numpy as np


def getHOGInfo(image_file, transform_size=(300,300)):
    """
    Generate HOG feature vector, rescaling image and converting to black and white
    :param image_file: filename of image
    :param transform_size: pixel size to transform to
    :return:
    """
    og_image = imread(image_file)
    return hog(rgb2gray(resize(og_image, transform_size)), pixels_per_cell=(16, 16), cells_per_block=(1, 1))



def featureVectorsFromDirectory(directory_name, transform_size=(200, 300)):
    filenames = [join(directory_name, f) for f in listdir(directory_name) if isfile(join(directory_name, f))]
    return np.array(map(partial(getHOGInfo, transform_size=transform_size), filenames))


