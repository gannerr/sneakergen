import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '15'
version = 'newShoes'
newShoe_path = './' + version