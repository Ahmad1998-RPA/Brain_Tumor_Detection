import os
import numpy as np
import random
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model


train_dir = 'E:/DataScience/BrainTumorDetection/MRI Images/Training/'
test_dir = "E:/DataScience/BrainTumorDetection/MRI Images/Testing/"

# load and shuffle the train data
train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for img in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir, label, img))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# load and shuffle the test data
test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for img in os.listdir(os.path.join(test_dir, label)):
        test_paths.append(os.path.join(test_dir, label, img))
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)