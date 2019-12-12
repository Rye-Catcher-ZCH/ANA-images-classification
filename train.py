# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys

sys.path.append('..')
from vgg_m import VGG_M


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-dtest", "--dataset_test", required=True,
    #                 help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
                    help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-3
BATCH_SIZE = 64
CLASS_NUM = 4
norm_size = 224


def load_data(path, flag):
    print("[INFO] loading images...")
    data = []
    labels = []
    imagePath = []
    # grab the image paths and randomly shuffle them
    train_file = open(path, 'r')
    files = train_file.readlines()
    for file in files:
        imagePath.append(os.path.join('ANA Database1', file.split(',')[-1][0], file.split(',')[0]))
    # imagePaths = sorted(list(paths.list_images(imagePath)))
    imagePaths = sorted(imagePath)
    random.seed(42)
    random.shuffle(imagePaths)
    num = int(0.8 * len(imagePaths))
    if flag == 'train':
        images = imagePaths[0:num]
    elif flag == 'val':
        images = imagePaths[num:len(imagePaths)]
    # loop over the input images
    for image in images:
        # load the image, pre-process it, and store it in the data list
        img = cv2.imread(image)
        img = cv2.resize(img, (norm_size, norm_size))
        img = img_to_array(img)
        data.append(img)

        # extract the class label from the image path and update the
        # labels list
        label = int(image.split(os.path.sep)[-2])
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels


def train(aug, trainX, trainY, testX, testY, args):
    # initialize the model
    print("[INFO] compiling model...")
    model = VGG_M.build(input=(norm_size, norm_size, 3), classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    checkpoint = ModelCheckpoint(filepath=args["model"], monitor='val_acc', mode='auto', save_best_only='True')

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
                            epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

    # save the model to disk
    print("[INFO] serializing network...")
    # model.save(args["model"])

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on ANA classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


# --dataset_test ../../traffic-sign/test
# python train.py --dataset_train train.csv  --model traffic_sign.model

if __name__ == '__main__':
    args = args_parse()
    train_file_path = args["dataset_train"]
    # test_file_path = args["dataset_test"]
    trainX, trainY = load_data(train_file_path, flag='train')
    testX, testY = load_data(train_file_path, flag='val')
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    train(aug, trainX, trainY, testX, testY, args)

# 报错的原因是keras不是2.1.0版本
