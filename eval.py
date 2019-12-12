# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import argparse
import imutils
import cv2
import os

norm_size = 224


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-s", "--show", action="store_true",
                    help="show predict image", default=False)
    args = vars(ap.parse_args())
    return args


def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('vgg_m0.93.h5')
    imagePath = []
    test_file = open(args["image"], 'r')
    files = test_file.readlines()
    for file in files:
        imagePath.append(os.path.join('DATA', file.split(',')[-1][0], file.split(',')[0]))
    # load the image
    count = 0
    for image in imagePath:
        img = cv2.imread(image)
        orig = img.copy()

        # pre-process the image for classification
        img = cv2.resize(img, (norm_size, norm_size))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        labels = int(image.split(os.path.sep)[-2])
        labels = to_categorical(labels, num_classes=4)

        # classify the input image
        result = model.predict(img)[0]
        # print (result.shape)
        proba = np.max(result)
        label = str(np.where(result == proba)[0])
        # label = "{}: {:.2f}%".format(label, proba * 100)
        # print(label)
        if label == str(np.where(labels == np.max(labels))[0]):
            count += 1
    acc = count / len(imagePath)
    print(acc)



# python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    args = args_parse()
    predict(args)
