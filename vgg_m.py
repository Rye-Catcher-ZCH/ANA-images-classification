from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout


md = [107, 107, 3]
imn = [224, 224, 3]


class VGG_M:
    def build(input, classes):
        model = Sequential()
        input_shape = input
        model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=input_shape, activation='relu'))
        # model.add(Lambda(lrn))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (5, 5), strides=(2, 2), activation='relu'))
        # model.add(Lambda(lrn))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(classes, activation='softmax'))

        return model