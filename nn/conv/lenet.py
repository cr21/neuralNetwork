from tensorflow.keras.layers import Conv2D
from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import  backend as K


class Lenet():

    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(20,(5,5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2,2), strides=(2,2)))

        model.add(Conv2D(50, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))


        return  model 
