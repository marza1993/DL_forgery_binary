import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D


class binary_classification_model(object):

    """
    classe che rappresenta il modello per effettuare la classificazione binaria FORGED/ORIGINAL
    """


    def build_and_compile(input_channels):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150, 150, input_channels), activation = 'relu'))
        model.add(MaxPool2D(pool_size = (2, 2)))

        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(units = 64, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units = 1, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model



    def getBinaryModel(input_channels):

        # Initialising the CNN
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(Conv2D(32, (3, 3), input_shape = (120, 180, input_channels), activation = 'relu'))

        # Step 2 - Pooling
        classifier.add(MaxPool2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))

        # Adding a third convolutional layer
        classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))

        # Adding a fourth convolutional layer
        classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(units = 64, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))

        return classifier

