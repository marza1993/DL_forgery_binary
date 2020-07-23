from keras.preprocessing.image import ImageDataGenerator

class data_preparation(object):
    """
    permette di configurare il percorso dei dati, le operazioni di preprocessing ed eventuale data augmentation,
    tramite i data generator di keras.
    precondizione è che i dati siano organizzati in un path del tipo:
        data -> | train         |-> forged
                                |-> original
                | validation    |-> forged
                                |-> original
                | test          |-> forged
                                |-> original
    questo permette di utilizzare i data generator di keras di default                                                            
    """

    def __init__(self, train_path, validation_path, test_path, batch_size):

        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.batch_size = batch_size
        # this is the augmentation configuration we will use for training
        self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        self.test_datagen = ImageDataGenerator(rescale=1./255)


        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        self.train_generator = self.train_datagen.flow_from_directory(
                self.train_path,  # this is the target directory
                target_size=(150, 150),  # all images will be resized to 150x150
                batch_size=batch_size,
                class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        self.validation_generator = self.test_datagen.flow_from_directory(
                self.validation_path,
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='binary')

        self.test_generator = self.test_datagen.flow_from_directory(
                self.test_path,  # this is the target directory
                target_size=(150, 150),  # all images will be resized to 150x150
                batch_size=batch_size,
                class_mode='binary')


    def get_train_generator(self):
        return self.train_generator

    def get_val_generator(self):
        return self.validation_generator

    def get_test_generator(self):
        return self.test_generator


