from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from data_preparation import data_preparation
from binary_classification_model import binary_classification_model
import matplotlib.pyplot as plt

PATH_BASE = "D:\\dottorato\\copy_move\\MICC-F600_DL\\"

N_train_samples = 370
N_val_samples = 150
batch_size = 16

# preparo e configuro i dati per il preprocessing e la data-augmentation
data = data_preparation(PATH_BASE + 'data\\train', PATH_BASE + 'data\\validation', PATH_BASE + 'data\\test', batch_size)

model = binary_classification_model.build_and_compile(input_channels = 3)


# imposto la callback per salvare i pesi migliori
model_path = PATH_BASE + "/models/"
checkpoint = ModelCheckpoint(model_path + "best_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

N_epochs = 200

# callback per early stopping quando l'accuracy non sale per più di N epoche
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
#early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [checkpoint, early_stopping]

# fitting del modello
history = model.fit_generator(
        data.get_train_generator(),
        steps_per_epoch= N_train_samples // batch_size,
        epochs=N_epochs,
        validation_data=data.get_val_generator(),
        validation_steps=N_val_samples // batch_size,
        callbacks = callbacks_list)


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='best')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'validation'], loc='best')
plt.show()


# ------------- evaluation ------------------
N_test_samples = 80

# carico i pesi migliori salvati
weight_file_name = "best_model_acc_0.977.hdf5"
model = binary_classification_model.build_and_compile(input_channels = 3)

model.load_weights(model_path + weight_file_name)

# valuto l'accuratezza sul test set
score = model.evaluate_generator(data.get_test_generator(), steps = N_test_samples // batch_size)
#score = model.evaluate(x_test, y_test, verbose=0)
print("score: {}".format(score))