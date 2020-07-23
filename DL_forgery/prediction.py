# importo i pacchetti necessari
import numpy as np
import matplotlib.pyplot as plt
from binary_classification_model import binary_classification_model
import os
import cv2
import re

PATH_BASE = "D:\\dottorato\\copy_move\\MICC-F600_DL\\"


# carico un po' di immagini di test

forged_images_folder = PATH_BASE + "\\data\\test\\forged\\"
original_images_folder = PATH_BASE + "\\data\\test\\original\\"

regex = r".*(?<!_gt)\.(png|jpg)$"
list_forged = [f for f in os.listdir(forged_images_folder) if re.search(regex,f)]
list_original = [f for f in os.listdir(original_images_folder) if re.search(regex,f)]

print("n. immagini forged: {}".format(len(list_forged)))
print("n. immagini originale: {}".format(len(list_original)))

immagini_forged = []
for i in range(0, len(list_forged)):
    img = cv2.imread(forged_images_folder + list_forged[i])
    immagini_forged.append(img)

immagini_forged = np.array(immagini_forged)
print("immagini_forged.shape: {}".format(immagini_forged.shape))

immagini_original = []
for i in range(0, len(list_original)):
    img = cv2.imread(original_images_folder + list_original[i])
    immagini_original.append(img)

immagini_original = np.array(immagini_original)
print("immagini_original.shape: {}".format(immagini_original.shape))

# carico il meodello con i pesi migliori

model_path = PATH_BASE + "\\models\\"
weight_file_name = "best_model_acc_0.977.hdf5"
model = binary_classification_model.build_and_compile(input_channels = 3)
model.load_weights(model_path + weight_file_name)


# effettuo alcune predizioni e visualizzo il risultato
print("*"*30)
print("predizione su immagini forged")
print("*"*30)



for i in range(0, immagini_forged.shape[0]):

    prediction = model.predict(np.expand_dims(cv2.resize(immagini_forged[i], (150,150)), axis=0))
    esito = "forged"
    if(prediction < 0.5):
        esito = "non forged"
    print("prediction: {}".format(prediction))
    print("esito: {}".format(esito))
    plt.imshow(immagini_forged[i])
    plt.show()


print("*"*30)
print("predizione su immagini originali")
print("*"*30)
for i in range(0, immagini_original.shape[0]):
    prediction = model.predict(np.expand_dims(cv2.resize(immagini_forged[i], (150,150)), axis=0))
    esito = "forged"
    if(prediction < 0.5):
        esito = "non forged"
    print("prediction: {}".format(prediction))
    print("esito: {}".format(esito))
    plt.imshow(immagini_original[i])
    plt.show()