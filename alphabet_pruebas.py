import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import random
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt



test_dir = 'datos/asl_alphabet_train/asl_alphabet_test'


# Generador de flujo de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

#clases
class_word_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q',
    19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'
}



model = Sequential([
    Conv2D(32, kernel_size=3, padding='same', activation="relu", input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(32, kernel_size=3, padding='same', strides=1, activation="relu"),
    Conv2D(16, kernel_size=3, padding='same', strides=1, activation="relu"),
    Conv2D(16, kernel_size=3, padding='same', strides=1, activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(29, activation='softmax')
])

# Cargar los pesos en el nuevo modelo desde el archivo
model.load_weights('pesos_del_modelo1.h5')


while True:
    print("Menu:")
    print("1. Realizar una predicción aleatoria")
    print("2. Salir")
    opcion = input("Seleccione una opción: ")
       
   
    if opcion == "1":

        # Elegir un índice de muestra aleatorio
        random_sample_index = random.randint(0, len(test_generator) - 1)

        # Obtener una muestra aleatoria 
        test_samples, test_labels = test_generator[random_sample_index]

        # Hacer la predicción usando el modelo
        predictions = model.predict(test_samples)

        # Obtener el índice de clase predicho para la muestra aleatoria
        predicted_class_index = np.argmax(predictions[0])

        # Obtener el nombre de la clase predicha
        predicted_class = list(test_generator.class_indices.keys())[predicted_class_index]

        # Obtener el nombre de la clase real
        true_class_index = np.argmax(test_labels[0])
        true_class = list(test_generator.class_indices.keys())[true_class_index]

        # Obtener la imagen correspondiente a la muestra aleatoria
        sample_image = test_samples[0]

        # Mostrar la imagen, la clase predicha y la clase real
        plt.imshow(sample_image)
        plt.title(f'Clase Predicha: {predicted_class}\nClase Real: {true_class}')
        plt.axis('off')
        plt.show()

    elif opcion == "2":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")
