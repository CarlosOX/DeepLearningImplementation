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
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array



train_dir = 'datos/asl_alphabet_train/asl_alphabet_train'

test_dir = 'datos/asl_alphabet_train/asl_alphabet_test'



train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    horizontal_flip = False,
                    validation_split = 0.2
                    
)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (150, 150),
                    batch_size = 20,
                    class_mode ='categorical')


# Generador de flujo de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

input_shape = train_generator[0][0][0].shape



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Congelar las capas convolucionales del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Agregar capas personalizadas para la clasificación de señas
x = Flatten()(base_model.output)
output = Dense(29, activation='softmax')(x)

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Cargar los pesos del modelo desde el archivo
model.load_weights('pesos_del_modelofinal.h5')



while True:

    print("Menu:")
    print("1. Realizar una predicción aleatoria de los datos de test")
    print("2. Realizar una predicción aleatoria con imagenes de internet")
    print("3. Salir")
    opcion = input("Seleccione una opción (1/2/3) : ")
       
   
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

        url = input("Ingrese la URL de la imagen: ")

        try:
            # Descargar la imagen desde la URL
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Guardar la imagen descargada
            with open('imagen_prediccion.jpg', 'wb') as file:
                for chunk in response.iter_content(8192):
                    file.write(chunk)

            # Cargar y preprocesar la imagen
            img = load_img('imagen_prediccion.jpg', target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 150, 150, 3))
            img_array = img_array / 255.0  # Normalizar los valores de píxeles al rango [0, 1]

            # Realizar la predicción usando el modelo final
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = list(test_generator.class_indices.keys())[predicted_class_index]

            # Mostrar la imagen y la clase predicha
            plt.imshow(img)
            plt.title(f'Clase Predicha: {predicted_class}')
            plt.axis('off')
            plt.show()

        except Exception as e:
            print("Error al procesar la imagen:", e)


    elif opcion == "3":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")


