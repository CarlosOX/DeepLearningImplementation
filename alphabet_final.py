import matplotlib.pyplot as plt
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

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='validation'  # Usar parte de los datos para validación
)

# Generador de flujo de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)


# Build the Sequential convolutional neural network model

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


# Congelar las capas convolucionales del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Agregar capas personalizadas para la clasificación de señas
x = Flatten()(base_model.output)
output = Dense(29, activation='softmax')(x)

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=output)

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

history = model.fit(
    train_generator,
    steps_per_epoch =200,
    epochs=15,  # Número de épocas de entrenamiento
    validation_data=val_generator,
    validation_steps = 100
)




test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')

model.save_weights('pesos_del_modelofinal.h5')


while True:
    print("Menu:")
    print("1. Ver graficas")
    print("2. Realizar una predicción aleatoria")
    print("3. Salir")
    opcion = input("Seleccione una opción: ")

    if opcion == "1":

    
        # Graficos de perdida y precision en entrenamiento y validacion
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Loss vs. Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy vs. Epochs')

        # Mostrar todos los graficos al mismo tiempo
        plt.show()

    elif opcion == "2":

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

    elif opcion == "3":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")

