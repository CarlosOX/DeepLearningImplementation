import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import random

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

def get_model(input_shape):
 
    model  =  Sequential([
                    Conv2D(32, kernel_size=3, padding='same', activation="relu", input_shape=input_shape),
                    MaxPooling2D((2,2)),
                    #Flatten(),
                    Conv2D(32, kernel_size=3, padding='same', strides = 1, activation="relu"),
                    Conv2D(16, kernel_size=3, padding='same', strides = 1, activation="relu"),
                    Conv2D(16, kernel_size=3, padding='same', strides = 1, activation="relu"),
                    MaxPooling2D((2,2)),
                    Flatten(),
                    Dense(29, activation='softmax')
    ])
    return model
    

input_shape = train_generator[0][0][0].shape

model = get_model(input_shape)

model.summary()

def compile_model(model):
    
    model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics=['accuracy'])



def train_model(model,train_generator,val_generator):
     
        # Entrenar el modelo usando el generador de datos
    history = model.fit(
        train_generator,
        steps_per_epoch =200,
        epochs=20,  # Número de épocas de entrenamiento
        validation_data=val_generator,
        validation_steps = 100
    )

    return history

compile_model(model)

history = train_model(model,train_generator,val_generator)


test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')

model.save_weights('pesos_del_modelo1.h5')


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
