import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import load

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

def Plot(history):
    plt.title("Epoch vs Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="red", label="test")
    
    plt.title("Epoch vs Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="red", label="test")

def CNN():

    CNN = Sequential()
    # Convolution 1st layer
    CNN.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform", activation="relu", padding="same",  input_shape=(100, 100, 3)))
    CNN.add(BatchNormalization())
    CNN.add(MaxPooling2D(pool_size=(2, 2))) # Pooling
    CNN.add(Dropout(0.2))

    # 2nd layer  ...and another one
    CNN.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    CNN.add(BatchNormalization())
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Dropout(0.2))

    # 3rd layer  ...and anudda wun
    CNN.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    CNN.add(BatchNormalization())
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Dropout(0.5))
    CNN.add(Dense(2, activation="softmax"))

    CNN.add(Flatten()) #Flattening
    CNN.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    CNN.add(Dense(1, activation="sigmoid"))

    opt = SGD(lr=0.001, momentum=0.9) # learning rate alpha
    CNN.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    # CNN.summary()
    return CNN

if __name__ == "__main__":
    CNN = CNN()

    imgGen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    imgGen = ImageDataGenerator(rescale=1.0/255.0)

    trainCNN = imgGen.flow_from_directory("CatsAndDogs/dbTrain/", class_mode="binary", batch_size=64, target_size=(100, 100))
    testCNN = imgGen.flow_from_directory("CatsAndDogs/dbTest", class_mode="binary", batch_size=64, target_size=(100, 100))

    history = CNN.fit_generator(trainCNN, steps_per_epoch=len(trainCNN), validation_data=testCNN, validation_steps=len(testCNN), epochs=50, verbose=0)
    _, acc = CNN.evaluate_generator(testCNN, steps=len(testCNN), verbose=0)

    print("Accuracy is %.3f" % (acc * 100.0))
    Plot(history)

    CNN.save("VGG3_MODEL2.h5")