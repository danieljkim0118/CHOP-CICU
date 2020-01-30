##################################################################################################
# A Python file that provides methods defining the structure of various CNN models for classifying
# different background patterns in neonatal EEG
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
##################################################################################################
import numpy as np
from keras import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Input


# A function that defines a baseline ANN without using any pretrained models for neonatal EEG
# background classification
# Inputs: x - input data, with shape
# Outputs: model - a Keras model object that encodes a multilayer perceptron that processes
#                  EEG features for detecting backgrounds
def baseline_ann(x):
    input_data = Input(shape=np.shape(x))
    fc1 = Dense(40, activation='relu', input_shape=np.shape(x))(input_data)
    drop1 = Dropout(0.2)(fc1)
    fc2 = Dense(12, activation='relu')(drop1)
    drop2 = Dropout(0.1)(fc2)
    out = Dense(2, activation='softmax')(drop2)
    model = Model(inputs=input_data, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# A function that defines an ANN built on top of the pretrained NICU-CNN for neonatal EEG
# background classification.
# Inputs: model_file - file name of the pretrained CNN model
# Outputs: new_model - a Keras model object that encodes a multilayer perceptron
def cicu_ann(model_file):
    model = load_model(model_file)
    # Freeze all layers except for the last fully connected layer
    for idx, layer in enumerate(model.layers):
        if idx < len(model.layers) - 4:
            layer.trainable = False
            # print(layer.name)
    # Stack a new ANN model above the predefined model
    intermediary_output = model.layers[-5].output
    fc3 = Dense(15, activation='relu')(intermediary_output)
    drop3 = Dropout(0.1)(fc3)
    out = Dense(2, activation='softmax')(drop3)
    new_model = Model(inputs=model.input, outputs=out)
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return new_model


# A function that defines a baseline ANN without using any pretrained models for cardiac arrest
# onset prediction through neonatal EEG analysis
# Inputs: x - input data, with shape
# Outputs: model - a Keras model object that encodes a multilayer perceptron that processes
#                  EEG features for detecting backgrounds
def baseline_ann_pred(x):
    input_data = Input(shape=np.shape(x))
    fc1 = Dense(36, activation='relu', input_shape=np.shape(x))(input_data)
    drop1 = Dropout(0.2)(fc1)
    fc2 = Dense(25, activation='relu')(drop1)
    drop2 = Dropout(0.1)(fc2)
    fc3 = Dense(10, activation='relu')(drop2)
    drop3 = Dropout(0.1)(fc3)
    out = Dense(2, activation='softmax')(drop3)
    model = Model(inputs=input_data, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# A function that defines an ANN built on top of the pretrained NICU-CNN for cardiac arrest
# onset prediction through neonatal EEG analysis
# Inputs: model_file - file name of the pretrained CNN model
# Outputs: new_model - a Keras model object that encodes a multilayer perceptron
def cicu_ann_pred(model_file):
    model = load_model(model_file)
    # Freeze all layers except for the last fully connected layer
    for idx, layer in enumerate(model.layers):
        if idx < len(model.layers) - 4:
            layer.trainable = False
    # Stack a new ANN model above the predefined model
    intermediary_output = model.layers[-5].output
    fc3 = Dense(20, activation='relu')(intermediary_output)
    drop3 = Dropout(0.1)(fc3)
    fc4 = Dense(10, activation='relu')(drop3)
    drop4 = Dropout(0.1)(fc4)
    out = Dense(2, activation='softmax')(drop4)
    new_model = Model(inputs=model.input, outputs=out)
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return new_model
