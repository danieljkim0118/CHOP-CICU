#################################################################################################
# A Python file that provides methods defining the structure of various CNN models used to detect
# seizures within the provided neonatal EEG dataset
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#################################################################################################
import numpy as np
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, concatenate


# A function that defines a convolutional neural network model for CWT-EEG analysis
# Inputs: x - input data, with shape (C x H x W), where C is the number of channels, H and W are
#             the height and width of the image
# Outputs: model - a Keras model object that encodes the structure of a multi-channel CNN
def eeg_cnn(x):
    input_data = Input(shape=np.shape(x))
    conv1 = Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=np.shape(x))(input_data)
    conv2 = Conv2D(12, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(12, kernel_size=(3, 3), activation='relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(16, kernel_size=(3, 3))(pool1)
    conv5 = Conv2D(16, kernel_size=(3, 3))(conv4)
    conv6 = Conv2D(32, kernel_size=(4, 4))(conv5)
    pool2 = MaxPooling2D(pool_size=(3, 3))(conv6)
    flat1 = Flatten()(pool2)
    fc1 = Dense(64, activation='relu', name='fc1')(flat1)
    drop1 = Dropout(0.25, name='drop1')(fc1)
    fc2 = Dense(30, activation='relu')(drop1)
    drop2 = Dropout(0.3, name='drop2')(fc2)
    fc3 = Dense(10, activation='relu', name='fc3')(drop2)
    out = Dense(2, activation='softmax', name='out')(fc3)
    model = Model(inputs=input_data, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# A function that defines a CNN model for CWT-EEG analysis that incorporates standard EEG features
# Inputs: x1 - input images with shape (C x H x W), where C is the number of channels, H and W are
#              the height and width of the image
#         x2 - input features, a list of length F (the number of features) for each segment
# Outputs: model - a Keras model object that encodes the structure of the hybrid CNN model
# noinspection DuplicatedCode
def merge_cnn(x1, x2):
    input_img = Input(shape=np.shape(x1), name='images')
    input_feats = Input(shape=np.shape(x2), name='features')
    conv1 = Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=np.shape(x1))(input_img)
    conv2 = Conv2D(12, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(12, kernel_size=(3, 3), activation='relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(16, kernel_size=(3, 3))(pool1)
    conv5 = Conv2D(16, kernel_size=(3, 3))(conv4)
    conv6 = Conv2D(32, kernel_size=(4, 4))(conv5)
    pool2 = MaxPooling2D(pool_size=(3, 3))(conv6)
    flat1 = Flatten()(pool2)
    fc1 = Dense(64, activation='relu', name='fc1')(flat1)
    drop1 = Dropout(0.25, name='drop1')(fc1)
    concat = concatenate([drop1, input_feats])
    fc2 = Dense(30, activation='relu', name='fc2')(concat)
    drop2 = Dropout(0.3, name='drop2')(fc2)
    fc3 = Dense(10, activation='relu', name='fc3')(drop2)
    out = Dense(2, activation='softmax', name='out')(fc3)
    model = Model(inputs=[input_img, input_feats], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# A function that defines a CNN model for CWT-EEG analysis that incorporates standard EEG features
# Inputs: x1 - input images with shape (C x H x W), where C is the number of channels, H and W are
#              the height and width of the image
#         x2 - input features, a list of length F (the number of features) for each segment
# Outputs: model - a Keras model object that encodes the structure of the hybrid CNN model
# noinspection DuplicatedCode
def network(x1, x2):
    input_img = Input(shape=np.shape(x1), name='images')
    input_feats = Input(shape=np.shape(x2), name='features')
    conv1 = Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=np.shape(x1))(input_img)
    conv2 = Conv2D(12, kernel_size=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(12, kernel_size=(3, 3), activation='relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(16, kernel_size=(3, 3))(pool1)
    conv5 = Conv2D(16, kernel_size=(3, 3))(conv4)
    conv6 = Conv2D(32, kernel_size=(4, 4))(conv5)
    pool2 = MaxPooling2D(pool_size=(3, 3))(conv6)
    flat1 = Flatten()(pool2)
    fc1 = Dense(64, activation='relu', name='fc1')(flat1)
    drop1 = Dropout(0.25, name='drop1')(fc1)
    concat = concatenate([drop1, input_feats])
    fc2 = Dense(30, activation='relu', name='fc2')(concat)
    drop2 = Dropout(0.3, name='drop2')(fc2)
    fc3 = Dense(10, activation='relu', name='fc3')(drop2)
    out = Dense(2, activation='softmax', name='out')(fc3)
    model = Model(inputs=[input_img, input_feats], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
