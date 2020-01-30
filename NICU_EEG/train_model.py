###############################################################################################
# A Python file that provides method for training Convolutional Neural Networks (CNNs) to
# detect seizure from neonatal EEG data
# Uses Keras on a TensorFlow backend
# For details on data loading/preprocessing steps, refer to load_data.py and preprocess_data.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###############################################################################################
from keras.backend import set_image_data_format
from models import *
import matplotlib.pyplot as plt
from test_model import *

# Reset the image ordering so that channels occupy the first dimension
set_image_data_format('channels_first')


# A function that extracts training, validation and testing data for use in the classifier
# Inputs: num_patients - total number of patients within available dataset
#         num_train - number of patients to use in the training data
#         num_test - number of patients to use in the testing data
#         num_validation - number of patients to use in the validation data (default is 1)
#         test_patients - a 1D numpy array of patient IDs (1 to num_patients, default is empty)
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: train_data - the data to be used during the training procedure
#          train_labels - the labels to be used during the training procedure
#          validation_data - the data to be used during the validation procedure
#          validation_labels - the labels to be used during the validation procedure
#          test_data - the data to be used during the testing procedure
#          test_labels - the labels to be used during the testing procedure
#          train_feats - the EEG features to be used during the training procedure
#          validation_feats - the EEG features to be used during the validation procedure
#          test_feats - the EEG features to be used during the testing procedure
def extract_data(num_patients, num_train, num_test, num_validation=1, test_patients=np.empty(0), use_remote=False):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, \
        train_feats, validation_feats, test_feats = None, None, None, None, None, None, None, None, None
    # Warn the user if the inputs do not add up correctly
    if num_train + num_validation + num_test > num_patients:
        print('Total number of training and testing samples should not exceed total sample size')
        return None, None, None, None, None, None, None, None, None
    # Manually assign test data if patient indices are designated
    if len(test_patients) > 0:
        num_list = np.array([x for x in np.arange(num_patients) + 1 if x not in test_patients])
        np.random.shuffle(num_list)
        print(len(num_list))
        train_idx = num_list[:num_train]
        print("Train idx: ", train_idx)
        validation_idx = num_list[num_train:num_train + num_validation]
        print("Validation idx: ", validation_idx)
        test_idx = np.array(test_patients)
        print("Test idx: ", test_idx)
    # Otherwise, randomly select patients for training, validation and testing
    else:
        num_list = np.arange(num_patients) + 1
        np.random.shuffle(num_list)
        train_idx = num_list[:num_train]
        print("Train idx: ", train_idx)
        validation_idx = num_list[num_train:num_train + num_validation]
        print("Validation idx: ", validation_idx)
        test_idx = num_list[num_train + num_validation:num_train + num_validation + num_test]
        print("Test idx: ", test_idx)
    # Set directory path for reading data
    path = 'D:/projects/NICU_EEG/' if use_remote else ''
    # Extract training data and labels
    for idx in train_idx:
        print('Loading patient data: ', idx)
        patient_data = np.load(path + 'patient%d_images0.npy' % idx)
        patient_feats = np.load(path + 'patient%d_feats0.npy' % idx)
        patient_annot = np.load(path + 'patient%d_labels.npy' % idx)
        if train_labels is None:
            train_data = patient_data
            train_feats = patient_feats
            train_labels = patient_annot
        else:
            train_data = np.r_[train_data, patient_data]
            train_feats = np.r_[train_feats, patient_feats]
            train_labels = np.r_[train_labels, patient_annot]
    # Reshuffle training data and labels for more generalized learning
    ordering = np.arange(np.size(train_data, axis=0))
    np.random.shuffle(ordering)
    train_data = train_data[ordering]
    train_feats = train_feats[ordering]
    train_labels = train_labels[ordering]
    # Extract validation data and labels
    for idx in validation_idx:
        print('Loading patient data: ', idx)
        patient_data = np.load(path + 'patient%d_images0.npy' % idx)
        patient_feats = np.load(path + 'patient%d_feats0.npy' % idx)
        patient_annot = np.load(path + 'patient%d_labels.npy' % idx)
        if validation_labels is None:
            validation_data = patient_data
            validation_feats = patient_feats
            validation_labels = patient_annot
        else:
            validation_data = np.r_[validation_data, patient_data]
            validation_feats = np.r_[validation_feats, patient_feats]
            validation_labels = np.r_[validation_labels, patient_annot]
    # Extract testing data and labels
    for idx in test_idx:
        print('Loading patient data: ', idx)
        patient_data = np.load(path + 'patient%d_images0.npy' % idx)
        patient_feats = np.load(path + 'patient%d_feats0.npy' % idx)
        patient_annot = np.load(path + 'patient%d_labels.npy' % idx)
        if test_labels is None:
            test_data = patient_data
            test_feats = patient_feats
            test_labels = patient_annot
        else:
            test_data = np.r_[test_data, patient_data]
            test_feats = np.r_[test_feats, patient_feats]
            test_labels = np.r_[test_labels, patient_annot]
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, \
        train_feats, validation_feats, test_feats


# A function that applies data augmentation by considering each channel to encode independent seizure
# information
# Inputs: train_x - training data of shape N x C x H x W, as specified by other headers
#         validation_x - validation data
#         test_x - testing data
#         train_y - training labels of length N
#         validation_y - validation labels
#         test_y - testing labels
#         train_f - training features
#         validation_f - validation features
#         test_f - testing features
# Outputs: train_x_new - augmented training data of shape NC x 1 x H x W
#          train_y_new - augmented training labels of length NC
#          validation_x_new - augmented validation data
#          validation_y_new - augmented validation labels
#          test_x_new - augmented testing data
#          test_y_new - augmented testing labels
#          train_f_new - augmented training features of shape NC x F
#          validation_f_new - augmented validation features
#          test_f_new - augmented testing features
def expand_dataset(train_x, validation_x, test_x, train_y, validation_y, test_y, train_f, validation_f, test_f):
    # Expand the labels by a factor of the number of channels in the dataset
    train_y_new = np.repeat(train_y, np.size(train_x, axis=1))
    validation_y_new = np.repeat(validation_y, np.size(validation_x, axis=1))
    test_y_new = np.repeat(test_y, np.size(test_x, axis=1))
    # Expand the features by a factor of the number of channels in the dataset
    train_f_new = np.repeat(train_f, np.size(train_x, axis=1), axis=0)
    validation_f_new = np.repeat(validation_f, np.size(validation_x, axis=1), axis=0)
    test_f_new = np.repeat(test_f, np.size(test_x, axis=1), axis=0)
    # Expand the EEG recordings, treating each channel as separate data
    train_x_new = np.reshape(train_x, newshape=(np.size(train_x, axis=0) * np.size(train_x, axis=1),
                                                np.size(train_x, axis=-2), np.size(train_x, axis=-1)))
    train_x_new = np.expand_dims(train_x_new, axis=1)
    validation_x_new = np.reshape(validation_x, newshape=(np.size(validation_x, axis=0) * np.size(validation_x, axis=1),
                                                      np.size(validation_x, axis=-2), np.size(validation_x, axis=-1)))
    validation_x_new = np.expand_dims(validation_x_new, axis=1)
    test_x_new = np.reshape(test_x, newshape=(np.size(test_x, axis=0) * np.size(test_x, axis=1),
                                              np.size(test_x, axis=-2), np.size(test_x, axis=-1)))
    test_x_new = np.expand_dims(test_x_new, axis=1)
    # Randomly shuffle the training data to improve the learning procedure
    ordering = np.arange(np.size(train_x_new, axis=0))
    np.random.shuffle(ordering)
    train_x_new = train_x_new[ordering]
    train_y_new = train_y_new[ordering]
    train_f_new = train_f_new[ordering]
    return train_x_new, train_y_new, validation_x_new, validation_y_new, test_x_new, test_y_new, \
        train_f_new, validation_f_new, test_f_new


# A function that trains a multi-channel CNN model
# Inputs: num_patients - total number of patients within available dataset
#         num_train - number of patients to use in the training data
#         num_test - number of patients to use in the testing data
#         num_epochs - number of training epochs
#         expand_data - whether to expand dataset to consider each channel as separate data input
#         include_feats - whether to incorporate quantitative EEG features
#         test_patients - a 1D numpy array of patient IDs (1 to num_patients, default is empty)
#         squeeze_channel - whether to simply average over all channels
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: model - the trained model object
def train_multicnn(num_patients, num_train, num_test, num_epochs, expand_data, include_feats, test_patients=np.empty(0),
                   batch_size=1, squeeze_channel=False, use_remote=False):
    train_x, train_y, validate_x, validate_y, test_x, test_y, train_f, validate_f, test_f = \
        extract_data(num_patients, num_train, num_test, num_validation=5, test_patients=test_patients,
                     use_remote=use_remote)
    print("Shape of training data", np.shape(train_x))
    print("Shape of validation data", np.shape(validate_x))
    print("Shape of testing data", np.shape(test_x))
    print("Number of seizures in training data", np.sum(train_y))
    print("Number of normals in training data", np.size(train_y, axis=0) - np.sum(train_y))
    # Augment the dataset based on CWT images from individual channels
    if expand_data:
        train_x, train_y, validate_x, validate_y, test_x, test_y, train_f, validate_f, test_f = \
            expand_dataset(train_x, validate_x, test_x, train_y, validate_y, test_y, train_f, validate_f, test_f)
        batch_size = 20
    # Average over all channels if squeeze_channels is true
    elif squeeze_channel:
        train_x = np.expand_dims(np.mean(train_x, axis=1), axis=1)
        validate_x = np.expand_dims(np.mean(validate_x, axis=1), axis=1)
        test_x = np.expand_dims(np.mean(test_x, axis=1), axis=1)
    # Define class weights
    class_weights = {0: 1.0, 1: ((np.size(train_y, axis=0) - np.sum(train_y)) / np.sum(train_y))}
    print('class weights: ', class_weights)
    # Convert labels to one-hot encoded vectors
    train_y = np_utils.to_categorical(train_y, num_classes=2)
    validate_y = np_utils.to_categorical(validate_y, num_classes=2)
    test_y = np_utils.to_categorical(test_y, num_classes=2)
    # Apply different training procedures for different model inputs
    if include_feats:
        # Define the CNN model (includes EEG features)
        sample_data, sample_feats = train_x[0], train_f[0]
        model = network(sample_data, sample_feats)
        # Train the model while performing validation
        history = model.fit({'images': train_x, 'features': train_f}, train_y, epochs=num_epochs, validation_data=
                            ({'images': validate_x, 'features': validate_f}, validate_y), class_weight=class_weights,
                            batch_size=batch_size, verbose=1)
        pred_y = model.predict({'images': test_x, 'features': test_f})
        model.save('NICU_CNN.h5')
    else:
        # Define the CNN model
        sample_data = train_x[0]
        model = eeg_cnn(sample_data)
        # Train the model while performing validation
        history = model.fit(train_x, train_y, epochs=num_epochs, validation_data=(validate_x, validate_y),
                            class_weight=class_weights, batch_size=batch_size, verbose=1)
        pred_y = model.predict(test_x)
        model.save('multicnn.h5')
    # Plot training/validation loss and accuracy
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_loss'], 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Model loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.plot(history.history['acc'], 'b')
    plt.plot(history.history['val_acc'], 'r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Model accuracy')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    # Calculate test metrics and display the results
    accuracy, sensitivity, specificity, precision, f1 = test_statistics(pred_y, test_y, expand_data=expand_data,
                                                                        one_hot=True)
    print("Test accuracy: ", accuracy)
    print("Test sensitivity: ", sensitivity)
    print("Test specificity: ", specificity)
    print("Test precision: ", precision)
    print("Test F1: ", f1)
    print("===========================")
    # Apply additional postprocessing on the outputs and compare the results
    pred_y = postprocess_outputs(pred_y, s_length=5)
    accuracy, sensitivity, specificity, precision, f1 = test_statistics(pred_y, test_y, expand_data=expand_data,
                                                                        one_hot=True)
    print("Test accuracy: ", accuracy)
    print("Test sensitivity: ", sensitivity)
    print("Test specificity: ", specificity)
    print("Test precision: ", precision)
    print("Test F1: ", f1)
    return model


# train_multicnn(num_patients=50, num_train=40, num_test=5, num_epochs=100, test_patients=np.array([1, 2, 3, 4, 5]),
#                include_feats=True, batch_size=25, squeeze_channel=False, expand_data=False, use_remote=False)
