###############################################################################################
# A Python file that provides method for training Convolutional Neural Networks (CNNs) to
# detect seizure from neonatal EEG data
# Uses Keras on a TensorFlow backend
# For details on data loading/preprocessing steps, refer to load_data.py and preprocess_data.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###############################################################################################
import numpy as np
from keras.utils import np_utils


# A function that computes the sensitivity and specificity of a given model output
# Inputs: y_pred - the array of predicted outputs
#         y_target - the array of target outputs
#         expand_data - whether to expand dataset to consider each channel as separate data input
#         num_channels - number of channels within each recording
#         one_hot - whether one-hot encoding is used for inputs
# Outputs: accuracy - number of correctly predicted samples / number of samples
#          sensitivity - number of correctly predicted seizures / number of actual seizures (recall)
#          specificity - number of correctly predicted normals / number of actual normals
#          precision - number of correctly predicted seizures / number of predicted seizures
#          f1 - 2 * (precision * recall) / (precision + recall)
def test_statistics(y_pred, y_target, expand_data, num_channels=19, one_hot=True):
    num_samples = np.size(y_pred, axis=0)
    # Decode one-hot encoded samples
    if one_hot or np.ndim(y_pred) > 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_target = np.argmax(y_target, axis=1)
    # In the case of augmented data, use voting method across channels to detect seizure
    if expand_data:
        y_target = y_target[0::num_channels]
        temp = np.zeros(int(num_samples / num_channels))
        for ii in range(int(num_samples / num_channels)):
            temp[ii] = np.round(np.mean(y_pred[ii * num_channels:(ii + 1) * num_channels]))
        y_pred = temp
    # Define relevant measures
    seizure_correct = np.sum(np.multiply(y_pred, y_target))
    normal_correct = np.sum(np.multiply(1 - y_pred, 1 - y_target))
    accuracy = (seizure_correct + normal_correct) / num_samples
    if np.sum(y_target) > 0:
        sensitivity = seizure_correct / np.sum(y_target)
    else:
        print("Sensitivity cannot be defined")
        sensitivity = 1
    specificity = normal_correct / (num_samples - np.sum(y_target))
    print("Number of samples: ", num_samples)
    print("Number of predicted seizures: ", np.sum(y_pred))
    print("Number of actual seizures: ", np.sum(y_target))
    if np.sum(y_pred) > 0:
        precision = seizure_correct / np.sum(y_pred)
    else:
        print("Precision cannot be defined")
        precision = 1
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    return accuracy, sensitivity, specificity, precision, f1


# A function that postprocesses the seizure predictions made by the model
# Inputs: y_pred - the array of predicted outputs from the model
#         s_length -
#         threshold -
# Outputs: y_pred_new - the new array of predicted outputs
def postprocess_outputs(y_pred, s_length, threshold=0.5):
    if np.ndim(y_pred) > 1:
        y_pred = np.array([1 if x[1] >= threshold else 0 for x in y_pred])
    # Copy prediction contents into new output to avoid referential aliasing
    y_pred_new = np.zeros(np.size(y_pred, axis=0))
    for ii, label in enumerate(y_pred):
        y_pred_new[ii] = label
    # Define relevant variables
    window_size = 24
    window_stride = 8
    idx = 0
    # Replace normal segments with seizure segments for interictal intervals in the sliding window
    while idx + window_size <= np.size(y_pred, axis=0):
        if np.sum(y_pred[idx:idx + window_size]) >= 8:
            y_pred_new[idx:idx + window_size] = 1
            # positions = np.where(y_pred[idx:idx + window_size] == 1)[0]
            # for ii in range(len(positions) - 1):
            #     if positions[ii + 1] - positions[ii] < window_size:
            #         y_pred_new[idx + positions[ii]:idx + positions[ii + 1]] = 1
        idx += window_stride
    # Remove any seizure segments that do not have any neighboring seizure segments
    search = int(20 / s_length)
    threshold = 5
    for ii in range(np.size(y_pred, axis=0)):
        if ii < search:
            if int(np.sum(y_pred[0:(ii + search + 1)])) <= threshold:
                y_pred_new[ii] = 0
        elif ii > np.size(y_pred, axis=0) - (search + 1):
            if int(np.sum(y_pred[ii - search:np.size(y_pred, axis=0)])) <= threshold:
                y_pred_new[ii] = 0
        else:
            if int(np.sum(y_pred[ii - search:ii + search + 1])) <= threshold:
                y_pred_new[ii] = 0
    y_pred_new = np_utils.to_categorical(y_pred_new, num_classes=2)
    return y_pred_new
