############################################################################################
# A Python file that provides methods for evaluating test metrics for the CHOP_CICU project.
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################################
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, \
    precision_recall_fscore_support


# A function that evaluates the test metrics, specialized for the CHOP_CICU model
# Inputs: y_pred - the array of predicted outputs
#         y_target - the array of target outputs
#         squeeze_labels - whether to squeeze the labels for improved background detection
#         one_hot - whether one-hot encoding is used for predictions and labels
# Outputs: accuracy - number of correctly predicted samples / number of samples
#          weighted_acc - accuracy that adjusts the imbalance between given datasets
#          test_results - a dictionary that records the precision, recall and f1 score
#                         of each class
def test_statistics(y_pred, y_target, squeeze_labels=True, one_hot=True):
    # Decode one-hot encoded samples
    if one_hot or np.ndim(y_pred) > 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_target = np.argmax(y_target, axis=1)
    # Evaluate raw accuracy and weighted accuracy
    accuracy = accuracy_score(y_target, y_pred)
    weighted_acc = balanced_accuracy_score(y_target, y_pred)
    # Obtain full classification report
    if squeeze_labels:
        class_names = ['NC/ND/CLV', 'ED/LVS']
    else:
        class_names = ['NC', 'ND', 'CLV', 'ED', 'LVS']
    # Compute test metrics from the labels and outputs
    precision, recall, fscore, support = precision_recall_fscore_support(y_target, y_pred)
    report = classification_report(y_target, y_pred, labels=np.arange(len(class_names)), target_names=class_names)
    return accuracy, weighted_acc, precision, recall, fscore, support, report


# A function that merges segment-wise predictions into larger windows
# Inputs: pred - a list of predicted labels for each EEG segment within the test dataset
#         actual - a list of actual labels for each EEG segment within the test dataset
#         test_idx - the patient corresponding to the test dataset (starts at 1)
#         s_length - duration of each EEG segment, in seconds
#         window_size - size of the observation window for EEG background evaluation, in seconds
#         window_disp - displacement of the observation window for EEG background evaluation, in seconds
#         one_hot - whether one-hot encoding is used for predictions and labels
# Outputs: pred_new - a list of predicted labels for each EEG window within the test dataset
#          true_new - a list of actual labels for each EEG window within the test dataset
def postprocess_outputs(pred, actual, test_idx, s_length=5, window_size=60, window_disp=20, one_hot=True):
    indices = np.load('patient%d_idx2.npy' % test_idx)
    num_test = np.size(actual, axis=0)
    # Decode one-hot encoded samples
    if one_hot or np.ndim(pred) > 1:
        pred = np.argmax(pred, axis=1)
        actual = np.argmax(actual, axis=1)
    # Convert numpy arrays into lists
    pred = pred.tolist()
    actual = actual.tolist()
    indices = indices.tolist()
    # Convert window size and displacement into units of 'segment length'
    window_size = int(window_size / s_length)
    window_disp = int(window_disp / s_length)
    # Initialize outputs and counter
    pred_new = []
    true_new = []
    cnt = 0
    # Merge predictions until window reaches the end
    while cnt + window_size <= num_test:
        window_idx = indices[cnt:cnt + window_size]
        # Check whether background indices within the window are identical
        if window_idx.count(window_idx[0]) == len(window_idx):
            pred_label = compute_mode(pred[cnt:cnt + window_size], largest=True)
            true_label = compute_mode(actual[cnt:cnt + window_size], largest=True)
            pred_new.append(pred_label)
            true_new.append(true_label)
        cnt += window_disp
    return pred_new, true_new


# A function that evaluates the test metrics, specialized for the cardiac arrest prediction model
# Inputs: y_pred - the array of predicted outputs
#         y_target - the array of target outputs
#         one_hot - whether one-hot encoding is used for predictions and labels
# Outputs: accuracy - number of correctly predicted samples / number of samples
#          weighted_acc - accuracy that adjusts the imbalance between given datasets
#          precision - the precision of the model
#          recall - the recall of the model
#          fscore - the f1 score of the model
#          support - the number of samples for each label
#          report - a string that encodes the precision, recall, f1 score and support of the test data
def test_statistics_pred(y_pred, y_target, one_hot=True):
    # Decode one-hot encoded samples
    if one_hot or np.ndim(y_pred) > 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_target = np.argmax(y_target, axis=1)
    # Evaluate raw accuracy and weighted accuracy
    accuracy = accuracy_score(y_target, y_pred)
    weighted_acc = balanced_accuracy_score(y_target, y_pred)
    # Obtain full classification report
    class_names = ['Normal', 'Cardiac arrest']
    precision, recall, fscore, support = precision_recall_fscore_support(y_target, y_pred)
    report = classification_report(y_target, y_pred, labels=np.arange(len(class_names)), target_names=class_names)
    return accuracy, weighted_acc, precision, recall, fscore, support, report


# A function that postprocesses the seizure predictions made by the model
# Inputs: y_pred - the array of predicted outputs from the model
#         s_length - duration of each EEG segment, in seconds
#         window_size - size of the observation window for cardiac arrest prediction, in seconds
#         window_disp - displacement of the observation window for cardiac arrest prediction, in seconds
#         threshold - the threshold probability for alerting clinicians of oncoming cardiac arrest
# Outputs: y_pred_new - the new 1D array of predicted outputs
def postprocess_outputs_pred(y_pred, s_length=5, window_size=180, window_disp=15, threshold=0.5):
    # Decode one-hot encoded samples
    if np.ndim(y_pred) > 1:
        y_pred = np.array([1 if x[1] >= threshold else 0 for x in y_pred])
    # Copy prediction contents into new output to avoid referential aliasing
    y_pred_new = np.zeros(np.size(y_pred, axis=0))
    for ii, label in enumerate(y_pred):
        y_pred_new[ii] = label
    # Convert window dimensions and initialize counter
    window_size = int(window_size / s_length)
    window_disp = int(window_disp / s_length)
    idx = 0
    # Replace negative outputs with positive outputs for specific areas within the sliding window
    while idx + window_size <= np.size(y_pred, axis=0):
        if np.sum(y_pred[idx:idx + window_size]) >= int(0.2 * window_size):
            y_pred_new[idx:idx + window_size] = 1
        idx += window_disp
    # Remove any positive segments that do not have any neighboring positive segments
    search = int(90 / s_length)
    num = int(0.25 * search)
    for ii in range(np.size(y_pred, axis=0)):
        if ii < search:
            if int(np.sum(y_pred_new[0:(ii + search + 1)])) <= int(0.5 * num):
                y_pred_new[ii] = 0
        elif ii > np.size(y_pred, axis=0) - (search + 1):
            if int(np.sum(y_pred_new[ii - search:np.size(y_pred, axis=0)])) <= int(0.5 * num):
                y_pred_new[ii] = 0
        else:
            if int(np.sum(y_pred_new[ii - search:ii + search + 1])) <= num:
                y_pred_new[ii] = 0
    return y_pred_new


# A helper function that computes the mode of a list
# Inputs: input_list - a list of inputs, must contain a continuous sequence of numbers
#         largest - whether to use the largest mode (if not, uses the smallest mode)
# Outputs: mode - the most frequently occurring element within the list
def compute_mode(input_list, largest=True):
    min_element = min(input_list)
    num_classes = max(input_list) - min(input_list) + 1
    counter = np.zeros(num_classes)
    # Update the frequency of each element by iterating through the list
    for _, element in enumerate(input_list):
        counter[element - min_element] += 1
    modes = [pos for pos, cnt in enumerate(counter) if cnt == max(counter)]
    # Choose the most frequent output
    if largest:
        mode = max(modes) + min_element
    else:
        mode = min(modes) + min_element
    return mode
