###############################################################################################################
# A Python file that leverages Transfer Learning from a CNN classifier for neonatal EEG-based seizure detection
# The pre-trained CNN model is used to classify neonatal EEG backgrounds for clinical evaluations
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###############################################################################################################
import numpy as np
from keras.utils import np_utils
from evaluate_model import test_statistics_pred, postprocess_outputs_pred
from models import baseline_ann_pred, cicu_ann_pred


# A function that extracts training and testing data for the classifier
# Inputs: num_patients - total number of patients within available CHOP_CICU dataset
#         test_idx - the index of the patient to be used for testing phase (starts at 1)
#         include_feats - whether to incorporate quantitative EEG features
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: train_data - the data to be used during the training procedure
#          train_labels - the labels to be used during the training procedure
#          test_data - the data to be used during the testing procedure
#          test_labels - the labels to be used during the testing procedure
#          train_feats - the features to be used during the training procedure
#          test_feats - the features to be used during the testing procedure
def extract_data(num_patients, test_idx, include_feats=True, use_remote=False):
    train_data, train_labels, test_data, test_labels, train_feats, test_feats = None, None, None, None, None, None
    # Set directory path for reading data
    path = 'D:/projects/CHOP_CICU_TF/' if use_remote else ''
    for ii in range(num_patients):
        patient_data = np.load(path + 'patient%d_images_PRED.npy' % (ii + 1))
        patient_labels = np.load('patient%d_annot_PRED.npy' % (ii + 1))
        patient_feats = np.load('patient%d_feats_PRED.npy' % (ii + 1))
        # Extract testing data from specified patient index
        if ii + 1 == test_idx:
            test_data = patient_data
            test_labels = patient_labels
            if include_feats:
                test_feats = patient_feats
        # Extract training data from all other patients
        else:
            if train_labels is None:
                train_data = patient_data
                train_labels = patient_labels
                if include_feats:
                    train_feats = patient_feats
            else:
                train_data = np.r_[train_data, patient_data]
                train_labels = np.r_[train_labels, patient_labels]
                if include_feats:
                    train_feats = np.r_[train_feats, patient_feats]
    # Reshuffle training data and labels for more generalized learning
    ordering = np.arange(np.size(train_data, axis=0))
    np.random.shuffle(ordering)
    train_data = train_data[ordering]
    train_labels = train_labels[ordering]
    if include_feats:
        train_feats = train_feats[ordering]
    return train_data, train_labels, test_data, test_labels, train_feats, test_feats


# A function that applies Transfer Learning from a CNN model that has been pretrained on detecting seizures
# in neonatal EEG
# Inputs: num_patients - total number of patients within available CHOP_CICU dataset
#         test_idx - the index of the patient to be used for testing phase
#         num_epochs - number of training epochs
#         squeeze_labels - whether to squeeze the labels for improved background detection
#         include_feats - whether to include quantitative EEG features (default is false)
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: model - the trained model object
def train_model(num_patients, test_idx, num_epochs, use_baseline=False, include_feats=True, use_remote=False):
    train_x, train_y, test_x, test_y, train_f, test_f = extract_data(num_patients, test_idx, include_feats=
                                                                     include_feats, use_remote=use_remote)
    # Define class weights
    class_weights = {0: 1.0, 1: 10.0}
    # Convert labels to one-hot encoded vectors
    num_classes = 2
    train_y = np_utils.to_categorical(train_y, num_classes=num_classes)
    test_y = np_utils.to_categorical(test_y, num_classes=num_classes)
    # Set directory path for reading data
    path = 'D:/projects/CHOP_CICU_TF/' if use_remote else ''
    # Choose whether to include quantitative EEG features in the transfer learning process
    if use_baseline:
        model = baseline_ann_pred(train_f[0])
        history = model.fit(train_f, train_y, epochs=num_epochs, class_weight=class_weights, verbose=1)
        pred_y = model.predict(test_f)
        model.save('CHOP_CICU_PRED_baseline%d.h5' % test_idx, model)
    elif include_feats:
        model = cicu_ann_pred(path + 'epoch30_40p.h5')
        history = model.fit({'images': train_x, 'features': train_f}, train_y, epochs=num_epochs,
                            class_weight=class_weights, verbose=1)
        pred_y = model.predict({'images': test_x, 'features': test_f})
        model.save('CHOP_CICU_PRED_model%d.h5' % test_idx, model)
    else:
        model = cicu_ann_pred(path + 'multicnn.h5')
        history = model.fit(train_x, train_y, epochs=num_epochs, class_weight=class_weights, verbose=1)
        pred_y = model.predict(test_x)
        model.save('CHOP_CICU_PRED_model%d.h5' % test_idx, model)
    # Save training loss and accuracy
    title = 'baseline' if use_baseline else ''
    np.save('training_loss_' + str(test_idx) + title + '_PRED.npy', history.history['loss'])
    np.save('training_acc_' + str(test_idx) + title + '_PRED.npy', history.history['acc'])
    # Obtain raw test results
    print("==========Test Results==========")
    acc, _, precision, recall, fscore, support, results = test_statistics_pred(pred_y, test_y, one_hot=True)
    print("Accuracy: ", acc)
    print(results)
    # Obtain processed test results
    print("==========Processed Test Results==========")
    pred_y = postprocess_outputs_pred(pred_y, s_length=5)
    test_y = np.argmax(test_y, axis=1)
    acc, _, precision, recall, fscore, support, results = test_statistics_pred(pred_y, test_y, one_hot=False)
    print("Accuracy: ", acc)
    print(results)
    return model, acc, precision, recall, fscore, support


# A function that trains a patient-specific model multiple times and saves the results
# Inputs: num_patients - total number of patients within available CHOP_CICU dataset
#         num_iter - number of iterations for repeating the training process
#         test_idx - the index of the patient to be used for testing phase
#         num_epochs - number of training epochs
#         squeeze_labels - whether to squeeze the labels for improved background detection
#         include_feats - whether to include quantitative EEG features (default is false)
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs:
def evaluate_classifier(num_patients, num_iter, test_idx, num_epochs, use_baseline=False,
                        include_feats=True, use_remote=False):
    # Define number of classes
    num_classes = 2
    # Pre-allocate the outputs to be saved
    patient_acc = np.zeros(num_iter)
    patient_precision = np.zeros((num_iter, num_classes))
    patient_recall = np.zeros((num_iter, num_classes))
    patient_f1 = np.zeros((num_iter, num_classes))
    patient_support = np.zeros((num_iter, num_classes))
    # Evaluate the model over the specified number of iterations
    for ii in range(num_iter):
        _, acc, precision, recall, fscore, support = train_model(num_patients, test_idx, num_epochs, use_baseline,
                                                                 include_feats, use_remote)
        patient_acc[ii] = acc
        patient_precision[ii] = np.array(precision)
        patient_recall[ii] = np.array(recall)
        patient_f1[ii] = np.array(fscore)
        patient_support[ii] = np.array(support)
    # Indicate whether the model is a baseline or a TF model
    title = 'base' if use_baseline else 'model'
    # Save the results
    np.save('patient%d_acc_' % test_idx + title + '_PRED.npy', patient_acc)
    np.save('patient%d_prec_' % test_idx + title + '_PRED.npy', patient_precision)
    np.save('patient%d_rec_' % test_idx + title + '_PRED.npy', patient_recall)
    np.save('patient%d_f1_' % test_idx + title + '_PRED.npy', patient_f1)
    np.save('patient%d_support_' % test_idx + title + '_PRED.npy', patient_support)
    return patient_acc, patient_precision, patient_recall, patient_f1, patient_support


train_model(num_patients=15, test_idx=13, num_epochs=20, use_baseline=False, include_feats=True, use_remote=False)

# evaluate_classifier(num_patients=15, num_iter=50, test_idx=2, num_epochs=20, use_baseline=True)

# for ii in range(15):
#     evaluate_classifier(num_patients=15, num_iter=50, test_idx=ii + 1, num_epochs=20, use_baseline=False)
