####################################################################################
# A Python file that visualizes the results of the transfer learning project for the
# CHOP_CICU dataset
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
####################################################################################
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from evaluate_model import postprocess_outputs_pred


# A function that plots the loss and accuracy curves for the model's training procedure
# Inputs: num_patients - number of patients within the dataset
# Outputs: loss_mean - average loss over all patients for every training epoch
#          loss_std - standard deviation of loss over all patients for every training epoch
#          acc_mean - average accuracy over all patients for every training epoch
#          acc_std - standard deviation of accuracy over all patients for every training epoch
def plot_curves(num_patients=15, num_epochs=20):
    loss = np.zeros((num_patients, num_epochs))
    acc = np.zeros((num_patients, num_epochs))
    # Obtain training loss and accuracy for all patients
    for ii in range(num_patients):
        patient_loss = np.load('training_loss_%d.npy' % (ii + 1))
        patient_acc = np.load('training_acc_%d.npy' % (ii + 1))
        # Repeat extraction process over all epochs
        for jj in range(num_epochs):
            loss[ii][jj] = patient_loss[jj]
            acc[ii][jj] = patient_acc[jj]
    # Compute mean and standard deviation for each test metric over all patients
    loss_mean = np.mean(loss, axis=0)
    loss_std = np.std(loss, axis=0)
    acc_mean = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    # Plot the training loss
    plt.plot(loss_mean, 'r')
    plt.fill_between(np.arange(num_epochs), loss_mean - loss_std, loss_mean + loss_std, alpha=0.3,
                     edgecolor='r', facecolor='r')
    plt.xlabel('epochs')
    plt.xticks(np.arange(num_epochs), [str(epoch) for epoch in np.arange(num_epochs) + 1])
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.show()
    # Plot the training accuracy
    plt.plot(acc_mean, 'b')
    plt.fill_between(np.arange(num_epochs), acc_mean - acc_std, acc_mean + acc_std, alpha=0.3,
                     edgecolor='b', facecolor='b')
    plt.xlabel('epochs')
    plt.xticks(np.arange(num_epochs), [str(epoch) for epoch in np.arange(num_epochs) + 1])
    plt.ylabel('accuracy')
    plt.title('Training Accuracy')
    plt.show()
    return loss_mean, loss_std, acc_mean, acc_std


# A function that visualizes the test accuracy for the cross-validation procedure
# over all patients
# Inputs: num_patients - number of patients within the dataset
#         display_error - whether to plot error bars on the graph
# Outputs: all_acc_base - average of the base model's accuracies over all patients
#          all_acc_base_err - standard deviation of the base model's accuracies over all patients
#          all_acc_model - average of the proposed model's accuracies over all patients
#          all_acc_model_err - standard deviation of the proposed model's accuracies over all patients
def plot_accuracy(num_patients=15, display_error=True):
    width = 0.4
    # Pre-allocate arrays that hold test statistics
    all_acc_base = np.zeros(num_patients)
    all_acc_base_err = np.zeros(num_patients)
    all_acc_model = np.zeros(num_patients)
    all_acc_model_err = np.zeros(num_patients)
    # Iterate over all patients within the dataset
    for ii in range(num_patients):
        patient_acc_base = np.load('patient%d_acc_base.npy' % (ii + 1))
        all_acc_base[ii] = np.mean(patient_acc_base, axis=0)
        patient_acc_model = np.load('patient%d_acc_model.npy' % (ii + 1))
        all_acc_model[ii] = np.mean(patient_acc_model, axis=0)
        if display_error:
            all_acc_base_err[ii] = np.std(patient_acc_base, axis=0)
            all_acc_model_err[ii] = np.std(patient_acc_model, axis=0)
    # Display bar graph
    plt.bar(np.arange(num_patients), all_acc_base, width=width, yerr=all_acc_base_err, color='c')
    plt.bar(np.arange(num_patients) + width, all_acc_model, width=width, yerr=all_acc_model_err, color='b')
    plt.xticks(width / 2 + np.arange(num_patients), ['%d' % x for x in np.arange(num_patients) + 1])
    plt.yticks(0.2 * np.arange(6), np.round(0.2 * np.arange(6), decimals=1))
    plt.xlabel('Patient')
    plt.ylabel('Accuracy')
    plt.title('Comparison of model accuracy over 50 iterations')
    plt.legend(['baseline', 'custom'], loc='upper left')
    plt.show()
    return all_acc_base, all_acc_base_err, all_acc_model, all_acc_model_err


# A function that visualizes the test precision, recall, or f1 score for the cross-validation
# procedure over all patients
# Inputs: num_patients - number of patients within the dataset
#         label - the label to evaluate the model upon
#                 'negative': NC/ND/CLV background states
#                 'positive': ED/LVS background states
#         metric - the metric to visualize, accepted as a string
#                  'rec': recall, 'prec': precision, 'f1': f1 score
#         display_error - whether to plot error bars on the graph
# Outputs: None
def plot_statistics(num_patients=15, label='average', metric='rec', display_error=True):
    width = 0.4
    # Pre-allocate arrays that hold test statistics
    all_stat_base = np.zeros(num_patients)
    all_stat_base_err = np.zeros(num_patients)
    all_stat_model = np.zeros(num_patients)
    all_stat_model_err = np.zeros(num_patients)
    # Iterate over all patients in the dataset
    for ii in range(num_patients):
        patient_stat_base = np.load('patient%d_%s_base.npy' % (ii + 1, metric))
        patient_stat_model = np.load('patient%d_%s_model.npy' % (ii + 1, metric))
        if label == 'average':
            all_stat_base[ii] = np.mean(np.mean(patient_stat_base, axis=1), axis=0)
            all_stat_model[ii] = np.mean(np.mean(patient_stat_model, axis=1), axis=0)
            if display_error:
                all_stat_base_err[ii] = np.std(np.mean(patient_stat_base, axis=1), axis=0)
                all_stat_model_err[ii] = np.std(np.mean(patient_stat_model, axis=1), axis=0)
        else:
            idx = 0 if label == 'negative' else 1
            all_stat_base[ii] = np.mean(patient_stat_base[:, idx], axis=0)
            all_stat_model[ii] = np.mean(patient_stat_model[:, idx], axis=0)
            if display_error:
                all_stat_base_err[ii] = np.std(patient_stat_base[:, idx], axis=0)
                all_stat_model_err[ii] = np.std(patient_stat_model[:, idx], axis=0)
    # Modify bar graph title based on test metric
    if metric == 'rec':
        title = 'recall'
    elif metric == 'prec':
        title = 'precision'
    else:
        title = 'f1'
    # Display bar graph
    plt.bar(np.arange(num_patients), all_stat_base, width=width, yerr=all_stat_base_err, color='c')
    plt.bar(np.arange(num_patients) + width, all_stat_model, width=width, yerr=all_stat_model_err, color='b')
    plt.xticks(width / 2 + np.arange(num_patients), ['%d' % x for x in np.arange(num_patients) + 1])
    plt.yticks(0.2 * np.arange(6), np.round(0.2 * np.arange(6), decimals=1))
    plt.xlabel('Patient')
    plt.ylabel(title)
    plt.ylim(0.0, 1.0)
    plt.title('Comparison of model %s over 50 iterations' % title)
    plt.legend(['baseline', 'custom'], loc='upper left')
    plt.show()
    return None


# A function that plots the model's cardiac arrest onset predictions for a patient
# Inputs: patient_idx - index of the specific patient to observe
#         threshold - the threshold probability for alerting clinicians of oncoming cardiac arrest
#         use_baseline - whether to use the baseline ANN model for training
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: None
def plot_arrest_predictions(patient_idx, threshold=0.5, use_baseline=False, use_remote=False):
    path = 'D:/projects/CHOP_CICU_TF/' if use_remote else ''
    # Load test images, features and labels
    test_img = np.load('patient%d_images_PRED.npy' % patient_idx)
    test_feat = np.load('patient%d_feats_PRED.npy' % patient_idx)
    test_label = np.load('patient%d_annot_PRED.npy' % patient_idx)
    # Load the model and compute predictions for specified patient
    if use_baseline:
        model = load_model(path + 'CHOP_CICU_PRED_baseline%d.h5' % patient_idx)
        test_pred = model.predict(test_feat)
    else:
        model = load_model(path + 'CHOP_CICU_PRED_model%d.h5' % patient_idx)
        test_pred = model.predict({'images': test_img, 'features': test_feat})
    # Plot the raw outputs of the model
    raw_output = np.array([x[1] for x in test_pred])
    minutes = [x for x in np.arange(np.size(test_label, axis=0)) if x % 120 == 0]
    plt.plot(np.flip(test_label, axis=0), 'r')
    plt.plot(np.flip(raw_output, axis=0) + 1e-2, 'b')
    plt.plot(np.array([x + 0.75 for x in np.zeros(np.size(test_label, axis=0))]), 'm:')
    plt.xticks(minutes, [str(int(x / 12)) for x in minutes])
    plt.xlabel('Minutes elapsed')
    plt.ylabel('Output probability')
    plt.title('Raw Model Outputs - Patient %d' % patient_idx)
    plt.legend(['labels', 'outputs'], loc='upper left')
    plt.show()
    # Postprocess the model's outputs
    processed_output = postprocess_outputs_pred(test_pred, s_length=5, threshold=threshold)
    # Plot the processed outputs of the model
    plt.plot(np.flip(test_label, axis=0), 'r')
    plt.plot(np.flip(processed_output, axis=0) + 1e-2, 'b')
    plt.xticks(minutes, [str(int(x / 12)) for x in minutes])
    plt.xlabel('Minutes elapsed')
    plt.ylabel('Output probability')
    plt.title('Processed Model Outputs - Patient %d' % patient_idx)
    plt.legend(['labels', 'outputs'], loc='upper left')
    plt.show()
    # print(np.array([i for i, x in enumerate(np.flip(processed_output, axis=0)) if x == 1]))
    return None


# A function that plots the number of false alerts, computed by
# Inputs: num_patients - number of patients within the dataset
#         threshold - the threshold probability for alerting clinicians of oncoming cardiac arrest
#         use_remote - whether to use my remote GPU-supported computer (default is false)
# Outputs: None
def plot_false_alerts(num_patients, threshold=0.5, use_remote=False):
    false_alerts = [np.zeros(num_patients), np.zeros(num_patients)]
    # Obtain number of false alerts for each patient within dataset
    for patient_idx in range(1, num_patients + 1):
        path = 'D:/projects/CHOP_CICU_TF/' if use_remote else ''
        # Load test images, features and labels
        test_img = np.load('patient%d_images_PRED.npy' % patient_idx)
        test_feat = np.load('patient%d_feats_PRED.npy' % patient_idx)
        test_label = np.load('patient%d_annot_PRED.npy' % patient_idx)
        test_label = np.flip(test_label, axis=0)
        # Load the model and compute predictions for specified patient
        model_base = load_model(path + 'CHOP_CICU_PRED_baseline%d.h5' % patient_idx)
        test_pred_base = model_base.predict(test_feat)
        model_custom = load_model(path + 'CHOP_CICU_PRED_model%d.h5' % patient_idx)
        test_pred_custom = model_custom.predict({'images': test_img, 'features': test_feat})
        # Process the raw outputs of the model
        processed_output_base = postprocess_outputs_pred(test_pred_base, s_length=5, threshold=threshold)
        processed_output_base = np.flip(processed_output_base, axis=0)
        processed_output_custom = postprocess_outputs_pred(test_pred_custom, s_length=5, threshold=threshold)
        processed_output_custom = np.flip(processed_output_custom, axis=0)
        processed_output_list = [processed_output_base, processed_output_custom]
        # Repeat window analysis for baseline and custom models
        for idx, processed_output in enumerate(processed_output_list):
            cnt = 0
            win_size = 180
            flag = -win_size  # Initialized to account for edge case at the first window
            # Search for false alerts over non-cardiac arrest intervals
            while test_label[cnt + win_size] == 0:
                if np.sum(processed_output[cnt:cnt + win_size], axis=0) == win_size:
                    false_alerts[idx][patient_idx - 1] += 1
                elif np.sum(processed_output[cnt:cnt + win_size], axis=0) == 0:
                    pass
                else:
                    # Inspect every output within the window
                    for ii in range(win_size):
                        if processed_output[cnt + ii] == 1:
                            if cnt + ii - flag > win_size:
                                flag = cnt + ii
                                false_alerts[idx][patient_idx - 1] += 1
                cnt += win_size
    # Display bar graph
    width = 0.4
    plt.bar(np.arange(num_patients), false_alerts[0], width=width, color='c')
    plt.bar(np.arange(num_patients) + width, false_alerts[1], width=width, color='b')
    plt.xticks(width / 2 + np.arange(num_patients), ['%d' % x for x in np.arange(num_patients) + 1])
    plt.xlabel('Patient')
    plt.ylabel('False Detections')
    plt.title('Comparison of models for cardiac arrest prediction')
    plt.legend(['baseline', 'custom'], loc='upper left')
    plt.show()
    return None


# plot_accuracy(num_patients=15, display_error=False)
# plot_statistics(15, 'positive', 'prec', display_error=False)
# plot_arrest_predictions(13, threshold=0.75, use_baseline=False)
# plot_curves(num_patients=15, num_epochs=20)
# plot_false_alerts(15, threshold=0.75)
