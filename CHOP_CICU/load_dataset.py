###############################################################################################################
# A Python file that loads and preprocesses neonatal EEG data from CHOP_CICU for background classification
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###############################################################################################################
import numpy as np
import pickle
from preprocess_data import clean_data, extract_features
from wave_features import wavelet_image


# A function that reads EEG recordings and labels from the CHOP_CICU dataset for neonatal
# EEG background detection
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         squeeze_labels - a boolean indicating whether to squeeze the labels for improved background detection
# Outputs: output_recordings - a list of EEG recordings for every patient. Each array has shape (C x Q), where
#                              C is the number of channels and Q is the total number of datapoints within
#                              the patient's EEG
#          output_labels - a list of EEG background annotations for every patient, with sampling rate of 1 Hz
#          output_idx - a list of EEG background index for every patient, with sampling rate of 1 Hz
#          (note that output_labels contains specific background information while output_idx merely distinguishes
#           different background stages within each patient's dataset by simple iterators)
#          sample_freq - a list of sampling frequencies of each patient
def load_data_backgrounds(num_patients=15, squeeze_labels=True):
    # Open the pickle file that contains the raw data
    with open('CHOP_CICU_InputData2.p', 'rb') as fp:
        input_dataset = pickle.load(fp)
    # Extract labels, recordings and sampling frequencies
    labels = input_dataset['labels']
    recordings = input_dataset['recordings']
    sample_freq = input_dataset['sample_freq']
    # Initialize output placeholders
    output_labels = []
    output_recordings = []
    output_idx = []
    # Iterate over all patients within the dataset
    for ii in range(num_patients):
        # Extract patient-specific labels, recordings and sampling frequency
        patient_labels = labels[ii]
        patient_rec = recordings[ii]
        patient_fs = sample_freq[ii]
        # Initialize patient-specific outputs
        patient_labels_new = np.empty(0)
        patient_rec_new = np.empty(0)
        patient_idx_new = np.empty(0)
        # Iterate over all background EEG patterns within each patient
        for jj in range(len(patient_rec)):
            time_length = int(np.size(patient_rec[jj], axis=1) / patient_fs)
            # Merge labels to retain only three classes (0 - NC, ND, CLV // 1 - ED, LVS)
            if squeeze_labels:
                if patient_labels[jj] == 1 or patient_labels[jj] == 2 or patient_labels[jj] == 3:
                    patient_labels[jj] = 0
                else:
                    patient_labels[jj] = 1
            # Concatenate labels, indices and recordings to patient dataset
            patient_labels_new = np.r_[patient_labels_new, np.repeat(patient_labels[jj], time_length)]
            patient_idx_new = np.r_[patient_idx_new, np.repeat(jj, time_length)]
            if jj == 0:
                patient_rec_new = patient_rec[jj]
            else:
                patient_rec_new = np.c_[patient_rec_new, patient_rec[jj]]
        # Add patient-specific labels and recordings to the output
        output_labels.append(patient_labels_new)
        output_recordings.append(patient_rec_new)
        output_idx.append(patient_idx_new)
    return output_recordings, output_labels, output_idx, sample_freq


# A function that preprocesses the neonatal EEG background dataset and saves the preprocessed features and annotations
# into .npy files for future use in classifiers
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         squeeze_labels - a boolean indicating whether to squeeze the labels for improved background detection
# Outputs: None (saved files of annotations, features and images)
def preprocess_data_backgrounds(num_patients=15, squeeze_labels=True):
    all_recordings, all_labels, all_indices, all_freq = load_data_backgrounds(num_patients, squeeze_labels)
    s_length = 5
    # Iterate over each patient within the loaded dataset
    for idx, (data, label) in enumerate(zip(all_recordings, all_labels)):
        fs = all_freq[idx]
        patient_indices = all_indices[idx]
        # Obtain filtered data and labels
        new_data, new_labels, remove = clean_data(data, label, fs, s_length, use_default=True)
        np.save('patient%d_annot2b.npy' % (idx + 1), new_labels)
        print('Artifact removal complete')
        # Obtain filtered background indices
        new_indices = process_background_info(patient_indices, s_length, remove)
        np.save('patient%d_idx2.npy' % (idx + 1), new_indices)
        # Obtain statistical features
        feats = extract_features(new_data, fs, normalize='default')
        np.save('patient%d_feats2.npy' % (idx + 1), feats)
        print('Feature extraction complete')
        # Obtain image data using CWT
        images = wavelet_image(new_data, 80, downsample_factor=2, method='both')
        # Average over channels to account for inconsistent number of channels
        images_minmax = np.expand_dims(np.mean(images[0], axis=1), axis=1)
        images_mean = np.expand_dims(np.mean(images[1], axis=1), axis=1)
        # Merge two different feature maps together
        images = np.concatenate((images_minmax, images_mean), axis=1)
        np.save('patient%d_images2.npy' % (idx + 1), images)
        print('Image extraction complete')
    return None


# A helper function that formats the specific background index of each processed EEG segment
# Inputs: patient_indices - the EEG background indices of a specific patient, sampled at 1Hz
#         s_length - length of each EEG segment
#         verifier - a list indicating whether each segment was removed in the artifact rejection process
# Outputs: output_indices - a list of processed EEG background indices for each EEG segment
def process_background_info(patient_indices, s_length, verifier):
    output_indices = []
    for ii in range(len(verifier)):
        window = patient_indices[ii * s_length: (ii + 1) * s_length]
        # Only record background indices of non-artifact segments
        if verifier[ii] == 0:
            output_indices.append(window[0])
    return output_indices
