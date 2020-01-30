###############################################################################################################
# A Python file that loads and preprocesses neonatal EEG data from CHOP_CICU for cardiac arrest prediction
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###############################################################################################################
import numpy as np
import pickle
from preprocess_data import clean_data, extract_features
from wave_features import wavelet_image


# A function that reads EEG recordings and labels from the CHOP_CICU dataset for neonatal
# cardiac arrest prediction
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         threshold - the number of minutes to start warning the patient for impending cardiac arrest
# Outputs: output_recordings - a list of EEG recordings for every patient. Each array has shape (C x Q), where
#                              C is the number of channels and Q is the total number of datapoints within
#                              the patient's EEG
#          output_labels - a list of annotations for each patient, labeled as 1 if the corresponding EEG is less than
#                          'threshold' minutes away from cardiac arrest and 0 otherwise
#          output_fs - a list of sampling frequencies for each patient
def load_data_predictions(num_patients=15, threshold=15):
    # Initialize output placeholders
    output_labels = []
    output_recordings = []
    output_fs = []
    # Iterate over all patients within the dataset
    for ii in range(num_patients):
        # Open the pickle file that contains the raw data
        with open('CHOP_CICU_PRED_InputData%d.p' % (ii + 1), 'rb') as fp:
            input_dataset = pickle.load(fp)
        # Extract labels, recordings and sampling frequencies
        patient_labels = input_dataset['labels']
        patient_recordings = input_dataset['recordings']
        patient_fs = input_dataset['sample_freq']
        # Initialize patient-specific outputs
        patient_labels_new = np.empty(0)
        patient_rec_new = np.empty(0)
        print('recordings: ', np.size(patient_recordings, axis=-1))
        # Iterate over all 1-minute segments of EEG recordings from the patient
        for jj in range(np.size(patient_recordings, axis=-1)):
            time_length = int(np.size(patient_recordings, axis=1) / patient_fs)
            # Concatenate labels, indices and recordings to patient dataset
            label = patient_labels[jj][threshold - 1]
            patient_labels_new = np.r_[patient_labels_new, np.repeat(label, time_length)]
            if jj == 0:
                patient_rec_new = patient_recordings[:, :, jj]
            else:
                patient_rec_new = np.c_[patient_rec_new, patient_recordings[:, :, jj]]
        # Add patient-specific labels and recordings to the output
        output_labels.append(patient_labels_new)
        output_recordings.append(patient_rec_new)
        output_fs.append(patient_fs)
    return output_recordings, output_labels, output_fs


# A function that preprocesses the CICU EEG cardiac arrest dataset and saves the preprocessed features and annotations
# into .npy files for future use in classifiers
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         threshold - the number of minutes to start warning the patient for impending cardiac arrest
# Outputs: None (saved files of annotations, features and images)
def preprocess_data_predictions(num_patients=15, threshold=15):
    all_recordings, all_labels, all_freq = load_data_predictions(num_patients, threshold)
    s_length = 5
    # Iterate over each patient within the loaded dataset
    for idx, (data, label) in enumerate(zip(all_recordings, all_labels)):
        if idx > 5:
            print("==========Preprocessing Data: Patient (%d)==========" % (idx + 1))
            fs = all_freq[idx]
            # Obtain filtered data and labels
            new_data, new_labels, _ = clean_data(data, label, fs, s_length, use_default=True)
            np.save('patient%d_annot_PRED.npy' % (idx + 1), new_labels)
            print('Artifact removal complete')
            # Obtain statistical features
            feats = extract_features(new_data, fs, normalize='default')
            np.save('patient%d_feats_PRED.npy' % (idx + 1), feats)
            print('Feature extraction complete')
            # Split the new data into thirds to the large memory size that is required
            idx1 = int(np.size(new_data, axis=0) / 3)
            idx2 = int(2 * np.size(new_data, axis=0) / 3)
            all_data = [new_data[:idx1], new_data[idx1:idx2], new_data[idx2:]]
            output_images = []
            # Obtain image data using CWT and combine results from all images
            for split_data in all_data:
                images = wavelet_image(split_data, 80, downsample_factor=2, method='both')
                images_minmax = np.expand_dims(np.mean(images[0], axis=1), axis=1)
                images_mean = np.expand_dims(np.mean(images[1], axis=1), axis=1)
                split_images = np.concatenate((images_minmax, images_mean), axis=1)
                if len(output_images) == 0:
                    output_images = split_images
                else:
                    output_images = np.concatenate((output_images, split_images), axis=0)
            np.save('patient%d_images_PRED.npy' % (idx + 1), output_images)
            print('Image extraction complete')
    return None
