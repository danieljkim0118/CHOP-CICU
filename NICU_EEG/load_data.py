##########################################################################################
# A Python file that provides methods for loading all measurements and annotations from a
# massive dataset of neonatal EEG recordings.
# To check specific EEG preprocessing techniques, please refer to preprocess_data.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
##########################################################################################
import mne.io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# A list of EEG channels to be analyzed
electrode_list = ['C3', 'C4', 'Cz', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'T3', 'T4', 'T5', 'T6',
                  'P3', 'P4', 'Pz', 'O1', 'O2']


# A function that returns a list of indices corresponding to EEG channels
# Inputs: channel_list - a list of strings that encode labels for all channels
# Outputs: a list of channel indices that represent EEG electrodes
def find_channels(channel_list):
    output = []
    for idx, electrode in enumerate(electrode_list):
        for idx2, channel in enumerate(channel_list):
            if electrode in channel:
                output.append(idx2)
    return output


# A function that loads all EEG data from the given EDF files
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         begin - the index of the first patient
#         use_filter - whether to apply a bandpass filter (0.5 - 60 Hz)
# Outputs: outputs - list of EEG recordings for every patient. Each array has shape (C x Q), where
#                    C is the number of channels and Q is the total number of datapoints within
#                    the patient's EEG
#          sample_fs - list of sampling frequencies for all patients
def load_all_data(num_patients, begin=1, use_filter=True):
    outputs = []
    sample_fs = np.zeros(num_patients)
    for ii in range(begin, begin + num_patients):
        print("==========Loading Data: Patient (%d)==========" % ii)
        file = mne.io.read_raw_edf('eeg%d.edf' % ii)  # Read the EDF file
        fs = file.info['sfreq']  # Obtain sampling frequency
        sample_fs[ii - begin] = fs
        channel_list = file.ch_names
        valid_channels = find_channels(channel_list)  # Filter out non-EEG channels
        data, _ = file[valid_channels, :]
        data = np.asarray(data)  # Obtain raw EEG recordings
        print("Shape of data: ", np.shape(data))
        # Apply an optional bandpass filter from 0.5 to 60 Hz
        if use_filter:
            coeff = butter(4, [0.5 / (fs / 2), 60 / (fs / 2)], 'bandpass')
            data = filtfilt(coeff[0], coeff[1], data)
        outputs.append(data)
    return outputs, sample_fs


# A function that loads annotations for all patients
# Inputs: num_patients - total number of patients to incorporate within the dataset
#         begin - the index of the first patient
#         weighted - decides whether seizure annotation will be voted or blindly accepted
# Outputs: a list of seizure annotations for each patient. Each seizure annotation indicates the
#          presence of seizure for its corresponding patient, with sampling rate of 1 Hz
def load_annotations(num_patients, begin=1, weighted=True):
    all_annots = []  # contains three separate annotated lists, each done by a different clinician
    # Iterate over all annotations available
    for ii in range(3):
        raw_annot = pd.read_csv('annotation%d.csv' % (ii + 1))
        annot = []  # list is used since patients contain different amounts of recordings
        for head in raw_annot.columns.values:
            patient_raw_annot = raw_annot[head]
            patient_annot = [x for x in patient_raw_annot if str(x) != 'nan']
            annot.append(patient_annot)
        all_annots.append(annot)
    output = []  # Initialize output list
    # Iterate over all designated patients
    for ii in range(begin, begin + num_patients):
        print("Reading annotations for patient %d" % ii)
        # Obtain patient-specific annotations
        patient_annot_1 = np.array(all_annots[0][ii - 1])
        patient_annot_2 = np.array(all_annots[1][ii - 1])
        patient_annot_3 = np.array(all_annots[2][ii - 1])
        # Ensure that patient annotation sizes are uniform
        cutoff_size = min(len(patient_annot_1), len(patient_annot_2), len(patient_annot_3))
        patient_annot_1 = patient_annot_1[:cutoff_size]
        patient_annot_2 = patient_annot_2[:cutoff_size]
        patient_annot_3 = patient_annot_3[:cutoff_size]
        stacked_annot = np.c_[patient_annot_1, patient_annot_2, patient_annot_3]
        if weighted:  # Apply voting method to each seizure annotation
            output.append(np.round(np.mean(stacked_annot, axis=1)))
        else:  # Consider every seizure annotation to be valid
            output.append(np.amax(stacked_annot, axis=1))
    return output
