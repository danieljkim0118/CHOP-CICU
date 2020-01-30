##########################################################################################
# A Python file that provides methods for loading all measurements and annotations from a
# massive dataset of neonatal EEG recordings.
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
##########################################################################################
import numpy as np
import scipy.stats
from wave_features import line_length, bandpower


# A function that performs artifact rejection over all EEG recordings within the given dataset
# and updates the annotation file in accordance with the processed EEG data
# Inputs: input_data - EEG sample of shape C x Q, where C is the number of valid channels and
#                      Q is the total number of datapoints within the given patient's EEG
#         input_annot - a list of seizure annotations of the given patient, with length (Q / fs) i.e. 1 Hz
#         fs - sampling frequency of the patient's EEG recording
#         s_length - length of each EEG segment to be inspected, in seconds
#         use_default - whether to use the default artifact rejection method, which simply mutes
#                       the EEG segments that violate several criteria for typical EEG signals
# Outputs: output_data - a set of processed EEG segments with shape N* x C x S, where N* is the number
#                        of valid EEG segments from the patient's dataset, C is the number of valid
#                        channels and S is the number of datapoints within each segment (S = fs * s_length)
#          output_annot - a modified list of seizure annotations of the given patient with length N*
#          verifier - a list indicating whether each EEG segment should be removed, with length N
def clean_data(input_data, input_annot, fs, s_length, use_default):
    print("Input annot: ", len(input_annot))
    num_segments = int(np.size(input_data, axis=1) / (fs * s_length))
    # Reshape the input data
    for ii in range(num_segments):
        # print('processing segment: %d' % (ii + 1))
        sample = input_data[:, int(ii * fs * s_length): int((ii + 1) * fs * s_length)]
        if ii == 0:
            stacked_sample = np.expand_dims(sample, axis=0)
        else:
            stacked_sample = np.r_[stacked_sample, np.expand_dims(sample, axis=0)]
    processed_data, remove = remove_artifacts(stacked_sample, fs, default=use_default)
    output_data = []
    output_annot = []
    # Remove all outlier artifacts
    for ii in range(len(remove)):
        if remove[ii] == 0:
            annot = np.amax(input_annot[int(ii * s_length): int((ii + 1) * s_length)])
            output_annot.append(annot)
            if len(output_data) == 0:
                output_data = np.expand_dims(processed_data[ii], axis=0)
            else:
                output_data = np.r_[output_data, np.expand_dims(processed_data[ii], axis=0)]
    output_annot = np.array(output_annot)
    return output_data, output_annot, remove


# A function that decides whether each EEG segment is an artifact-free signal
# Inputs: input_data - EEG data of shape N x C x S, where N is the number of EEG segments from
#                      a patient's dataset, C is the number of valid EEG channels and S is the
#                      number of samples within the EEG segment
#         fs - sampling frequency of the EEG recording
#         default - whether to use the default artifact rejection method, which removes segments
#                   that exceed a certain threshold under
#                   1) Range 2) Line Length 3) Bandpower in beta frequency band (25 - 60 Hz)
#                   otherwise, use a more sophisticated artifact rejection method (TBD)
#                   If default=True, then output_data is the same as input_data, while if
#                   default=False, 'remove' only contains True for segments that have NaNs or
#                   that have flat recordings (zero variance)
# Outputs: input_data - the input  with shape N x C x S (as described above)
#          remove - a list indicating whether each EEG segment should be removed, with length N
# noinspection SpellCheckingInspection
def remove_artifacts(input_data, fs, default=True):
    # Remove NaN values
    output_data, remove = remove_nans(input_data)
    # A naive statistical method is used as default, rejecting outlier EEG segments based on z-scores
    if default:
        minmax = np.mean(np.amax(output_data, axis=2) - np.amin(output_data, axis=2), axis=1)
        range_z = scipy.stats.zscore(minmax)
        llength = np.mean(line_length(output_data), axis=1)
        llength_z = scipy.stats.zscore(llength)
        bdpower = np.mean(bandpower(output_data, fs, 25, 60), axis=1)
        bdpower_z = scipy.stats.zscore(bdpower)
        # Initialize iterator variables for update procedure
        cnt = 0
        idx = 0
        # Update the removal list whenever statistical outliers are found
        while cnt < np.size(output_data, axis=0):
            if remove[idx] == 0:
                # if (range_z[cnt] > 3) or (llength_z[cnt] > 3) or (bdpower_z[cnt] > 3):
                #     remove[idx] = 1
                if range_z[cnt] > 3:
                    remove[idx] = 2
                elif llength_z[cnt] > 3:
                    remove[idx] = 3
                elif bdpower_z[cnt] > 3:
                    remove[idx] = 4
                cnt += 1
            idx += 1
        print('Number of rejected segments: ', np.sum(remove))
    return input_data, remove


# A function that removes NaN recordings from an input EEG dataset
# Inputs: input_data - EEG data of shape N x C x S, where N is the number of EEG segments from
#                      a patient's dataset, C is the number of valid EEG channels and S is the
#                      number of samples within the EEG segment
# Outputs: output_data - EEG data of shape N' x C x S, where N' is the number of EEG segments
#                        that do not contain NaN values
#          remove - a list indicating whether each EEG segment should be removed, with length N
def remove_nans(input_data):
    boolean_mask = np.isnan(input_data)
    output_data = []
    remove = np.zeros(np.size(input_data, axis=0), dtype=bool)
    # Iterate over all segments, adding non-NaN recordings to the output
    print('Input data: ', np.size(input_data, axis=0))
    for ii in range(np.size(input_data, axis=0)):
        if not boolean_mask[ii].any():
            if len(output_data) == 0:
                output_data = np.expand_dims(input_data[ii], axis=0)
            else:
                output_data = np.r_[output_data, np.expand_dims(input_data[ii], axis=0)]
        else:
            remove[ii] = 1
    return output_data, remove


# A function that extracts statistical features from a set of EEG segments from a given patient
# Inputs: input_data - a set of processed EEG segments with shape N* x C x S, where N* is the number
#                      of valid EEG segments from the patient's dataset, C is the number of valid
#                      channels and S is the number of datapoints within each segment
#         fs - sampling frequency of the patient's EEG recording
#         normalize - normalization method to be used for the extracted features
#                     'default' normalizes features from 0 to 1
#                     'zscore' normalizes features according to their z-scores
#                     'minmax' normalizes features from -1 to 1
# Outputs: output_feats - an array of shape N* x F, where N* is the number of non-artifact EEG segments
#                         and F is the number of features for each EEG segment
# noinspection PyUnboundLocalVariable
def extract_features(input_data, fs, normalize='default'):
    # Standard deviation of the measurements
    stdv = np.std(input_data, axis=-1)
    # Interquartile range of the measurements
    iqr = np.percentile(input_data, 75, axis=-1) - np.percentile(input_data, 25, axis=-1)
    # Skewness of the signal
    skew = scipy.stats.skew(input_data, axis=-1)
    # Kurtosis of the signal
    kurt = scipy.stats.kurtosis(input_data, axis=-1)
    # Line length of the signal
    llength = line_length(input_data)
    # Bandpower of the signal in delta, theta, alpha and beta range
    bdpower_delta = bandpower(input_data, fs, 0.5, 4)
    bdpower_theta = bandpower(input_data, fs, 4, 8)
    bdpower_alpha = bandpower(input_data, fs, 8, 12)
    bdpower_beta = bandpower(input_data, fs, 12, 25)
    # Create a list of all features and compute mean and standard deviation of each feature
    # across all valid EEG channels of the patient
    feature_list = [stdv, iqr, skew, kurt, llength, bdpower_delta, bdpower_theta, bdpower_alpha, bdpower_beta]
    for idx, feature in enumerate(feature_list):
        feat_mean = np.mean(feature, axis=-1)
        feat_std = np.std(feature, axis=-1)
        if idx == 0:
            output_feats = np.c_[feat_mean, feat_std]
        else:
            output_feats = np.c_[output_feats, feat_mean, feat_std]
    # Normalize the feature outputs from 0 to 1
    output_feats = (output_feats - np.amin(output_feats, axis=0)) / \
                   (np.amax(output_feats, axis=0) - np.amin(output_feats, axis=0))
    # Normalize according to z-score
    if normalize == 'zscore':
        output_feats = scipy.stats.zscore(output_feats, axis=0)
    # Normalize from -1 to +1
    elif normalize == 'minmax':
        output_feats = output_feats * 2 - 1
    return output_feats
