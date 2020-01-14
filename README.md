# CHOP-CICU
A project to apply image-based transfer learning on neonatal EEG recorded at Cardiac Intensive Care Units (CICU) at the Children's Hospital of Philadelphia to classify EEG background patterns and predict oncoming cardiac arrest.

Please refer to the following README to gain a better understanding of the project.

## Motivation
Forty-thousand neonates are born annually in the U.S. with congenital heart diseases (CHD), and 1% of these patients experience cardiac arrest [1]. With seizures occurring in many neonates after pediatric cardiac surgery, some hospitals have monitored their CHD patients’ EEG. From these recordings, a number of clinicians have hypothesized that EEG background patterns change preceding cardiac arrest [2]. These patterns include (but are not limited to) Normal Continuity (NC), Normal Discontinuity (ND), Continuous Low Voltage (CLV), Excessive Discontinuity (ED) and Low Voltage Suppression (LVS). Here, these five states are ordered with respect to increasing associativity with higher risk of oncoming cardiac arrest.

Although the EEG-based approach for monitoring CHD patients is increasingly being adopted by clinicians, this approach has the drawback of requiring clinicians to manually annotate the EEG recordings, resulting in an unwanted loss of time and resources for both the clinician and the affiliated medicial facility. Furthermore, it has been demonstrated that current methods for monitoring neonatal EEG recordings is not reliable to a satisfying degree - a recent study shows a poor quality of interrater agreements between discontinuity patterns within neonatal EEG [3]. This naturally calls for a quantitative method to distinguish EEG background patterns and predict cardiac arrest, reducing the dependence on clinicians' individual assessments and allowing for automated EEG analysis to run in real-time as the patients' EEG recordings are processed.

In this project, we apply deep learning to automate the process of EEG background identification and cardiac arrest prediction. The dataset is obtained from EEG recordings of 15 neonate patients at cardiac intensive care units at CHOP. A major obstacle to applying deep learning techiniques on medical data is the size of the available dataset - the number of samples is not enough to build a classifier with sufficient depth that is able to capture complex, subtle patterns that exist within the dataset. To solve this major problem that also exists within the CHOP dataset, we propose inter-EEG transfer learning as a solution.

## Transfer Learning
Transfer learning a technique that pre-trains a deep learning model upon a large dataset, and fine-tunes the model on the problem-specific dataset to enhance the model's performance when limitied data is available. This technique is often employed in the field of image recognition, where researchers often employ pre-trained deep convolutional neural networks on the famous ImageNet dataset and then fine-tuning the model on a smaller image dataset of interest. Using this idea, we propose an inter-EEG transfer learning method where a deep learning model is trained upon a large, independent EEG dataset and fine-tuned on the specific EEG dataset of interest. More specifically, we first train a deep convolutional neural network (CNN) that learns to detect seizure traces within neonatal EEG from a dataset of 50 neonates in a neonatal intensive care unit (NICU). Then, we use the preconfigured model and only re-train the last several layers with the CHOP dataset to 1) detect backgrounds within neonatal EEG and 2) predict cardiac arrest onset.

## Seizure Detection in Neonatal EEG
The dataset for seizure detection of neonatal EEG is obtained from a free online database [4] as raw edf files. The contents from these files are loaded by **load_data.py**, and the artifacts are removed by **preprocess_data.py** to obtain relevant EEG recordings without noise. The latter file also performs feature/image extractions from the EEG recordings. In this project, all EEG recordings are split into non-overlapping 5 second segments for data analysis including artifact removal and feature extraction. EEG features incorporated into this project are: standard deviation, interquartile range, skewness, kurtosis, line length, bandpower in the delta/theta/alpha/beta frequency range. The numerical mean and standard deviation of these values are calculated over all EEG channels and used as input features for the deep learning model.

To obtain image features from a one-dimensional EEG signal, we use the continuous wavelet transform (CWT) to create a scaleogram that encodes the distribution of a specific EEG segment with respect to time and scale (inversely proportional to frequency). Compared to other time-frequency analysis methods such as the Fast Fourier Transform (FFT) or Short-Time Fourier Transform (STFT), the CWT offers a richer depiction of the input signal by increasing temporal resolution in high-frequency domains and increasing frequency resolution in low-frequency domains. Since the raw scaleogram is too large for efficient processing, both max-pooling and average-pooling are employed to downsample the data and both outputs are used as different "channels" of a single image. Examples of such outputs are shown below:



## EEG Background Classification


## Cardiac Arrest Onset Prediction


## References
1.	KC Odegard et al., The frequency of cardiac arrests in patients with congenital heart disease undergoing cardiac catherization. Anesth Analg. 2014;118(1): 175-182
2.	NS Abend et al., A review of long-term EEG monitoring in critically ill children with hypoxic-ischemic encephalopathy, congenital heart disease, ECMO, and stroke. J Clin Neurophysiol. 2014;30(2): 134-142
3.  Massey, Shavonne L., et al. “Interrater and Intrarater Agreement in Neonatal Electroencephalogram Background Scoring.” Journal of Clinical Neurophysiology, vol. 36, no. 1, Jan. 2019, pp. 1–8., doi:10.1097/wnp.0000000000000534.
4.  Stevenson, N. J., Tapani, K., Lauronen, L., & Vanhatalo, S. Zenodo https://doi.org/10.5281/zenodo.2547147 (2019)
