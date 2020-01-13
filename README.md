# CHOP-CICU
A project to apply image-based transfer learning on neonatal EEG recorded at Cardiac Intensive Care Units (CICU) at the Children's Hospital of Philadelphia to classify EEG background patterns and predict oncoming cardiac arrest.

Please refer to the following README to gain a better understanding of the project.

## Motivation
Forty-thousand neonates are born annually in the U.S. with congenital heart diseases (CHD), and 1% of these patients experience cardiac arrest [1]. With seizures occurring in many neonates after pediatric cardiac surgery, some hospitals have monitored their CHD patients’ EEG. From these recordings, a number of clinicians have hypothesized that EEG background patterns change preceding cardiac arrest [2]. These patterns include (but are not limited to) Normal Continuity (NC), Normal Discontinuity (ND), Continuous Low Voltage (CLV), Excessive Discontinuity (ED) and Low Voltage Suppression (LVS). Here, these five states are ordered with respect to increasing associativity with higher risk of oncoming cardiac arrest.

Although the EEG-based approach for monitoring CHD patients is increasingly being adopted by clinicians, this approach has the drawback of requiring clinicians to manually annotate the EEG recordings, resulting in an unwanted loss of time and resources for both the clinician and the affiliated medicial facility. Furthermore, it has been demonstrated that current methods for monitoring neonatal EEG recordings is not reliable to a satisfying degree - a recent study shows a poor quality of interrater agreements between discontinuity patterns within neonatal EEG [3]. This naturally calls for a quantitative method to distinguish EEG background patterns and predict cardiac arrest, reducing the dependence on clinicians' individual assessments and allowing for automated EEG analysis to run in real-time as the patients' EEG recordings are processed.

In this project, we apply deep learning to automate the process of EEG background identification and cardiac arrest prediction. The dataset is obtained from EEG recordings of 15 neonate patients at cardiac intensive care units at CHOP. A major obstacle to applying deep learning techiniques on medical data is the size of the available dataset - the number of samples is not enough to build a classifier with sufficient depth that is able to capture complex, subtle patterns that exist within the dataset. To solve this major problem that also exists within the CHOP dataset, we propose inter-EEG transfer learning as a solution.

## Transfer Learning
Transfer learning a technique that pre-trains a deep learning model upon a large dataset, and fine-tunes the model on the problem-specific dataset to enhance the model's performance when limitied data is available. An example can be found in (), where the authors built a classifier for () by pre-training a deep convolutional neural network on the famous ImageNet dataset and then fine-tuning the model on a dataset for (). Using this idea, we propose an inter-EEG transfer learning method where a deep learning model is trained upon a large, independent EEG dataset and fine-tuned on the specific EEG dataset of interest. More specifically, we first train a deep convolutional neural network (CNN) that learns to detect seizure traces within neonatal EEG from a dataset of 50 neonates in a neonatal intensive care unit (NICU). Then, we use the preconfigured model and only re-train the last several layers with the CHOP dataset to 1) detect backgrounds within neonatal EEG and 2) predict cardiac arrest onset.

## Seizure Detection in Neonatal EEG


## EEG Background Classification


## Cardiac Arrest Onset Prediction


## References
1.	KC Odegard et al., The frequency of cardiac arrests in patients with congenital heart disease undergoing cardiac catherization. Anesth Analg. 2014;118(1): 175-182
2.	NS Abend et al., A review of long-term EEG monitoring in critically ill children with hypoxic-ischemic encephalopathy, congenital heart disease, ECMO, and stroke. J Clin Neurophysiol. 2014;30(2): 134-142
3.  Massey, Shavonne L., et al. “Interrater and Intrarater Agreement in Neonatal Electroencephalogram Background Scoring.” Journal of Clinical Neurophysiology, vol. 36, no. 1, Jan. 2019, pp. 1–8., doi:10.1097/wnp.0000000000000534.
