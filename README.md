# CHOP-CICU
A project to apply image-based transfer learning on neonatal EEG recorded at Cardiac Intensive Care Units (CICU) at the Children's Hospital of Philadelphia to classify EEG background patterns and predict oncoming cardiac arrest.

Please refer to the following README to gain a better understanding of the project.

## Motivation
Forty-thousand neonates are born annually in the U.S. with congenital heart diseases (CHD), and 1% of these patients experience cardiac arrest [1]. With seizures occurring in many neonates after pediatric cardiac surgery, some hospitals have monitored their CHD patients’ EEG. From these recordings, a number of clinicians have hypothesized that EEG background patterns change preceding cardiac arrest [2]. These patterns include (but are not limited to) Normal Continuity (NC), Normal Discontinuity (ND), Continuous Low Voltage (CLV), Excessive Discontinuity (ED) and Low Voltage Suppression (LVS). Here, these five states are ordered with respect to increasing associativity with higher risk of oncoming cardiac arrest.

Although the EEG-based approach for monitoring CHD patients is increasingly being adopted by clinicians, this approach has the obvious drawback in that it requires clinicians to manually annotate the EEG recordings, resulting in an unwanted loss of time and money for the clinician and the medicial facility, respectively. Furthermore, it has been demonstrated that current methods for monitoring neonatal EEG recordings is not reliable to a satisfying degree - a recent study shows a poor quality of interrater agreements between discontinuity patterns within neonatal EEG.

To address this issue, we apply deep learning to automate the process of EEG background labeling and cardiac arrest prediction. This project relies on this widely-accepted clinical hypothesis to build classifiers that can 1) distinguish EEG background patterns that frequently occur in neonatal EEG and 2) process raw EEG data to predict oncoming cardiac arrest.

## Transfer Learning


## Seizure Detection in Neonatal EEG


## EEG Background Classification


## Cardiac Arrest Onset Prediction


## References
1.	KC Odegard et al., The frequency of cardiac arrests in patients with congenital heart disease undergoing cardiac catherization. Anesth Analg. 2014;118(1): 175-182
2.	NS Abend et al., A review of long-term EEG monitoring in critically ill children with hypoxic-ischemic encephalopathy, congenital heart disease, ECMO, and stroke. J Clin Neurophysiol. 2014;30(2): 134-142
3.  Massey, Shavonne L., et al. “Interrater and Intrarater Agreement in Neonatal Electroencephalogram Background Scoring.” Journal of Clinical Neurophysiology, vol. 36, no. 1, Jan. 2019, pp. 1–8., doi:10.1097/wnp.0000000000000534.
