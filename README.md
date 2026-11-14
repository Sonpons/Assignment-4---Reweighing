# Assignment-4---Reweighing

This project evaluates the Reweighing algorithm introduced by Kamiran and Calders (2012). Reweighing is a preprocessing method that reduces bias by assigning different weights to training observations based on their protected attribute and class label. This algorithm is implemented using the AI Fairness 360 (AIF360) Python library.

Datasets
Three publicly available datasets are used:
•	Adult Income: Predicts whether someone's income exceeds $50,000. Protected attribute: sex.
•	German Credit: Predicts good or bad credit risk. Protected attribute: age (over 25 versus 25 or younger).
•	COMPAS: Predicts whether someone reoffends within two years. Protected attribute: African-American versus non-African-American.
The required dataset files are:
•	adult.csv
•	german.data
•	propublica_data_for_fairml.csv

Requirements
The experiment uses Python and the following libraries:
•	AIF360
•	NumPy
•	pandas
•	scikit-learn
Install them using:
pip install aif360==0.6.1 scikit-learn==1.6.1 pandas==2.2.3 numpy==2.4.6

How to Run the Code
1.	Download the repository.
2.	Make sure all three dataset files are in the notebook's working directory. In Colab, upload them to /content.
3.	Open Assignment 4.ipynb.
4.	Run all cells in order.
5.	The notebook will display the fairness and accuracy results for each dataset.
The notebook uses a 70/30 train/test split, stratified by protected group and class label, with random_state=42.


Methodology
For each dataset, the following steps are performed:
1.	Calculate the original fairness metrics.
2.	Split the dataset into training and testing sets.
3.	Train a baseline logistic regression classifier without mitigation.
4.	Apply Reweighing to the training data.
5.	Train another classifier using the calculated weights.
6.	Evaluate the predictions on the test set.
7.	Compare fairness and predictive accuracy before and after mitigation.
   
The evaluation uses four metrics:
•	DP_Diff: Difference in positive prediction rates 
•	DIR: Ratio of positive prediction rates.
•	Consistency: Similarity of outcomes for similar individuals.
•	Accuracy: Percentage of correct predictions.

Results
The following results compare baseline predictions with predictions after Reweighing.
Dataset	DP_Diff Before	DP_Diff After	DIR Before	DIR After	Accuracy Before	Accuracy After
Adult	      -0.170	      -0.074	      0.341	    0.651	        84.87%	      84.24%
German	    -0.150	      -0.037	      0.824	    0.957	        77.00%	      77.67%
COMPAS	    -0.272	      -0.049	      0.635	    0.927	        67.01%	      65.93%
Reweighing improved group fairness on all three datasets. Accuracy decreased slightly for Adult and COMPAS but improved slightly for German Credit. Changes in consistency were small. Consistency measurements also depend on the features and their scales.

References
Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems.
Original paper: https://doi.org/10.1007/s10115-011-0463-8
Reweighing implementation: https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/preprocessing/reweighing.py
<img width="470" height="658" alt="image" src="https://github.com/user-attachments/assets/753f2dde-e077-4032-846c-0c5c6008758b" />
