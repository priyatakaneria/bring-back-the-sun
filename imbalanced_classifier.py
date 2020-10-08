import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

#Load in data
training = pd.read_csv("training.csv", index_col=0)
testing = pd.read_csv("testing.csv", index_col=0)
additional_training = pd.read_csv("additional_training.csv", index_col=0)
annotation_confidence = pd.read_csv("annotation_confidence.csv", index_col=0)

#Combine Data
combined_data = pd.concat([training, additional_training], sort=False)
combined_data['confidence'] = annotation_confidence

#Remove non-confident 1's to imbalance data
combined_data = combined_data.fillna(np.nanmean(combined_data))
indexes = combined_data[(combined_data['prediction'] == 1) & (combined_data['confidence'] == 0.66)].index
count_values = combined_data['prediction'].value_counts()
count_1 = counts[1]
count_0 = counts[0]
imbalance = (count_1 - count_0)
combined_data.drop(indexes, inplace=True)

#Remove unncessary columns
trimmed_predictions = combined_data['prediction']
trimmed_annotation_confidence = combined_data['confidence']
combined_data.drop(['prediction'], axis=1, inplace=True)
combined_data.drop(['confidence'], axis=1, inplace=True)

#Normalise data
combined_data[combined_data.columns] = Normalizer().fit_transform(combined_data[combined_data.columns])
testing[testing.columns] = Normalizer().fit_transform(testing[testing.columns])

#Put values in numpy arrays
X = combined_data.values
y = trimmed_predictions.values
sample_weight = trimmed_annotation_confidence.values
Z = testing.values

#Train and predict using Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=510,min_samples_split=10, min_samples_leaf=2, max_features='sqrt', max_depth=410,random_state=1)
rf_classifier.fit(X,y,sample_weight)
predictions = rf_classifier.predict(Z)

# Save results to a csv ready for submission
data = pd.DataFrame({'prediction': predictions})
data.index += 1
data.index.name = 'ID'
submission = data.to_csv('submission.csv')