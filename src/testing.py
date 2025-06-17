import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def load_data(dataset_name, file_format='csv'):
    if file_format == 'csv':
        file_path = f'WTD/results/{dataset_name}_features_scaled.csv'
        df = pd.read_csv(file_path)
    elif file_format == 'h5':
        file_path = f'WTD/results/{dataset_name}_features_scaled.h5'
        df = pd.read_hdf(file_path, key='df')
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'h5'.")

    # Handle missing values (if any)
    df.fillna(df.mean(), inplace=True)
    
    # Separate features and labels
    feature_names = df.columns[:-1]  # Exclude the label column
    labels = df['label']
    
    return df, feature_names, labels





def train(features, n_folds=3):

    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # SVM classifier
    svm = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4)

    val_acc_scores = []

    for train_index, test_index in cv.split(features_scaled, labels):

        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the model
        svm.fit(X_train, y_train)
        
      
        # Validation scores
        val_predictions = svm.predict(X_test)
        val_acc = accuracy_score(y_test, val_predictions)
        val_acc_scores.append(val_acc)


    return np.mean(val_acc_scores)

dataset_name = 'X-SDD'
n_folds = 2

# Load the dataset
df, feature_names, labels = load_data(dataset_name)
features = df[feature_names].values
print("Feature_vectors shape : ", features.shape)

acc_scores = train(features, n_folds = n_folds)


print('-----------------------------------------------------------------------------------------------')
print('Average scores for all folds:')
print('- Dataset name : ', dataset_name)
print(f'> Accuracy: {acc_scores*100:.2f} %')
print('----------------------------------------------------------------')