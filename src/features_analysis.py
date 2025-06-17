from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# Define the types of features to analyze
initial_feature_types = ['mean', 'variance', 'skewness', 'kurtosis', 
                         'contrast', 'dissimilarity', 'homogeneity', 'energy', 
                         'correlation', 'lbp', 'spectral_energy', 'spectral_entropy']
all_feature = "mean_LL,variance_LL,skewness_LL,kurtosis_LL,contrast_1_LH,dissimilarity_1_LH,homogeneity_1_LH,energy_1_LH,correlation_1_LH,mean_1_LH,variance_1_LH,skewness_1_LH,kurtosis_1_LH,lbp_1_LH_0,lbp_1_LH_1,lbp_1_LH_2,lbp_1_LH_3,lbp_1_LH_4,lbp_1_LH_5,lbp_1_LH_6,lbp_1_LH_7,lbp_1_LH_8,lbp_1_LH_9,spectral_energy_1_LH,spectral_entropy_1_LH,contrast_1_HL,dissimilarity_1_HL,homogeneity_1_HL,energy_1_HL,correlation_1_HL,mean_1_HL,variance_1_HL,skewness_1_HL,kurtosis_1_HL,lbp_1_HL_0,lbp_1_HL_1,lbp_1_HL_2,lbp_1_HL_3,lbp_1_HL_4,lbp_1_HL_5,lbp_1_HL_6,lbp_1_HL_7,lbp_1_HL_8,lbp_1_HL_9,spectral_energy_1_HL,spectral_entropy_1_HL,contrast_1_HH,dissimilarity_1_HH,homogeneity_1_HH,energy_1_HH,correlation_1_HH,mean_1_HH,variance_1_HH,skewness_1_HH,kurtosis_1_HH,lbp_1_HH_0,lbp_1_HH_1,lbp_1_HH_2,lbp_1_HH_3,lbp_1_HH_4,lbp_1_HH_5,lbp_1_HH_6,lbp_1_HH_7,lbp_1_HH_8,lbp_1_HH_9,spectral_energy_1_HH,spectral_entropy_1_HH,contrast_2_LH,dissimilarity_2_LH,homogeneity_2_LH,energy_2_LH,correlation_2_LH,mean_2_LH,variance_2_LH,skewness_2_LH,kurtosis_2_LH,lbp_2_LH_0,lbp_2_LH_1,lbp_2_LH_2,lbp_2_LH_3,lbp_2_LH_4,lbp_2_LH_5,lbp_2_LH_6,lbp_2_LH_7,lbp_2_LH_8,lbp_2_LH_9,spectral_energy_2_LH,spectral_entropy_2_LH,contrast_2_HL,dissimilarity_2_HL,homogeneity_2_HL,energy_2_HL,correlation_2_HL,mean_2_HL,variance_2_HL,skewness_2_HL,kurtosis_2_HL,lbp_2_HL_0,lbp_2_HL_1,lbp_2_HL_2,lbp_2_HL_3,lbp_2_HL_4,lbp_2_HL_5,lbp_2_HL_6,lbp_2_HL_7,lbp_2_HL_8,lbp_2_HL_9,spectral_energy_2_HL,spectral_entropy_2_HL,contrast_2_HH,dissimilarity_2_HH,homogeneity_2_HH,energy_2_HH,correlation_2_HH,mean_2_HH,variance_2_HH,skewness_2_HH,kurtosis_2_HH,lbp_2_HH_0,lbp_2_HH_1,lbp_2_HH_2,lbp_2_HH_3,lbp_2_HH_4,lbp_2_HH_5,lbp_2_HH_6,lbp_2_HH_7,lbp_2_HH_8,lbp_2_HH_9,spectral_energy_2_HH,spectral_entropy_2_HH,contrast_3_LH,dissimilarity_3_LH,homogeneity_3_LH,energy_3_LH,correlation_3_LH,mean_3_LH,variance_3_LH,skewness_3_LH,kurtosis_3_LH,lbp_3_LH_0,lbp_3_LH_1,lbp_3_LH_2,lbp_3_LH_3,lbp_3_LH_4,lbp_3_LH_5,lbp_3_LH_6,lbp_3_LH_7,lbp_3_LH_8,lbp_3_LH_9,spectral_energy_3_LH,spectral_entropy_3_LH,contrast_3_HL,dissimilarity_3_HL,homogeneity_3_HL,energy_3_HL,correlation_3_HL,mean_3_HL,variance_3_HL,skewness_3_HL,kurtosis_3_HL,lbp_3_HL_0,lbp_3_HL_1,lbp_3_HL_2,lbp_3_HL_3,lbp_3_HL_4,lbp_3_HL_5,lbp_3_HL_6,lbp_3_HL_7,lbp_3_HL_8,lbp_3_HL_9,spectral_energy_3_HL,spectral_entropy_3_HL,contrast_3_HH,dissimilarity_3_HH,homogeneity_3_HH,energy_3_HH,correlation_3_HH,mean_3_HH,variance_3_HH,skewness_3_HH,kurtosis_3_HH,lbp_3_HH_0,lbp_3_HH_1,lbp_3_HH_2,lbp_3_HH_3,lbp_3_HH_4,lbp_3_HH_5,lbp_3_HH_6,lbp_3_HH_7,lbp_3_HH_8,lbp_3_HH_9,spectral_energy_3_HH,spectral_entropy_3_HH,contrast_4_LH,dissimilarity_4_LH,homogeneity_4_LH,energy_4_LH,correlation_4_LH,mean_4_LH,variance_4_LH,skewness_4_LH,kurtosis_4_LH,lbp_4_LH_0,lbp_4_LH_1,lbp_4_LH_2,lbp_4_LH_3,lbp_4_LH_4,lbp_4_LH_5,lbp_4_LH_6,lbp_4_LH_7,lbp_4_LH_8,lbp_4_LH_9,spectral_energy_4_LH,spectral_entropy_4_LH,contrast_4_HL,dissimilarity_4_HL,homogeneity_4_HL,energy_4_HL,correlation_4_HL,mean_4_HL,variance_4_HL,skewness_4_HL,kurtosis_4_HL,lbp_4_HL_0,lbp_4_HL_1,lbp_4_HL_2,lbp_4_HL_3,lbp_4_HL_4,lbp_4_HL_5,lbp_4_HL_6,lbp_4_HL_7,lbp_4_HL_8,lbp_4_HL_9,spectral_energy_4_HL,spectral_entropy_4_HL,contrast_4_HH,dissimilarity_4_HH,homogeneity_4_HH,energy_4_HH,correlation_4_HH,mean_4_HH,variance_4_HH,skewness_4_HH,kurtosis_4_HH,lbp_4_HH_0,lbp_4_HH_1,lbp_4_HH_2,lbp_4_HH_3,lbp_4_HH_4,lbp_4_HH_5,lbp_4_HH_6,lbp_4_HH_7,lbp_4_HH_8,lbp_4_HH_9,spectral_energy_4_HH,spectral_entropy_4_HH,contrast_5_LH,dissimilarity_5_LH,homogeneity_5_LH,energy_5_LH,correlation_5_LH,mean_5_LH,variance_5_LH,skewness_5_LH,kurtosis_5_LH,lbp_5_LH_0,lbp_5_LH_1,lbp_5_LH_2,lbp_5_LH_3,lbp_5_LH_4,lbp_5_LH_5,lbp_5_LH_6,lbp_5_LH_7,lbp_5_LH_8,lbp_5_LH_9,spectral_energy_5_LH,spectral_entropy_5_LH,contrast_5_HL,dissimilarity_5_HL,homogeneity_5_HL,energy_5_HL,correlation_5_HL,mean_5_HL,variance_5_HL,skewness_5_HL,kurtosis_5_HL,lbp_5_HL_0,lbp_5_HL_1,lbp_5_HL_2,lbp_5_HL_3,lbp_5_HL_4,lbp_5_HL_5,lbp_5_HL_6,lbp_5_HL_7,lbp_5_HL_8,lbp_5_HL_9,spectral_energy_5_HL,spectral_entropy_5_HL,contrast_5_HH,dissimilarity_5_HH,homogeneity_5_HH,energy_5_HH,correlation_5_HH,mean_5_HH,variance_5_HH,skewness_5_HH,kurtosis_5_HH,lbp_5_HH_0,lbp_5_HH_1,lbp_5_HH_2,lbp_5_HH_3,lbp_5_HH_4,lbp_5_HH_5,lbp_5_HH_6,lbp_5_HH_7,lbp_5_HH_8,lbp_5_HH_9,spectral_energy_5_HH,spectral_entropy_5_HH"
# Convert the string to a list
all_feature_types = all_feature.split(',')

# Define the level weights
level_weights = {
    '_1_': 0.7899,   # Example weight for level '1'
    '_2_': 0.9105,   # Example weight for level '2'
    '_3_': 0.9788,   # Example weight for level '3'
    '_4_': 0.9989,   # Example weight for level '4'
    '_5_': 1    # Example weight for level '5'
}

def load_data(dataset_name):

    file_path = f'E:/ProjetDoctorat/Workspace/WTD/results/{dataset_name}_features_scaled.csv'
    df = pd.read_csv(file_path)

    # Handle missing values (if any)
    df.fillna(df.mean(), inplace=True)
    
    # Separate features and labels
    feature_names = df.columns[:-1]  # Exclude the label column
    labels = df['label']
    
    return df[feature_names], feature_names, labels

def apply_level_weights(df, feature_names, level_weights):

    weighted_df = df.copy()
    for feature in feature_names:
        for level, weight in level_weights.items():
            if level in feature:
                weighted_df[feature] *= weight
    return weighted_df

def train(features, labels, fold=5):

    #features = features.values
    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # SVM classifier
    model = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=4)
    accuracies = []

    for train_index, test_index in cv.split(features_scaled, labels):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies)

def train_grid_search(features, labels, fold=5):

    #features = features.values
    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    param_grid = {
        'C': [1, 10, 100],  # More options: 0.01, 0.1, 1, 10, 100, 1000
        'gamma': [0.001, 0.01, 0.1, 'scale'],  # More options: 0.0001, 0.001, 0.01, 0.1, 1, 10
    }
    # SVM classifier
    svm = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=4)
    # Setup Grid Search
    grid_search = GridSearchCV(svm, param_grid, cv = cv, n_jobs=-1)
    
    # Fit Grid Search
    grid_search.fit(features_scaled, labels)
    # Best estimator with optimized parameters
    model = grid_search.best_estimator_

    accuracies = []

    for train_index, test_index in cv.split(features_scaled, labels):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies)


def remove_feature(feature, features):

    feature_columns = [col for col in features.columns if feature in col]
    new_features = features.drop(columns=feature_columns)
    return new_features

def update_features_type(feature, feature_types):

    return [ft for ft in feature_types if ft != feature]

def rfe(features, labels, feature_types, iteration_number, fold=5, grid_search=False):
    print("\n-----------------------------------------------")
    print(f"- Iteration: {iteration_number}")
    if grid_search:
        combined_features_acc = train_grid_search(features, labels, fold)
    else:
        combined_features_acc = train(features, labels, fold)
    print(f"Accuracy with all features: {combined_features_acc*100:.2f}")
    best_acc = combined_features_acc
    removed_feature = None
    acc =0

    for feature in feature_types:
        current_features = remove_feature(feature, features)
        if grid_search:
            acc = train_grid_search(current_features.values, labels, fold)
        else:
            acc = train(current_features.values, labels, fold)
        print(f"Accuracy after removing {feature}: {acc*100:.2f} ({(acc - combined_features_acc)*100:.4f}) ")
        if acc >= best_acc:
            best_acc = acc
            removed_feature = feature

    if removed_feature:
        print(f"- Removing feature: {removed_feature} improved accuracy to: {best_acc*100:.2f} ({(best_acc - combined_features_acc)*100:.2f}) ")
        new_features = remove_feature(removed_feature, features)
        new_feature_types = update_features_type(removed_feature, feature_types)
        iteration_number += 1
        return rfe(new_features, labels, new_feature_types, iteration_number ,fold, grid_search)
    else:
        return best_acc, feature_types, features
    
def isolate_level_features(df, level = '_1_'):
    # Extract LL sub-band features
    ll_features = ['mean_LL', 'variance_LL', 'skewness_LL', 'kurtosis_LL']
    ll_features_updated = []
    for feature in ll_features:
        if feature in df.columns:
            ll_features_updated.append(feature)
    
    # Extract first level detail coefficients (LH, HL, HH)
    level_features = [col for col in df.columns if level in col]
    
    # Combine LL features and first level detail coefficient features
    level_features = ll_features_updated + level_features
    
    # Isolate the features from the DataFrame
    df_level = df[level_features]

   
    return df_level

def aggregate_features(df, feature_types):
    aggregated_df = pd.DataFrame()
    for feature_type in feature_types:
        cols = [col for col in df.columns if feature_type in col]
        if cols:
            # Create a new column for the aggregated feature
            aggregated_df[f'aggregate_{feature_type}'] = df[cols].mean(axis=1)
    return aggregated_df

# Assuming `aggregated_df` is your DataFrame with aggregated features
def plot_correlation_matrix(aggregated_df):
    # Calculate the correlation matrix
    corr_matrix = aggregated_df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
    
    # Add title
    plt.title('Correlation Matrix for Aggregated Features', fontsize=16)
    
    # Show plot
    plt.show()

def save_selected_features(features, labels, dataset_name):

    # Add labels to the DataFrame
    features['label'] = labels

    # Save the DataFrame to a file (e.g., CSV)
    features.to_csv(f'WTD/results/{dataset_name}_selected_features.csv', index=False)

def load_selected_features(dataset_name):

    # Load the selected features DataFrame
    selected_features = pd.read_csv(f'E:/ProjetDoctorat/Workspace/WTD/results/{dataset_name}_selected_features.csv')

    # Separate features and labels
    feature_names = selected_features.columns[:-1]  # Exclude the label column
    labels = selected_features['label']
    
    return selected_features[feature_names], feature_names, labels

def select_features(df, features_types):
    selected_columns = []
    for feature in features_types:
        selected_columns.extend([col for col in df.columns if feature in col])
    new_features = df[selected_columns]
    return new_features

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded_output = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded_output)
    encoder = Model(input_layer, encoded_output)
    
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder, encoder

if __name__ == '__main__':

    features_types = ['contrast', 'correlation', 'lbp']

    dataset_name = 'X-SDD'
    df, feature_names, labels = load_data(dataset_name)
    iteration_number = 1
    fold = 10


    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df.values)

    # Build and train the deep autoencoder
    input_dim = features_scaled.shape[1]
    encoding_dim = 32  # Dimension to reduce to, adjust based on your needs
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)

    autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2)

    # Use the encoder to transform the features
    encoded_features = encoder.predict(features_scaled)

    # Train and evaluate the model using the reduced features
    acc = train(encoded_features, labels, fold)
    print("------------------------------------------")
    print(" Evaluate with Deep Autoencoder with selected features:")
    print("> Dataset_name: ", dataset_name)
    print(f"Accuracy: {acc*100:.2f}")



