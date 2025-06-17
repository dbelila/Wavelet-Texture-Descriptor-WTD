import time
from matplotlib import pyplot as plt
import pandas as pd

import cv2
from scipy.stats import skew, kurtosis
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from data_preprocessing import get_data
from wavelet_decomposition import wavelet_decompose

def extract_features(coeffs):
    features = []
    LL, *detail_coeffs = coeffs  # Separate LL from other detail coefficients

    # Flatten the LL (approximation) coefficients and include them in feature extraction
    LL_flat = LL.flatten()
    features.append(np.mean(LL_flat))
    features.append(np.var(LL_flat))
    features.append(skew(LL_flat))
    features.append(kurtosis(LL_flat))

    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
   
    # Process each detail coefficients tuple (LH, HL, HH)
    for LH, HL, HH in detail_coeffs:
        for coeff in [LH, HL, HH]:

            coeff_flat = coeff.flatten()

            # Prepare the coefficient for GLCM calculations
            # Scale to the range [0, 255] as required by greycomatrix
            coeff_rescaled = ((coeff_flat - coeff_flat.min()) / 
                              (coeff_flat.ptp() + 1e-12)) * 255
            coeff_rescaled = coeff_rescaled.astype('uint8').reshape(coeff.shape)
            glcm = graycomatrix(coeff_rescaled, distances, angles, 256, symmetric=True, normed=True)
            
            # GLCM features
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            features.extend([contrast, dissimilarity, homogeneity, energy, correlation])

            # Statistical features
            features.append(np.mean(coeff_flat))
            features.append(np.var(coeff_flat))
            features.append(skew(coeff_flat))
            features.append(kurtosis(coeff_flat))

            # Frequency features
            psd1D = np.abs(np.fft.fft(coeff_flat))**2
            psd1D /= len(psd1D)  # Normalize
            spectral_energy = np.sum(psd1D)
            spectral_entropy = -np.sum((psd1D / spectral_energy) * np.log2(psd1D / spectral_energy + 1e-12))

            features.append(spectral_energy)
            features.append(spectral_entropy)
    
    # Convert the list of features to a numpy array
    feature_vector = np.array(features)

    return feature_vector

def wt_choice_features(coeffs):
    features = []
    LL, *detail_coeffs = coeffs  # Separate LL from other detail coefficients

    # Flatten the LL (approximation) coefficients and include them in feature extraction
    LL_flat = LL.flatten()
    #features.append(np.mean(LL_flat))
    features.append(np.var(LL_flat))
    #features.append(skew(LL_flat))
    #features.append(kurtosis(LL_flat))

    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
   
    # Process each detail coefficients tuple (LH, HL, HH)
    for LH, HL, HH in detail_coeffs:
        for coeff in [LH, HL, HH]:

            coeff_flat = coeff.flatten()

            # Prepare the coefficient for GLCM calculations
            # Scale to the range [0, 255] as required by greycomatrix
            coeff_rescaled = ((coeff_flat - coeff_flat.min()) / 
                              (coeff_flat.ptp() + 1e-12)) * 255
            coeff_rescaled = coeff_rescaled.astype('uint8').reshape(coeff.shape)
            glcm = graycomatrix(coeff_rescaled, distances, angles, 256, symmetric=True, normed=True)
            
            # GLCM features
            #contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            #homogeneity = graycoprops(glcm, 'homogeneity').mean()
            #energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            #features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
            features.extend([dissimilarity, correlation])

            # Statistical features
            #features.append(np.mean(coeff_flat))
            features.append(np.var(coeff_flat))
            #features.append(skew(coeff_flat))
            #features.append(kurtosis(coeff_flat))

            '''# Frequency features
            psd1D = np.abs(np.fft.fft(coeff_flat))**2
            psd1D /= len(psd1D)  # Normalize
            spectral_energy = np.sum(psd1D)
            spectral_entropy = -np.sum((psd1D / spectral_energy) * np.log2(psd1D / spectral_energy + 1e-12))

            features.append(spectral_energy)
            features.append(spectral_entropy)'''
    
    # Convert the list of features to a numpy array
    feature_vector = np.array(features)

    return feature_vector

def features_selection(coeffs):
    features = []
    feature_names = []
    LL, *detail_coeffs = coeffs  # Separate LL from other detail coefficients

    # Flatten the LL (approximation) coefficients and include them in feature extraction
    LL_flat = LL.flatten()
    features.extend([np.mean(LL_flat), np.var(LL_flat), skew(LL_flat), kurtosis(LL_flat)])
    feature_names.extend(["mean_LL", "variance_LL", "skewness_LL", "kurtosis_LL"])

    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
   
    # Process each detail coefficients tuple (LH, HL, HH)
    for level, (LH, HL, HH) in enumerate(detail_coeffs, start=1):
        for band, coeff in zip(["LH", "HL", "HH"], [LH, HL, HH]):
            coeff_flat = coeff.flatten()

            # Prepare the coefficient for GLCM calculations
            # Scale to the range [0, 255] as required by greycomatrix
            coeff_rescaled = ((coeff_flat - coeff_flat.min()) / 
                              (coeff_flat.ptp() + 1e-12)) * 255
            coeff_rescaled = coeff_rescaled.astype('uint8').reshape(coeff.shape)
            glcm = graycomatrix(coeff_rescaled, distances, angles, 256, symmetric=True, normed=True)
            
            # GLCM features
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
            feature_names.extend([
                f"contrast_{level}_{band}",
                f"dissimilarity_{level}_{band}",
                f"homogeneity_{level}_{band}",
                f"energy_{level}_{band}",
                f"correlation_{level}_{band}"
            ])

            # Statistical features
            features.extend([np.mean(coeff_flat), np.var(coeff_flat), skew(coeff_flat), kurtosis(coeff_flat)])
            feature_names.extend([
                f"mean_{level}_{band}",
                f"variance_{level}_{band}",
                f"skewness_{level}_{band}",
                f"kurtosis_{level}_{band}"
            ])

            # Local Binary Pattern (LBP)
            lbp = local_binary_pattern(coeff_rescaled, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, lbp.max() + 2), range=(0, lbp.max() + 1), density=True)
            features.extend(lbp_hist)
            feature_names.extend([f"lbp_{level}_{band}_{i}" for i in range(len(lbp_hist))])

            # Frequency features
            psd1D = np.abs(np.fft.fft(coeff_flat))**2
            psd1D /= len(psd1D)  # Normalize
            spectral_energy = np.sum(psd1D)
            spectral_entropy = -np.sum((psd1D / spectral_energy) * np.log2(psd1D / spectral_energy + 1e-12))

            features.extend([spectral_energy, spectral_entropy])
            feature_names.extend([f"spectral_energy_{level}_{band}", f"spectral_entropy_{level}_{band}"])
    
    # Convert the list of features to a numpy array
    feature_vector = np.array(features)

    return feature_vector, feature_names

def extract_and_save_features(dataset_name, wavelet='haar', level=1):
    # Load dataset
    data, labels, target_dict = get_data(dataset_name)

    # Convert one-hot encoded labels to 1D array of class indices
    labels = np.argmax(labels, axis=1)
  
    # Feature extraction
    first_image_features, feature_names = features_selection(wavelet_decompose(data[0], wavelet=wavelet, level=level))
    feature_vectors = [first_image_features]  # the first image's features
    
    for image in data[1:]:  # Process the rest of the images
        features, _ = features_selection(wavelet_decompose(image, wavelet=wavelet, level=level))
        feature_vectors.append(features)
    
    # Convert to numpy array for scaling
    feature_vectors = np.array(feature_vectors)

    # Print the shape of the feature vectors and feature names
    print(f"Feature vectors shape: {feature_vectors.shape}")
    print(f"Number of feature names: {len(feature_names)}")

    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_vectors)

    # Check the shape after scaling
    print(f"Scaled feature vectors shape: {features_scaled.shape}")

    # Create a DataFrame with feature names as columns
    features_df = pd.DataFrame(features_scaled, columns=feature_names)

    # Add labels to the DataFrame
    features_df['label'] = labels

    # Save the DataFrame to a file (e.g., CSV)
    features_df.to_csv(f'WTD/results/{dataset_name}_features_scaled.csv', index=False)


def load_data(dataset_name, file_format='csv'):

    if file_format == 'csv':
        file_path = f'WTD/results/{dataset_name}_features_scaled.csv'
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'h5'.")

    # Handle missing values (if any)
    df.fillna(df.mean(), inplace=True)
    
    # Separate features and labels
    feature_names = df.columns[:-1]  # Exclude the label column
    labels = df['label']
    
    return df, feature_names, labels


# Example usage:
if __name__ == '__main__':
    
    dataset_name = 'test'
    level=5

    dataset_name = 'X-SDD-20'
    extract_and_save_features(dataset_name, level=level)

    dataset_name = 'NEU-CLS-20'
    extract_and_save_features(dataset_name, level=level)

    print("Features extracted and saved successfully!")
    '''
    # Example usage
    df = load_data(dataset_name, file_format='csv')

    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()[:5]}...")  # Display first 5 column names as a sample
    print(df.head())  # Display the first few rows of the DataFrame
    '''
    
    
