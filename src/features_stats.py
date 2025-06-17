from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from features_extraction import load_data

# Define the types of features to analyze
feature_types = ['mean', 'variance', 'skewness', 'kurtosis', 
                 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 
                 'spectral_energy', 'spectral_entropy']

def aggregate_features(df, feature_types):
    for feature_type in feature_types:
        cols = [col for col in df.columns if feature_type in col]
        if cols:
            # Create a new column for the aggregated feature
            df[f'aggregate_{feature_type}'] = df[cols].mean(axis=1)
    return df

def compute_aggregated_correlation(df, feature_types):
    # Select only the aggregated columns for correlation
    aggregated_columns = [f'aggregate_{feature_type}' for feature_type in feature_types if f'aggregate_{feature_type}' in df.columns]
    correlation_matrix = df[aggregated_columns].corr()
    return correlation_matrix

def compute_aggregate_stats(df, feature_types):
    aggregate_stats = {}
    for feature_type in feature_types:
        cols = [col for col in df.columns if feature_type in col]
        aggregate_stats[feature_type] = df[cols].mean(axis=1).describe()
    return pd.DataFrame(aggregate_stats)

def isolate_level_features(df, level = '_1_', aggregate = False):
    # Extract LL sub-band features
    ll_features = ['mean_LL', 'variance_LL', 'skewness_LL', 'kurtosis_LL']
    
    # Extract first level detail coefficients (LH, HL, HH)
    level_features = [col for col in df.columns if level in col and 'lbp' not in col]
    
    # Combine LL features and first level detail coefficient features
    level_features = ll_features + level_features
    
    # Isolate the features from the DataFrame
    df_level = df[level_features]

    if aggregate:
        df_level = aggregate_features(df_level, feature_types)
        aggregated_columns = [f'aggregate_{feature_type}' for feature_type in feature_types if f'aggregate_{feature_type}' in df_level.columns]
        df_level = df_level[aggregated_columns]
    
    return df_level

def plot_aggregate_distributions(df, feature_types):
    for feature_type in feature_types:
        cols = [col for col in df.columns if feature_type in col]
        if cols:
            # Aggregate across all columns of the same feature type
            df[f'aggregate_{feature_type}'] = df[cols].mean(axis=1)
            
            plt.figure(figsize=(15, 5))
            
            # Histogram
            plt.subplot(1, 3, 1)
            df[f'aggregate_{feature_type}'].hist(bins=30)
            plt.title(f'{feature_type} Histogram')
            
            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(df[f'aggregate_{feature_type}'])
            plt.title(f'{feature_type} Boxplot')
            
            # Density Plot
            plt.subplot(1, 3, 3)
            df[f'aggregate_{feature_type}'].plot(kind='density')
            plt.title(f'{feature_type} Density Plot')
            
            plt.tight_layout()
            plt.show()

def plot_aggregate_lbp_distributions(df):
    lbp_features = [col for col in df.columns if 'lbp' in col]
    if lbp_features:
        df['aggregate_lbp'] = df[lbp_features].mean(axis=1)
        
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(1, 3, 1)
        df['aggregate_lbp'].hist(bins=30)
        plt.title('LBP Histogram')
        
        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(df['aggregate_lbp'])
        plt.title('LBP Boxplot')
        
        # Density Plot
        plt.subplot(1, 3, 3)
        df['aggregate_lbp'].plot(kind='density')
        plt.title('LBP Density Plot')
        
        plt.tight_layout()
        plt.show()


def plot_correlation(correlation_matrix):

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix for First Level Features')
    plt.show()


if __name__ == '__main__':

    dataset_name = 'test'

    # Example usage
    df, feature_names, labels = load_data(dataset_name)

    # Isolate first level features
    df_level = isolate_level_features(df, level = '_1_', aggregate=True)

    aggregate_level_stats = df_level.describe()
    # Save to CSV file named 'aggregate_stats.csv'
    aggregate_level_stats.to_csv(f'WTD/results/{dataset_name}_level_1_aggregate_stats.csv')

    # Plot correlation matrix
    correlation_matrix = compute_aggregated_correlation(df_level, feature_types)

    plot_correlation(correlation_matrix)

    # Aggregate features
    #df = aggregate_features(df, feature_types)

    '''# Compute aggregate descriptive statistics
    aggregate_stats = compute_aggregate_stats(df, feature_types)
    print("Aggregate Descriptive Statistics:")
    print(aggregate_stats)
    # Save to CSV file named 'aggregate_stats.csv'
    aggregate_stats.to_csv(f'WTD/results/{dataset_name}_aggregate_stats.csv')'''

    # Plot aggregate distributions
    #plot_aggregate_distributions(df, feature_types)

    # Plot aggregate LBP distributions
    #plot_aggregate_lbp_distributions(df)