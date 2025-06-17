from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

from wavelet_decomposition import wavelet_decompose
from features_extraction import wt_choice_features
from data_preprocessing import get_data

def wt_choice_model(dataset_name, n_folds=3, wavelet='haar', level=2):
    # Load dataset
    data, labels, target_dict = get_data(dataset_name)

    # Convert one-hot encoded labels to 1D array of class indices
    labels = np.argmax(labels, axis=1)

    # Feature extraction
    feature_vectors = np.array([wt_choice_features(wavelet_decompose(image, wavelet=wavelet, level=level)) 
                                                                    for image in data])

    print("Feature_vectors shape : ", feature_vectors.shape)

    # Reverse the target_dict so the class indices are the keys to use to identify misclassified data
    target_dict = {v: k for k, v in target_dict.items()}

    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_vectors)

    # SVM classifier
    svm = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4)

    fold_no = 1

    val_acc_scores = []
    val_loss_scores = []

    for train_index, test_index in cv.split(features_scaled, labels):

        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the model
        svm.fit(X_train, y_train)
        
        # Validation scores
        val_prob_predictions = svm.predict_proba(X_test)
        val_predictions = svm.predict(X_test)

        val_acc = accuracy_score(y_test, val_predictions)
        val_loss = log_loss(y_test, val_prob_predictions)

        val_acc_scores.append(val_acc)
        val_loss_scores.append(val_loss)

        fold_no += 1

    return (val_acc_scores, val_loss_scores)

def print_results( dataset_name, level, acc_scores, loss_scores):
    # Print the average scores
    print('--------------------------------------------')
    print('Score per fold')
    for i in range(len(acc_scores)):
        print('--------------------------------------------')
        print(f'> Fold {i+1} - Accuracy: {acc_scores[i]*100:.2f} % - Loss: {loss_scores[i]:.4f}')
    print('--------------------------------------------')
    print('Average scores for all folds:')
    print('- Dataset name : ', dataset_name)
    print('- Level of decomposition : ', level)
    print(f'> Accuracy: {np.mean(acc_scores)*100:.2f} (+- {np.std(acc_scores)*100:.2f})')
    print(f'> Loss: {np.mean(loss_scores):.4f}')
    print('--------------------------------------------')

if __name__ == '__main__':
    dataset_name = 'X-SDD'
    n_folds = 10
    wavelet = 'haar'
    level = 1
    val_acc_scores, val_loss_scores = wt_choice_model(dataset_name = dataset_name,
                           n_folds = n_folds,
                           wavelet = wavelet,
                           level = level)

    print_results(dataset_name, level, val_acc_scores, val_loss_scores)