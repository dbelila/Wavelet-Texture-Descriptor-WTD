import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

from wavelet_decomposition import wavelet_decompose
from features_extraction import extract_features
from data_preprocessing import get_data

def svm_wt_model(dataset_name, n_folds=3, wavelet='haar', level=2):
    # Load dataset
    data, labels, target_dict = get_data(dataset_name)

    # Convert one-hot encoded labels to 1D array of class indices
    labels = np.argmax(labels, axis=1)

    # Feature extraction
    feature_vectors = np.array([extract_features(wavelet_decompose(image, wavelet=wavelet, level=level))
                                                                for image in data])

    print("Feature_vectors shape : ", feature_vectors.shape)

    # Reverse the target_dict so the class indices are the keys to use to identify misclassified data
    target_dict = {v: k for k, v in target_dict.items()}

    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_vectors)

    # Parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],  # More options: 0.01, 0.1, 1, 10, 100, 1000
        'gamma': [0.0001, 0.001, 0.01, 0.1, 'scale'],  # More options: 0.0001, 0.001, 0.01, 0.1, 1, 10
    }

    # SVM classifier
    svm = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4)

    # Setup Grid Search
    grid_search = GridSearchCV(svm, param_grid, cv = cv)
    
    # Fit Grid Search
    grid_search.fit(features_scaled, labels)

    # Best estimator with optimized parameters
    best_svm = grid_search.best_estimator_
    fold_no = 1

    train_acc_scores = []
    val_acc_scores = []
    train_loss_scores = []
    val_loss_scores = []
    val_precision_scores = []
    val_recall_scores = []
    val_f1_scores = []
    misclassified_info = []

    for train_index, test_index in cv.split(features_scaled, labels):

        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the model
        best_svm.fit(X_train, y_train)
        
        # Training scores
        train_prob_predictions = best_svm.predict_proba(X_train)
        train_predictions = best_svm.predict(X_train)

        train_acc = accuracy_score(y_train, train_predictions)
        train_loss = log_loss(y_train, train_prob_predictions)
        train_acc_scores.append(train_acc)
        train_loss_scores.append(train_loss)
        
        # Validation scores
        val_prob_predictions = best_svm.predict_proba(X_test)
        val_predictions = best_svm.predict(X_test)

        # Check for misclassifications
        misclassified_indices = np.where(val_predictions != y_test)[0]
        for index in misclassified_indices:
            misclassified_info.append({
                'image': data[test_index[index]], # originale image
                'predicted': target_dict[val_predictions[index]], # predicted class
                'true': target_dict[y_test[index]], # true class
                'fold': fold_no
            })

        val_acc = accuracy_score(y_test, val_predictions)
        val_loss = log_loss(y_test, val_prob_predictions)
        val_precision = precision_score(y_test, val_predictions, average='macro')
        val_recall = recall_score(y_test, val_predictions, average='macro')
        val_f1 = f1_score(y_test, val_predictions, average='macro')


        val_acc_scores.append(val_acc)
        val_loss_scores.append(val_loss)
        val_precision_scores.append(val_precision)
        val_recall_scores.append(val_recall)
        val_f1_scores.append(val_f1)

        fold_no += 1

    return (grid_search.best_params_,
             train_acc_scores,
             val_acc_scores,
             train_loss_scores,
             val_loss_scores,
             val_precision_scores,
             val_recall_scores,
             val_f1_scores,
             misclassified_info)


def plot_results(train_acc_scores, val_acc_scores, train_loss_scores, val_loss_scores):
    epochs = range(1, len(train_acc_scores) + 1)
    
    plt.figure(figsize=(8, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_scores, label='Training acc')
    plt.plot(epochs, val_acc_scores, label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_scores, label='Training loss')
    plt.plot(epochs, val_loss_scores, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def plot_misclassified(dataset_name, level, misclassified_info, cols=3, image_size=(4,4)):
    rows = len(misclassified_info) // cols + 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*image_size[0], rows*image_size[1]))
    for i, misclassified in enumerate(misclassified_info):
        ax = axs[i // cols, i % cols]
        ax.imshow(misclassified['image'], cmap='gray')
        ax.set_title(f'Predicted: {misclassified["predicted"]}\n'
                     f'True: {misclassified["true"]}\n'
                     f'Fold: {misclassified["fold"]}')
        ax.axis('off')

    # If there are any remaining empty subplots, turn their axes off
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # save the figure to a file
    image_name = f'WTD/results/{dataset_name}_svm_decomp_{level}_missed_images.png'

    plt.savefig(image_name, dpi=300)

    plt.tight_layout()
    plt.show()

def print_results( dataset_name, level, best_params, acc_scores, loss_scores, 
                  precision_scores, recall_scores, f1_scores):
    # Print the average scores
    print('----------------------------------------------------------------')
    print('Score per fold')
    for i in range(len(acc_scores)):
        print('-------------------------------------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Accuracy: {acc_scores[i]*100:.2f} % - Loss: {loss_scores[i]:.4f}'
              f' - Precision: {precision_scores[i]*100:.2f} %'
              f' - Recall: {recall_scores[i]*100:.2f} %' 
              f' - F1 Score: {f1_scores[i]*100:.2f} %')

    print('-------------------------------------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print('- Dataset name : ', dataset_name)
    print('- Level of decomposition : ', level)
    print('- SVM parameters: ', best_params)
    print(f'> Accuracy: {np.mean(acc_scores)*100:.2f} (+- {np.std(acc_scores)*100:.2f})')
    print(f'> Loss: {np.mean(loss_scores):.4f}')
    print(f'> Precision: {np.mean(precision_scores)*100:.2f} (+- {np.std(precision_scores)*100:.2f})')
    print(f'> Recall: {np.mean(recall_scores)*100:.2f} (+- {np.std(recall_scores)*100:.2f})')
    print(f'> F1 Score: {np.mean(f1_scores)*100:.2f} (+- {np.std(f1_scores)*100:.2f})')
    print('----------------------------------------------------------------')

if __name__ == '__main__':
    dataset_name = 'X-SDD'
    n_folds = 10
    wavelet = 'haar'
    level = 4
    results = svm_wt_model(dataset_name = dataset_name,
                           n_folds = n_folds,
                           wavelet = wavelet,
                           level = level)

    (best_params, train_acc_scores, val_acc_scores, train_loss_scores, val_loss_scores, 
     precision_scores, recall_scores, f1_scores, misclassified_info) = results
    
    print_results(dataset_name, level, best_params, val_acc_scores, val_loss_scores, 
                  precision_scores, recall_scores, f1_scores)
    
    plot_results(train_acc_scores, val_acc_scores, train_loss_scores, val_loss_scores)

    plot_misclassified(dataset_name, level, misclassified_info)

