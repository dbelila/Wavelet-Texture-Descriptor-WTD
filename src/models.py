
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from features_analysis import load_data, select_features


features_types_neu = ['mean', 'variance', 'dissimilarity', 'correlation']

features_types_xsdd = ['mean', 'variance', 'skewness', 'contrast', 'dissimilarity', 
                       'homogeneity', 'correlation', 'lbp', 'spectral_entropy']
def set_random_seeds(seed=42):
    np.random.seed(seed)

def svm_model(features, labels, n_folds=3):

    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],  # More options: 0.01, 0.1, 1, 10, 100, 1000
        'gamma': [0.001, 0.01, 0.1, 'scale'],  # More options: 0.0001, 0.001, 0.01, 0.1, 1, 10
    }

    # SVM classifier
    svm = SVC(kernel='rbf', probability=True)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4)

    # Setup Grid Search
    grid_search = GridSearchCV(svm, param_grid, cv = cv, n_jobs=-1)
    
    # Fit Grid Search
    grid_search.fit(features_scaled, labels)

    # Best estimator with optimized parameters
    best_svm = grid_search.best_estimator_

    val_acc_scores = []
    val_loss_scores = []
    val_precision_scores = []
    val_recall_scores = []
    val_f1_scores = []
    all_confusion_matrices = []

    for train_index, test_index in cv.split(features_scaled, labels):

        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the model
        best_svm.fit(X_train, y_train)
        
        # Validation scores
        val_prob_predictions = best_svm.predict_proba(X_test)
        val_predictions = best_svm.predict(X_test)

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

        # Compute confusion matrix
        cm = confusion_matrix(y_test, val_predictions)
        all_confusion_matrices.append(cm)

    return (grid_search.best_params_,
             val_acc_scores,
             val_loss_scores,
             val_precision_scores,
             val_recall_scores,
             val_f1_scores, 
             all_confusion_matrices)

def average_confusion_matrices(confusion_matrices):
    return np.mean(confusion_matrices, axis=0)

def plot_confusion_matrix(cm, class_names,dataset_name=None, normalize=False):

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize the confusion matrix
    sns.heatmap(cm, annot=True, fmt=".2f" , cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, annot_kws= {'size': 14})
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)

    plt.tight_layout()
    plt.savefig(f'E:/ProjetDoctorat/Workspace/WTD/results/confusion_matrix_{dataset_name}_1.png', dpi=300)
    plt.show()

def random_forest_model(features, labels, n_folds=3):
    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [200, 250, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # RandomForest classifier
    rf = RandomForestClassifier(random_state=42)

    # Run k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Setup Grid Search
    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)
    
    # Fit Grid Search
    grid_search.fit(features_scaled, labels)

    # Best estimator with optimized parameters
    best_rf = grid_search.best_estimator_

    val_acc_scores = []
    val_loss_scores = []
    val_precision_scores = []
    val_recall_scores = []
    val_f1_scores = []

    for train_index, test_index in cv.split(features_scaled, labels):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the model
        best_rf.fit(X_train, y_train)
        
        # Validation scores
        val_prob_predictions = best_rf.predict_proba(X_test)
        val_predictions = best_rf.predict(X_test)

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

    return (grid_search.best_params_,
            val_acc_scores,
            val_loss_scores,
            val_precision_scores,
            val_recall_scores,
            val_f1_scores)


def mlp_classifier(input_shape, num_classes, dropout_rate=0):
   
    # Input layer
    input_layer = Input(shape=input_shape,)  # Feature vector size

    # Hidden layers
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def mlp_model(features, labels, n_folds=3, batch_size=32, epochs=10, dropout_rate=0):

    set_random_seeds()
    features = np.array(features, dtype=np.float32)
    # Normalize the feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert labels to single class predictions
    #class_labels = np.argmax(labels, axis=1)
    num_classes = len(np.unique(labels))

    # Convert labels to categorical if they aren't already
    labels_categorical = to_categorical(labels, num_classes=num_classes)

    # Define StratifiedKFold cross validator
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)

    # Configuring EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',         # Metric to monitor
        patience=10,                # Number of epochs with no improvement after which training will be stopped
        verbose=1,                  # Log level
        restore_best_weights=True   # Restore model weights from the epoch with the best value of the monitored quantity
    )

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    recall_per_fold = []
    precision_per_fold = []
    f1_per_fold = []

    # Initialize lists to store the fold-wise histories
    fold_histories = []

    for train_indices, test_indices in kfold.split(features_scaled, labels):

        model = mlp_classifier(features_scaled[0].shape, num_classes, dropout_rate=dropout_rate)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #print(f'Training for fold {fold_no} ...')

        history = model.fit(features_scaled[train_indices], labels_categorical[train_indices],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(features_scaled[test_indices], labels_categorical[test_indices]),
                            callbacks=[early_stopping]) # Early stopping
        
        # Store the history
        fold_histories.append(history.history)

        # Evaluate the model
        predictions = model.predict(features_scaled[test_indices], verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = labels[test_indices]


        # Calculate precision, recall, and F1-score
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        recall_per_fold.append(recall)
        precision_per_fold.append(precision)
        f1_per_fold.append(f1)

        # Generate generalization metrics
        scores = model.evaluate(features_scaled[test_indices], labels_categorical[test_indices], verbose=0)
        #print(f'Score for fold {fold_no}: {model.metrics_names[0]} : {scores[0]:.4f}; {model.metrics_names[1]} : {scores[1]*100:.2f} %')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no += 1
        # Break the loop after the first fold to save time
        if fold_no > n_folds:
            break
    
    model.summary()

    return (acc_per_fold, loss_per_fold, recall_per_fold, precision_per_fold, f1_per_fold, fold_histories)


def print_results( dataset_name,  acc_scores, loss_scores, 
                  precision_scores, recall_scores, f1_scores, best_params=None):
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
    if best_params:
        print('- Classifier parameters: ', best_params)
    print(f'> Accuracy: {np.mean(acc_scores)*100:.2f} (+- {np.std(acc_scores)*100:.2f})')
    print(f'> Loss: {np.mean(loss_scores):.4f}')
    print(f'> Precision: {np.mean(precision_scores)*100:.2f} (+- {np.std(precision_scores)*100:.2f})')
    print(f'> Recall: {np.mean(recall_scores)*100:.2f} (+- {np.std(recall_scores)*100:.2f})')
    print(f'> F1 Score: {np.mean(f1_scores)*100:.2f} (+- {np.std(f1_scores)*100:.2f})')
    print('----------------------------------------------------------------')


def plot_results(fold_histories):
    # calculate the average of the metrics per epoch across all folds
    average_history = {}
    num_epochs = len(fold_histories[0]['accuracy'])
    for metric in fold_histories[0].keys():
        average_history[metric] = [np.mean([x[metric][epoch] for x in fold_histories]) for epoch in range(num_epochs)]

    # Plotting
    plt.figure(figsize=(8, 4))

    # Plot average training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), average_history['accuracy'], label='Average Training Accuracy')
    plt.plot(range(1, num_epochs + 1), average_history['val_accuracy'], label='Average Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot average training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), average_history['loss'], label='Average Training Loss')
    plt.plot(range(1, num_epochs + 1), average_history['val_loss'], label='Average Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    '''
    dataset_name = 'test'
    fold = 10

    df, feature_names, labels = load_data(dataset_name)
    selected_features = select_features(df, features_types)

    results = random_forest_model(selected_features.values, labels, n_folds=fold)
    best_params, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores = results

    print_results(dataset_name, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores, best_params=best_params)

    '''

    dataset_name = 'NEU-CLS'
    fold = 10
    batch_size = 16
    epochs= 15
    dropout_rate = 0.5

    
    df, feature_names, labels = load_data(dataset_name)

    selected_features = select_features(df, features_types_neu)

    results = mlp_model(selected_features.values, labels, n_folds=fold, 
                        batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)
    
    acc_scores, loss_scores, recall_scores, precision_scores, f1_scores, fold_histories = results
    print_results(dataset_name, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores)
    plot_results(fold_histories)
    

    '''
    dataset_name = 'X-SDD'
    fold = 10

    df, feature_names, labels = load_data(dataset_name)
    selected_features = select_features(df, features_types)

    results = random_forest_model(selected_features.values, labels, n_folds=fold)
    best_params, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores = results

    print_results(dataset_name, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores, best_params=best_params)
    '''




    '''
    dataset_name = 'X-SDD'
    fold = 10

    df, feature_names, labels = load_data(dataset_name)

    selected_df = select_features(df, features_types)
    results = svm_model(selected_df.values, labels, fold)


    best_params, acc_scores, loss_scores, precision_scores, recall_scores, f1_scores = results

    print_results(dataset_name, acc_scores, loss_scores, 
                  precision_scores, recall_scores, f1_scores, best_params=best_params)'''