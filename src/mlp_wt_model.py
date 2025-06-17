from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from features_extraction import extract_features
from wavelet_decomposition import wavelet_decompose
from data_preprocessing import get_data

def mlp_lw_model(input_shape, num_classes, dropout_rate=0):
   
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

def train(dataset_name='X-SDD', n_folds=3, wavelet='haar',level=2,batch_size=32, epochs=10, dropout_rate=0):
    # Load dataset
    data, labels, target_dict = get_data(dataset_name) # labels are one hot encoded

    num_classes = labels.shape[1]
    # Feature extraction
    feature_vectors = np.array([extract_features(wavelet_decompose(image, 
                                                                   wavelet = wavelet,
                                                                   level = level)) for image in data])
    print("Feature length: ", feature_vectors[0].shape)

    # Reverse the target_dict so the class indices are the keys to use to identify misclassified data
    target_dict = {v: k for k, v in target_dict.items()}

    # Normalize the feature vectors
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_vectors)

    # Convert labels to single class predictions
    class_labels = np.argmax(labels, axis=1)

    # Convert labels to categorical if they aren't already
    labels_categorical = to_categorical(class_labels, num_classes=num_classes)


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
    macro_recall_per_fold = []
    macro_precision_per_fold = []
    macro_f1_per_fold = []
    misclassified_info = []

    # Initialize lists to store the fold-wise histories
    fold_histories = []

    for train_indices, test_indices in kfold.split(scaled_data, class_labels):

        model = mlp_lw_model(scaled_data[0].shape, num_classes, dropout_rate=dropout_rate)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #print(f'Training for fold {fold_no} ...')

        history = model.fit(scaled_data[train_indices], labels_categorical[train_indices],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(scaled_data[test_indices], labels_categorical[test_indices]),
                            callbacks=[early_stopping]) # Early stopping
        
        # Store the history
        fold_histories.append(history.history)

        # Evaluate the model
        predictions = model.predict(scaled_data[test_indices], verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = class_labels[test_indices]


        # Check for misclassifications
        misclassified_indices = np.where(y_pred != y_true)[0]
        for index in misclassified_indices:
            misclassified_info.append({
                'image': data[test_indices[index]],  # Original image
                'predicted': target_dict[y_pred[index]],  # Predicted label
                'true': target_dict[y_true[index]],  # True label
                'fold': fold_no
            })

        # Calculate precision, recall, and F1-score
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        macro_recall_per_fold.append(macro_recall)
        macro_precision_per_fold.append(macro_precision)
        macro_f1_per_fold.append(macro_f1)

        # Generate generalization metrics
        scores = model.evaluate(scaled_data[test_indices], labels_categorical[test_indices], verbose=0)
        #print(f'Score for fold {fold_no}: {model.metrics_names[0]} : {scores[0]:.4f}; {model.metrics_names[1]} : {scores[1]*100:.2f} %')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no += 1
        # Break the loop after the first fold to save time
        if fold_no > n_folds:
            break
    
    model.summary()

    return (acc_per_fold, loss_per_fold, fold_histories, 
            macro_recall_per_fold, macro_precision_per_fold, macro_f1_per_fold,
            misclassified_info)



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

def plot_misclassified(misclassified_info, cols=3, image_size=(3, 3)):
    # plot the misclassified data in one figure
    # Determine the number of rows needed for the subplot
    rows = len(misclassified_info) // cols + int(len(misclassified_info) % cols > 0)

    # Calculate the total figure size
    fig_width = cols * image_size[0]
    fig_height = rows * image_size[1]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Adjust the layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.3)
    
    # Flatten the axes array for easy iteration
    axes = axes.ravel()

    for i, misclassified in enumerate(misclassified_info):
        ax = axes[i]
        img, pred, true = misclassified['image'], misclassified['predicted'], misclassified['true']
        
        # Check if image needs color conversion if it has channels
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()

        # Plot the image
        ax.imshow(img, cmap='gray')  # Use 'gray' colormap for grayscale images
        
        ax.set_title(f'Predicted: {pred}\nTrue: {true}', fontsize=9)
        ax.axis('off')  # Turn off the axis
    
    # If there are any remaining empty subplots, turn their axes off
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Save the figure to a file
    plt.savefig('WTD/data/misclassified_images.png', dpi=300)

    # Display the plot
    plt.show()




'''
def plot_misclassified(misclassified_info, cols=3):

    # plot the misclassified data in one figure
    # Determine the number of rows needed for the subplot
    rows = len(misclassified_info) // cols + int(len(misclassified_info) % cols > 0)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.tight_layout(pad=3.0)
    
    # Flatten the axes array for easy iteration
    axes = axes.ravel()
    
    for i, misclassified in enumerate(misclassified_info):
        ax = axes[i]
        img, pred, true = misclassified['image'], misclassified['predicted'], misclassified['true']
        
        # If the images are in a format that plt.imshow() expects (e.g., (M, N, 3) for RGB)
        ax.imshow(img)
        
        ax.set_title(f'Predicted: {pred}\nTrue: {true}')
        ax.axis('off')  # Turn off the axis
    
    # If there are any remaining empty subplots, turn their axes off
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.show()
'''



def print_results(dataset_name, level,acc_per_fold, loss_per_fold, macro_recall_per_fold, 
                 macro_precision_per_fold, macro_f1_per_fold):
    # Print scores section with the average scores
    print('--------------------------------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(len(acc_per_fold)):
        print('--------------------------------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Accuracy: {acc_per_fold[i]*100:.2f} % - Loss: {loss_per_fold[i]:.4f}'
              f' - Recall: {macro_recall_per_fold[i]*100:.2f} %'
              f' - Precision: {macro_precision_per_fold[i]*100:.2f} %'
              f' - F1: {macro_f1_per_fold[i]*100:.2f} %')
    print('----------------------------------------------------------------')
    print('Average scores for all folds:')
    print('- Dataset name : ', dataset_name)
    print('- Level of decomposition :', level)
    print(f'> Accuracy: {np.mean(acc_per_fold)*100:.2f} % (+- {np.std(acc_per_fold)*100:.2f})')
    print(f'> Loss: {np.mean(loss_per_fold):.4f} (+- {np.std(loss_per_fold):.4f})')
    print(f'> Macro Recall: {np.mean(macro_recall_per_fold)*100:.2f} (+- {np.std(macro_recall_per_fold)*100:.2f})')
    print(f'> Macro Precision: {np.mean(macro_precision_per_fold)*100:.2f} (+- {np.std(macro_precision_per_fold)*100:.2f})')
    print(f'> Macro F1 Score: {np.mean(macro_f1_per_fold)*100:.2f} (+- {np.std(macro_f1_per_fold)*100:.2f})')
    print('----------------------------------------------------------------')


if __name__ == '__main__':
    dataset_name = 'X-SDD'
    n_folds = 10
    wavelet = 'haar'
    level = 1
    batch_size = 32
    epochs= 12
    dropout_rate = 0.2
    (acc_per_fold, loss_per_fold, fold_histories, macro_recall_per_fold, 
     macro_precision_per_fold, macro_f1_per_fold, misclassified_info) = train(dataset_name = dataset_name, 
                                                        n_folds = n_folds, 
                                                        wavelet = wavelet, 
                                                        level = level,
                                                        batch_size = batch_size,
                                                        epochs = epochs,
                                                        dropout_rate = dropout_rate)

    print_results(acc_per_fold, loss_per_fold, macro_recall_per_fold, 
                 macro_precision_per_fold, macro_f1_per_fold)
    plot_results(fold_histories)
    #plot_misclassified(misclassified_info)
