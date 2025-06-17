import os
import sys
import time
import cv2
import numpy as np
from keras.utils import to_categorical

# Ensure the root directory of the project is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_dataset(dataset_name, dataset_resize_sizes):
    resize_size = dataset_resize_sizes.get(dataset_name, (128, 128))  # Default resize_size if not specified
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataset_name)
    print(" data_dir : ", data_dir)
    data_array = []
    class_names = []

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, resize_size)
                image = image.astype('float32') / 255.0
                data_array.append(image)
                class_names.append(folder_name)
            else:
                print(f"Warning: Unable to read file {image_path}. It's being skipped.")
                
    return np.array(data_array), np.array(class_names)

def preprocess_data(data, labels):
    target_dict = {k: v for v, k in enumerate(np.unique(labels))}
    labels = np.array([target_dict[class_name] for class_name in labels])
    
   # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(target_dict))
    
    return data, labels, target_dict

def save_preprocessed_data(data, labels, target_dict, file_path):
    print("save file_path : ", file_path )
    np.savez_compressed(file_path, data=data, labels=labels, target_dict=target_dict)
    print(f"Preprocessed data saved to {file_path}")

def load_preprocessed_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        # Here, if target_dict was saved as an array with a single item (a dictionary),
        # we retrieve that item to convert it back to a dictionary.
        target_dict = data['target_dict'].item() if isinstance(data['target_dict'].tolist(), dict) else {}
        return data['data'], data['labels'], target_dict

def maybe_preprocess_and_save(dataset_name, dataset_resize_sizes):
    preprocessed_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          'data', f'preprocessed_{dataset_name}.npz')
    if not os.path.exists(preprocessed_file_path):
        data, class_names = load_dataset(dataset_name, dataset_resize_sizes)
        data, labels, target_dict = preprocess_data(data, class_names)
        save_preprocessed_data(data, labels, target_dict, preprocessed_file_path)
    #else:
        #print(f"Preprocessed data for {dataset_name} already exists.")
    return preprocessed_file_path


def get_data(dataset_name):
    preprocessed_data_path = maybe_preprocess_and_save(dataset_name, dataset_resize_sizes)
    print(f"Loading preprocessed data for {dataset_name}.")
    data, labels, target_dict = load_preprocessed_data(preprocessed_data_path)

    # Print dataset information
    #print(f"Number of data elements: {len(data)}")
    #print(f"Number of classes: {len(target_dict)}")
    #print(f"Classes and their corresponding indices: {target_dict}")

    return data, labels, target_dict

dataset_resize_sizes = {
    'X-SDD': (256, 256),
    'dtd': (224, 224),
    'Brodatz': (80, 80),
    'NEU-CLS': (200, 200),
    'test' : (256, 256),
    'X-SDD-10': (256, 256),
    'X-SDD-20': (256, 256),
    'NEU-CLS-10': (200, 200),
    'NEU-CLS-20': (200, 200),
    # Add more datasets and their resize sizes here
}

# usage
if __name__ == '__main__':
    dataset_name = 'X-SDD'
    start_time = time.time()
    load_dataset(dataset_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f" time: {elapsed_time:.4f} seconds")
    '''
    data, labels, target_dict = get_data('X-SDD')
    print("data shape: ", data.shape)
    print("label shape : ", labels.shape)

    datasets = dataset_resize_sizes.keys()
    for dataset_name in datasets:
        maybe_preprocess_and_save(dataset_name, dataset_resize_sizes)
    '''