import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

def wavelet_decompose(image, wavelet='haar', level=2):

    #print(" level in decompose : ", level)
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    return coeffs

def decompose(image, level=1):

    # Define the low-pass and high-pass filters
    low_pass = np.array([1, 1]) / np.sqrt(2)
    high_pass = np.array([1, -1]) / np.sqrt(2)
    
    def apply_filters(image, low_pass, high_pass):
        """Apply low-pass and high-pass filters to the image and downsample."""
        # Horizontal pass
        low_filtered_row = np.array([np.convolve(row, low_pass, mode='same')[::2] for row in image])
        high_filtered_row = np.array([np.convolve(row, high_pass, mode='same')[::2] for row in image])
        
        # Apply filters along columns
        LL = np.array([np.convolve(col, low_pass, mode='same')[::2] for col in low_filtered_row.T]).T
        LH = np.array([np.convolve(col, high_pass, mode='same')[::2] for col in low_filtered_row.T]).T
        HL = np.array([np.convolve(col, low_pass, mode='same')[::2] for col in high_filtered_row.T]).T
        HH = np.array([np.convolve(col, high_pass, mode='same')[::2] for col in high_filtered_row.T]).T
            
        return LL, LH, HL, HH

    current_image = image
    decomposition = []

    for _ in range(level):
        LL, LH, HL, HH = apply_filters(current_image, low_pass, high_pass)
        decomposition.append((LL, LH, HL, HH))
        current_image = LL

    return decomposition

def plot_decomposition(components, title='Decomposition'):
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
    
    # Flatten the axes array for easy indexing
    axs = axs.ravel()
    
    for i, (component, title) in enumerate(zip(components[0], titles)):
        # Plot each component
        ax = axs[i]
        ax.imshow(component, cmap='gray', interpolation='nearest')
        ax.set_title(f'{title} + {i}')
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example
    image = cv2.imread('WT/data/image_sample.jpg', cv2.IMREAD_GRAYSCALE)
    low_pass = np.array([1, 1]) / np.sqrt(2)
    high_pass = np.array([1, -1]) / np.sqrt(2)
    coeffs1 = decompose(image, level=2)
    coeffs2 = wavelet_decompose(image, wavelet='haar', level=1)

    print("len 01: ", len(coeffs1)," len02: ", len(coeffs2))

    # Plot the decomposition from the custom function
    plot_decomposition(coeffs1, title='Custom Wavelet Decomposition')
