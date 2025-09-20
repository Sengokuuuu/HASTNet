import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load images from {image_path}")
    
    f = np.fft.fft2(img)
    
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    magnitude_spectrum_img = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    cv2.imwrite(output_path, magnitude_spectrum_img)
    
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.title("Magnitude Spectrum")
    plt.imshow(magnitude_spectrum_img, cmap='gray')
    plt.savefig('crack_fft.png')

input_image_path = 'dataset/val/images/78.png'
output_image_path = 'fft_output.jpg'

fft_image(input_image_path, output_image_path)
