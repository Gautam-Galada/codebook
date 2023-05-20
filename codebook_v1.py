import cv2
import numpy as np
image = cv2.imread('image.jpg')
new_size = (300, 200)
resized_image = cv2.resize(image, new_size)
codebook_size = 64
codebook = np.random.rand(codebook_size, 3) * 255

def quantize_image(image, codebook):
    quantized_image = np.zeros_like(image)
    for i, pixel in enumerate(image.reshape(-1, 3)):
        distances = np.linalg.norm(codebook - pixel, axis=1)
        nearest_index = np.argmin(distances)
        quantized_image.reshape(-1, 3)[i] = codebook[nearest_index]
    return quantized_image

quantized_image = quantize_image(resized_image, codebook)
