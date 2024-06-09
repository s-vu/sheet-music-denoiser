import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image


def show_image(image, scale=1):
    """
    Displays an image using matplot
    """
    scale = int(100 / scale)
    plt.figure(figsize=(image.shape[1] / scale, image.shape[0] / scale))  # Set the figure size based on the image dimensions, 100 = default
    plt.imshow(image)
    plt.axis('off')  # Turn off axis if not needed
    plt.show()

def add_noise(image, amt=10):
    """
    Adds random noise by adding random amount to each pixel
    """
    noise = np.random.normal(0, amt, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def add_salt_pepper_noise(image, salt_prob=.07, pepper_prob=.07):
    """
    Adds random salt (white pixels) and pepper (black pixels) noise
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob

    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0

    return noisy_image

def add_grain_noise(image, intensity=50):
    noise = np.random.normal(scale=intensity, size=image.shape)
    noise += image.astype(np.float64)
    return np.clip(noise, 0, 255).astype(np.uint8)

def apply_blur(image, kernel_size=(9, 9)):
    """
    Applies a slight gaussian blur to filter
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def apply_rotate(image, angle=90):
    """
    rotates image by [angle] degrees counterclockwise
    """

    height, width = image.shape[:2]
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    # Calculate the bounding box of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    rotation_matrix[0, 2] += (new_width / 2) - width / 2
    rotation_matrix[1, 2] += (new_height / 2) - height / 2
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated_image

def save_image(image, filename):
    pil_image = Image.fromarray(image)
    file_path = '../new_noise/new_noise_' + filename #change '../data/output_' to folder you want to save to
    pil_image.save(file_path)

if __name__ == '__main__':

    for filename in os.listdir('../image'): #change ../image to folder images are in
        funcs = [apply_blur, add_salt_pepper_noise, add_noise]
        img = plt.imread('../image/' + filename)

        img = add_grain_noise(img, 80)
        #img = add_noise(img, np.random.randint(10, 30))
        #img = add_salt_pepper_noise(img, np.random.uniform(low=.07, high=.12), np.random.uniform(low=.07, high=.12))

        #for func in funcs:
        #    img = func(img)
        #Pick a random number of filters to use
        #num_filters = random.randint(1, len(funcs))
        #for _ in range(num_filters):
        #    #Pick a random filter, apply it, then remove it from the list
        #    idx = random.randint(0, len(funcs)-1)
        #    func = funcs.pop(idx)
        #    print(func.__name__)
        #    img = func(img)


        save_image(img, filename)