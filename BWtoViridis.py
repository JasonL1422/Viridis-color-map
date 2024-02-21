from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Define thresholds for near-white and near-black
near_white_threshold = 245 # Pixels brighter than this will be considered white
near_black_threshold = 10  # Pixels darker than this will be considered black

# Get a list of all .png files in the current directory
png_files = glob.glob('*.png')

# Process each file
for image_file_name in png_files:
    # Load the image
    uploaded_image = io.imread(image_file_name)

    # Convert the image to grayscale while preserving the three channels to retain the shape
    gray_image = color.rgb2gray(uploaded_image)
    gray_image_3c = color.gray2rgb(gray_image)

    # Extract the mask of areas that are not near-white and not near-black
    non_white_black_mask = ((uploaded_image[:, :, 0] < near_white_threshold) | \
                            (uploaded_image[:, :, 1] < near_white_threshold) | \
                            (uploaded_image[:, :, 2] < near_white_threshold)) & \
                           ((uploaded_image[:, :, 0] > near_black_threshold) & \
                            (uploaded_image[:, :, 1] > near_black_threshold) & \
                            (uploaded_image[:, :, 2] > near_black_threshold))

    # Expand the mask to work for 3 channels
    non_white_black_mask_3c = np.stack([non_white_black_mask] * 3, axis=-1)

    # Apply the 'viridis' colormap to the grayscale image
    viridis_colored_image = plt.get_cmap('viridis')(gray_image)[:, :, :3]

    # Combine the original image and the new viridis image using the mask
    combined_image = np.where(non_white_black_mask_3c, viridis_colored_image, uploaded_image / 255)

    # Convert combined image to uint8 type for display and saving
    combined_image_uint8 = (combined_image * 255).astype(np.uint8)
    combined_image_pil = Image.fromarray(combined_image_uint8)

    # Save the combined image with the '_viridis' suffix
    combined_image_name = image_file_name.rsplit('.', 1)[0] + '_viridis.' + image_file_name.rsplit('.', 1)[1]
    combined_image_pil.save(combined_image_name)

    # Optionally, display the combined image
    plt.imshow(combined_image_pil)
    plt.title(f'Image with Viridis Colormap on Non-White/Black Areas - {image_file_name}')
    plt.axis('off')
    plt.show()
