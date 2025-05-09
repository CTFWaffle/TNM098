import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Load in all images from the folder 'images', store them in a list called 'images'
images = []
for i in range(1, 13):
    img_path = f'Labs/Lab3/images/{i:02d}.jpg'
    img = mpimg.imread(img_path)
    images.append(img)

# Display the images in a grid
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Feature extraction functions
def toGray(image):
    img_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return img_gray.flatten()

def colorContent(image):
    img_flat = image.flatten()
    return img_flat[::1000]  # Sample every 1000th pixel for simplicity

def colorDistribution(image):
    return image.flatten()

def colorDistributionMult(image, points):
    img_flat = image.flatten()
    return [img_flat[point] for point in points]

def luminanceDistribution(image, points):
    img_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return [img_gray.flatten()[point] for point in points]

def edgeDetection(image):
    img_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_edge_x = np.abs(np.convolve(img_gray.flatten(), sobel_x.flatten(), mode='same'))
    img_edge_y = np.abs(np.convolve(img_gray.flatten(), sobel_y.flatten(), mode='same'))
    img_edge = np.sqrt(img_edge_x**2 + img_edge_y**2)
    return img_edge[:1024]  # Truncate or pad to a fixed size

# Pad or truncate feature vectors to a fixed length
def normalize_vector_length(vector, length=1024):
    if len(vector) > length:
        return vector[:length]
    else:
        return np.pad(vector, (0, length - len(vector)), 'constant')

# Calculate distance between all feature vectors
def calculate_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

# Compare all feature vectors of all images
def compare_feature_vectors(feature_vectors):
    num_images = len(feature_vectors)
    distance_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                distance_matrix[i][j] = calculate_distance(feature_vectors[i], feature_vectors[j])
    return distance_matrix

# Show the distance matrix as a heatmap
def show_distance_matrix(distance_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.show()

# Main processing
points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
color_content_array = []
color_distr_array = []
color_distr_array_multiple_points = []
grayImages_array = []
luminance_array = []
edgedetection_array = []

for image in images:
    color_content_array.append(normalize_vector_length(colorContent(image)))
    color_distr_array.append(normalize_vector_length(colorDistribution(image)))
    color_distr_array_multiple_points.append(normalize_vector_length(colorDistributionMult(image, points)))
    grayImages_array.append(normalize_vector_length(toGray(image)))
    luminance_array.append(normalize_vector_length(luminanceDistribution(image, points)))
    edgedetection_array.append(normalize_vector_length(edgeDetection(image)))

# Display distance matrices
show_distance_matrix(compare_feature_vectors(color_content_array), "Color Content Distance Matrix")
show_distance_matrix(compare_feature_vectors(color_distr_array), "Color Distribution Distance Matrix")
show_distance_matrix(compare_feature_vectors(color_distr_array_multiple_points), "Color Distribution Multiple Points Distance Matrix")
show_distance_matrix(compare_feature_vectors(grayImages_array), "Grayscale Distance Matrix")
show_distance_matrix(compare_feature_vectors(luminance_array), "Luminance Distance Matrix")
show_distance_matrix(compare_feature_vectors(edgedetection_array), "Edge Detection Distance Matrix")