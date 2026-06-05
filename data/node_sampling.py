import cv2
import numpy as np
from detectron2.structures import PolygonMasks
from detectron2.structures.masks import polygons_to_bitmask


def boundary_sampling_in_mask_uni(mask, num_samples=10, inward_offset=0):
    """
     mask  (x, y), . 
    :param mask:  mask,  [H, W], , True  mask . 
    :param num_samples: . 
    :param inward_offset: . 
    :return:  (x, y) . 
    """
    #  OpenCV  mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    boundary_points = np.vstack(contours).squeeze(1)  #  (x, y)

    if len(boundary_points) == 0:
        return []

    # ,
    offset_boundary_points = []
    for point in boundary_points:
        normal_vector = np.array([point[0] - mask.shape[1] // 2, point[1] - mask.shape[0] // 2])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        inward_point = point - inward_offset * normal_vector
        inward_point = np.clip(inward_point, 0, [mask.shape[1] - 1, mask.shape[0] - 1])
        offset_boundary_points.append(inward_point)

    offset_boundary_points = np.array(offset_boundary_points)


    sampled_indices = np.linspace(0, len(offset_boundary_points) - 1, num_samples, dtype=int)
    sampled_points = offset_boundary_points[sampled_indices]

    #  (x, y)
    return [(int(x), int(y)) for x, y in sampled_points]

def grid_sampling_in_mask(mask, grid_size=20):
    """
     mask . 
    :param mask:  mask,  [H, W], , True  mask . 
    :param grid_size: , . 
    :return:  (x, y) . 
    """
    height, width = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(0, width, grid_size), np.arange(0, height, grid_size))
    
    grid_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    
    #  mask
    sampled_points = [tuple(point) for point in grid_points if mask[point[1], point[0]]]

    return sampled_points

def process_polygon_masks(polygon_masks, image_height, image_width, num_samples_boundary=10, num_samples_centroid=5, radius_centroid=10):
    all_boundary_samples = []
    all_centroid_samples = []


    for instance_mask in polygon_masks:

        binary_mask = polygons_to_bitmask(instance_mask, image_height, image_width)
        

        boundary_samples = boundary_sampling_in_mask_uni(binary_mask, num_samples=num_samples_boundary)
        all_boundary_samples.extend(boundary_samples)
        

        centroid_samples = grid_sampling_in_mask(binary_mask, num_samples=num_samples_centroid, radius=radius_centroid)
        all_centroid_samples.extend(centroid_samples)
    
    return all_boundary_samples, all_centroid_samples

def boundary_sampling_in_mask(mask, num_samples=10):
    """
     mask  (x, y). 
    :param mask:  mask,  [H, W], , True  mask . 
    :param num_samples: . 
    :return:  (x, y) . 
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_points = np.vstack(contours).squeeze(1)  #  (x, y)
    if len(boundary_points) == 0:
        return []
    sampled_indices = np.random.choice(len(boundary_points), num_samples, replace=False)
    sampled_points = boundary_points[sampled_indices]
    return [(x, y) for x, y in sampled_points]

def centroid_sampling_in_mask(mask, num_samples=10, radius = 10):
    """
     mask  (x, y). 
    :param mask:  mask,  [H, W], , True  mask . 
    :param num_samples: . 
    :return:  (x, y) . 
    """
    #  mask
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] == 0:
        return []

    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])


    sampled_points = []
    for i in range(num_samples):
        angle = 2 * np.pi * i / num_samples
        x = int(centroid_x + radius * np.cos(angle))
        y = int(centroid_y + radius * np.sin(angle))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
            sampled_points.append((x, y))

    return sampled_points

def visualize_sampling_on_mask(image_path, binary_mask, boundary_samples, centroid_samples):
    """
    . 
    :param image_path: . 
    :param binary_mask: . 
    :param boundary_samples:  (x, y) . 
    :param centroid_samples:  (x, y) . 
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #  figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for instance_mask in binary_mask:

        binary_mask = polygons_to_bitmask(instance_mask, image.shape[0], image.shape[1])


    binary_mask_colored = np.zeros_like(image)
    binary_mask_colored[binary_mask == 1] = [255, 0, 0]
    plt.imshow(binary_mask_colored, alpha=0.4)  #  alpha


    if boundary_samples:
        boundary_samples = np.array(boundary_samples)
        plt.scatter(boundary_samples[:, 0], boundary_samples[:, 1], color='yellow', label='Boundary Points', s=30)


    if centroid_samples:
        centroid_samples = np.array(centroid_samples)
        plt.scatter(centroid_samples[:, 0], centroid_samples[:, 1], color='green', label='Centroid Points', s=50)


    plt.legend(loc="best")
    plt.title("Visualization of Boundary and Centroid Sampling")
    plt.axis('off')
    plt.savefig('xxxx.png')