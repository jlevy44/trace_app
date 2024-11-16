import numpy as np
import cv2
from scipy.ndimage import generic_filter, median_filter
import pysnooper

def calculate_mean(arr, x, y):
    neighbors = [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) if (i, j) != (x, y)]
    values = []
    for i, j in neighbors:
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
            values.append(arr[i, j])
    values = [0 if np.isnan(v) else v for v in values]
    return np.mean(values) if values else np.nan

def replace_nan_with_mean(arr):
    new_arr = np.copy(arr)
    nan_map = np.isnan(new_arr)
    while np.any(nan_map):
        for x in range(new_arr.shape[0]):
            for y in range(new_arr.shape[1]):
                if nan_map[x, y]:
                    new_arr[x, y] = calculate_mean(new_arr, x, y)
        nan_map = np.isnan(new_arr)

    return new_arr

def replace_nan_with_median(arr, use_convolve=False):
    # Work directly with a copy of the input array
    result = np.copy(arr)

    if not use_convolve:
        # Find indices of NaN values
        nan_indices = np.argwhere(np.isnan(result))
        
        # Iterate over each NaN index to replace it
        for x, y in nan_indices:
            # Define the bounds for a 3x3 neighborhood
            x_min, x_max = max(0, x-1), min(result.shape[0], x+2)
            y_min, y_max = max(0, y-1), min(result.shape[1], y+2)
            
            # Extract the 3x3 neighborhood
            neighborhood = result[x_min:x_max, y_min:y_max]
            
            # Calculate the median of non-NaN neighbors
            median_val = np.nanmedian(neighborhood)
            
            # Replace the NaN with the calculated median
            result[x, y] = median_val
    else:
        # Boolean mask for NaN values
        nan_mask = np.isnan(result)
        
        if np.any(nan_mask):
            # Apply the median filter only once across the whole array
            median_filtered = generic_filter(result, np.nanmedian, size=3, mode='constant', cval=np.nan)
            
            # Update only NaN positions in the original array with the filtered result
            result[nan_mask] = median_filtered[nan_mask]
    
    return result

def replace_nan_with_median(arr):
    # Create a copy of the array and mask to remember original NaN positions
    result = np.copy(arr)
    nan_mask = np.isnan(result)
    
    # Replace NaNs with -Inf temporarily
    result[nan_mask] = -np.inf
    
    # Apply the median filter
    median_filtered = median_filter(result, size=3, mode='constant', cval=-np.inf)
    
    # Replace only original NaNs with their median-filtered values
    result[nan_mask] = median_filtered[nan_mask]
    
    return result

def test_one_point_rotate(x_1, y_1, x_2, y_2, x_1_c, y_1_c, x_2_c, y_2_c):
    if ((x_1-x_1_c)/(x_2-x_2_c))/((y_1-y_1_c)/(y_2-y_2_c)) > 0:
        return 1
    else:
        return -1

def warp_with_nan(element_image, homo, output_w, output_h):
    # Define a unique value to represent NaN temporarily
    unique_value = -1
    # Apply perspective warp with the unique border value
    warped_image = cv2.warpPerspective(
        src=element_image,
        M=homo,
        dsize=(output_w, output_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(unique_value,) * element_image.shape[2] if len(element_image.shape) > 2 else unique_value
    )
    
    # Convert the result to float and replace unique_value with NaN
    warped_image = warped_image.astype(float)
    warped_image[warped_image == unique_value] = np.nan
    return warped_image

def warp_metals_new(slide_x, slide_y, metals_x, metals_y, dfs_new_df_co, metals_shape, hne_shape):
    warped_metals = dict()
    
    # Prepare source and destination points for homography
    points_src = list(zip(metals_x, metals_y))
    points_dst = list(zip(slide_x, slide_y))
    if len(points_src) > len(points_dst):
        points_src = points_src[:len(points_dst)]
    else:
        points_dst = points_dst[:len(points_src)]
    
    points_src = np.array(points_src)
    points_dst = np.array(points_dst)
    
    # Compute homography matrix
    homo, status = cv2.findHomography(points_src, points_dst)
    output_h, output_w = hne_shape[0], hne_shape[1]
    
    # Apply the transformation for each element
    for _, (element, element_image) in dfs_new_df_co[['element', 'image']].iterrows():
        warped_metals[element] = warp_with_nan(element_image, homo, output_w, output_h)
    
    return dict(warped_metals=warped_metals, homo=homo)
