import numpy as np
import cv2

def postprocess_mask(mask, original_size, confidence_threshold=0.5):
    """
    Postprocess model output mask.
    
    Args:
        mask (np.array): Model output mask
        original_size (tuple): Original image size (H, W)
        confidence_threshold (float): Threshold for binary mask
    
    Returns:
        dict: Postprocessed results
    """
    try:
        # Apply confidence threshold
        binary_mask = (mask > confidence_threshold).astype(np.uint8)
        
        # Resize back to original size
        resized_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        
        # Calculate object statistics
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Object detection results
        objects_found = len(contours) > 0
        num_objects = len(contours)
        total_area = np.sum(resized_mask > 0)
        coverage_percentage = (total_area / (original_size[0] * original_size[1])) * 100
        
        results = {
            'binary_mask': resized_mask,
            'probability_mask': cv2.resize(mask, (original_size[1], original_size[0])),
            'objects_found': objects_found,
            'num_objects': num_objects,
            'total_area_pixels': total_area,
            'coverage_percentage': coverage_percentage,
            'contours': contours,
            'confidence_threshold': confidence_threshold
        }
        
        return results
        
    except Exception as e:
        raise Exception(f"Postprocessing failed: {str(e)}")