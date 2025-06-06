import numpy as np
import cv2

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess input image for segmentation model.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size for model input
    
    Returns:
        tuple: (preprocessed_image, original_image, preprocessing_info)
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        original_size = img_array.shape[:2]
        
        # Store original for overlay purposes
        original_image = img_array.copy()
        
        # Resize image
        resized_img = cv2.resize(img_array, target_size)
        
        # Normalize pixel values to [0, 1]
        normalized_img = resized_img.astype(np.float32) / 255.0
        
        # TODO: Add model-specific preprocessing
        # - Mean subtraction
        # - Standard deviation normalization
        # - Channel reordering (RGB vs BGR)
        
        
        return normalized_img, original_image
        
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")