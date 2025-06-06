import numpy as np
import matplotlib.pyplot as plt

def create_segmentation_visualization(original_image, results, model_name):
    """
    Create comprehensive visualization of segmentation results.
    
    Args:
        original_image (np.array): Original input image
        results (dict): Postprocessed segmentation results
        model_name (str): Name of the model used
    
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Original Image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Binary Mask
    axes[1].imshow(results['binary_mask'], cmap='gray')
    axes[1].set_title('Binary Segmentation Mask', fontweight='bold')
    axes[1].axis('off')
    
    # Probability Mask
    im = axes[2].imshow(results['probability_mask'], cmap='jet', alpha=0.8)
    axes[2].set_title('Probability Mask', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
