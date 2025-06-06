import numpy as np
import matplotlib.pyplot as plt
import torch
import gradio as gr
from config import config
from PIL import Image
from models.model_loader import load_segmentation_model,run_segmentation_inference
from processing.image_preprocessing import preprocess_image
from utils.visualization import create_segmentation_visualization

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_sample_images():
    """Load sample images for the GUI."""
    # Create mock sample images for demonstration
    sample_images = {}
    
    samples = [
        ("Sample 1", "Sam1"),
        ("Sample 2", "Sam2"),
        ("Sample 3", "Sam3")
    ]
    
    for name, category in samples:
        if category == "Sam1":
            sample_data = np.load("data/sample1.npz")
        elif category == "Sam2":
            sample_data = np.load("data/sample2.npz")
        elif category == "Sam3":
            sample_data = np.load("data/sample3.npz")
        else:
            continue
        image_array = sample_data['X']  # [20,128,128]
        # For display, use the first channel or a composite
        display_img = np.transpose(image_array, (1,2,0))  # [128,128,20]
        display_img = display_img[...,0]  # [128,128] (first channel)
        display_img = ((display_img - display_img.min()) / (display_img.ptp() + 1e-8) * 255).astype(np.uint8)
        sample_images[name] = {
            "raw": image_array,  # [20,128,128]
            "display": Image.fromarray(display_img)
        }
    return sample_images

def process_segmentation(uploaded_image, selected_sample, model_name, confidence_threshold):
    """
    Main segmentation processing function.
    
    Args:
        uploaded_image: User uploaded image
        selected_sample: Selected sample image
        model_name: Selected model name
        confidence_threshold: Confidence threshold for segmentation
    
    Returns:
        tuple: (plot, text_output, status_message)
    """
    try:
        # Determine input image
        if uploaded_image is not None:
            input_image = uploaded_image
            image_source = "uploaded image"
        elif selected_sample is not None:
            sample_images = get_sample_images()
            input_image = sample_images[selected_sample]["raw"]  # [20,128,128]
            display_image = sample_images[selected_sample]["display"]  # PIL.Image for visualization
            image_source = f"sample: {selected_sample}"
            preprocessed_img = input_image  # No preprocessing
            original_img = np.array(display_image)
            preprocessing_info = {
                'original_size': preprocessed_img.shape[1:],
                'target_size': preprocessed_img.shape[1:],
                'normalization': 'none',
                'resize_method': 'none'
            }
        else:
            return None, "⚠️ Please upload an image or select a sample image.", "No image provided"
        
        # Update confidence threshold in config
        config.confidence_threshold = 0.5
        
        if uploaded_image is not None:
            # Preprocessing
            preprocessed_img, original_img = preprocess_image(
                uploaded_image, config.input_size
            )
        else:
            preprocessed_img = input_image
        
        # Load model and run inference
        model = load_segmentation_model(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Postprocess results
        postprocess_results = run_segmentation_inference(
            preprocessed_img, model
        )
        
        # Generate visualization
        plot = create_segmentation_visualization(
            original_img, postprocess_results, model_name
        )
        
        # Generate text results
        text_output = ""
        
        # Status message
        # detection_status = "detected" if postprocess_results['objects_found'] else "not detected"
        # status_message = f"✅ Processed {image_source} with {model_name} - Objects {detection_status}"
        
        return plot, text_output, "Processing completed successfully"
        
    except Exception as e:
        error_message = f"❌ Processing failed: {str(e)}"
        return None, error_message, "Error occurred"

def create_segmentation_interface():
    """Create the segmentation GUI interface."""
    
    sample_images = get_sample_images()
    sample_names = list(sample_images.keys())
    
    with gr.Blocks(title="Object Segmentation GUI", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# Methane Plume detection GUI")
        gr.Markdown("Upload an image or select a sample"+"\nUploading image should contain a sentinel-2 image with 20 channels")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model Configuration
                gr.Markdown("### 🔧 Model Configuration")
                
                model_dropdown = gr.Dropdown(
                    choices=config.model_names,  # change it to user based
                    value=config.model_names[0],
                    label="Segmentation Model",
                    info="Select the model architecture"
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=config.confidence_threshold,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence for object detection"
                )
                
                # Image Input
                gr.Markdown("### 📁 Image Input")
                
                uploaded_image = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                
                gr.Markdown("**OR**")
                
                sample_dropdown = gr.Dropdown(
                    choices=sample_names,
                    label="Select Sample Image",
                    info="Choose from predefined samples",
                    value="Sample 1"
                )
                
                # Process Controls
                process_btn = gr.Button(
                    "🚀 Run Segmentation", 
                    variant="primary", 
                    size="lg"
                )
                
                status_display = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Ready for processing..."
                )
            
            with gr.Column(scale=2):
                # Results Display
                gr.Markdown("### 📊 Segmentation Results")
                
                plot_output = gr.Plot()
                
                text_output = gr.Markdown(
                    value="Results will appear here after processing...",
                    label="Analysis Report"
                )
        
        # Event Handlers
        process_btn.click(
            fn=process_segmentation,
            inputs=[uploaded_image, sample_dropdown, model_dropdown, confidence_slider],
            outputs=[plot_output, text_output, status_display]
        )
        
        # Clear interactions
        uploaded_image.change(
            fn=lambda img: None if img is not None else gr.update(),
            inputs=[uploaded_image],
            outputs=[sample_dropdown]
        )
        
        sample_dropdown.change(
            fn=lambda sample: None if sample is not None else gr.update(),
            inputs=[sample_dropdown],
            outputs=[uploaded_image]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_segmentation_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=False,
        debug=True
    )