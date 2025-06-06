class Config:
    """Configuration class for the segmentation GUI"""
    def __init__(self):
        self.model_names = [
            "SimpleUNet"
        ]
        self.input_size = (128, 128)
        self.confidence_threshold = 0.5
        self.sample_images_path = "data/"

config = Config()