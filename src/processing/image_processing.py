from PIL import Image

def load_image(file_path):
    """Load an image from a file."""
    return Image.open(file_path)

def save_image(image, file_path):
    """Save an image to a file."""
    image.save(file_path) 

def crop_image(image, crop_area, fixed_ratio=None):
    """
    Crop the image based on the specified crop area.
    
    :param image: The original image to crop.
    :param crop_area: A tuple (x, y, width, height) defining the crop area.
    :param fixed_ratio: Optional aspect ratio for cropping (e.g., (16, 9)).
    :return: Cropped image.
    """
    # ... existing code ...
    if fixed_ratio:
        # Adjust crop_area to maintain the fixed aspect ratio
        # ... code to adjust crop_area ...
    
        cropped_image = image.crop(crop_area)
        return cropped_image 