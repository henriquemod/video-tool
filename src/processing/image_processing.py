from PIL import Image

def load_image(file_path):
    """Load an image from a file."""
    return Image.open(file_path)

def save_image(image, file_path):
    """Save an image to a file."""
    image.save(file_path) 