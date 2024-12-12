import cv2

def load_video(file_path):
    """Load a video from a file."""
    return cv2.VideoCapture(file_path)

def get_frame(video_capture, frame_number):
    """Retrieve a specific frame from a video."""
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    return frame if ret else None 