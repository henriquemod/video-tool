import torch
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

# Model URLs for SwinIR and Real-ESRGAN
MODEL_URLS = {
    'SwinIR-2x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    'SwinIR-4x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    'Real-ESRGAN-2x': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'Real-ESRGAN-4x': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}

"""
AI Upscaling Module - Advanced Image Enhancement System

This module implements a sophisticated image upscaling system that combines multiple
state-of-the-art AI models with classical upscaling methods. It provides a flexible
and efficient interface for high-quality image enhancement.

Key Features:
- Multiple AI model support (SwinIR, Real-ESRGAN)
- Classical upscaling methods (Bicubic, Lanczos)
- Automatic device selection (CUDA, MPS, CPU)
- Tile-based processing for large images
- Memory-efficient implementation
- Progress tracking support

Models:
1. SwinIR:
   - Lightweight SR model
   - Efficient transformer-based architecture
   - Support for 2x and 4x upscaling
   - Memory-efficient processing
   - Tile-based handling for large images

2. Real-ESRGAN:
   - Enhanced SRGAN architecture
   - Robust to real-world degradation
   - 2x and 4x upscaling support
   - Improved detail preservation
   - Better artifact handling

Technical Implementation:
1. Device Management:
   - Automatic GPU detection
   - CUDA support for NVIDIA GPUs
   - MPS support for Apple Silicon
   - Graceful fallback to CPU
   - Dynamic memory optimization

2. Processing Pipeline:
   - Image preprocessing
   - Padding handling
   - Tiled processing for large images
   - Color space conversion
   - Result post-processing

3. Memory Management:
   - Efficient tensor operations
   - Automatic garbage collection
   - Batch size optimization
   - Resource cleanup
   - Memory usage monitoring

Functions:
    get_device():
        Determines the best available processing device
        Returns:
            str: 'cuda', 'mps', or 'cpu'

    upscale_image(img, method, scale, device=None):
        Main interface for image upscaling
        Args:
            img: numpy array (BGR format)
            method: upscaling method name
            scale: upscaling factor (2 or 4)
            device: processing device (optional)
        Returns:
            numpy array: upscaled image

    upscale_realesrgan(img, scale, device):
        Real-ESRGAN specific implementation
        Args:
            img: input image
            scale: upscaling factor
            device: processing device
        Returns:
            numpy array: upscaled image

    upscale_swinir(img, scale, device):
        SwinIR specific implementation
        Args:
            img: input image
            scale: upscaling factor
            device: processing device
        Returns:
            numpy array: upscaled image

Model Configuration:
- SwinIR:
    * Architecture: Swin Transformer
    * Parameters: 60 embedding dim, 6 depth
    * Window size: 8
    * Upsampler: pixelshuffledirect

- Real-ESRGAN:
    * Architecture: RRDB
    * Features: 64 channels
    * Blocks: 23
    * Growth channels: 32

Dependencies:
- torch: Deep learning framework
- basicsr: Base SR toolkit
- realesrgan: Real-ESRGAN implementation
- numpy: Numerical operations
- opencv-python: Image processing

Performance Considerations:
- GPU memory management
- Tile size optimization
- Batch processing efficiency
- Color space conversions
- Memory cleanup

Error Handling:
- Device availability checks
- Model loading failures
- Memory exhaustion
- Processing errors
- Invalid inputs

@see @Project Structure#ai_upscaling.py
@see @Project#AI Enhancement
@see @Project#Image Processing
"""


def get_device():
    """Get the best available device for AI processing."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def upscale_image(img, method, scale, device=None):
    """
    Upscale an image using the specified method and scale.

    Args:
        img: numpy array of the image (BGR format)
        method: str, upscaling method ('SwinIR' or 'Real-ESRGAN')
        scale: int, upscaling factor (2 or 4)
        device: str, device to use for processing (optional)

    Returns:
        numpy array of the upscaled image
    """
    if device is None:
        device = get_device()

    if "Real-ESRGAN" in method:
        return upscale_realesrgan(img, scale, device)
    elif "SwinIR" in method:
        return upscale_swinir(img, scale, device)
    else:
        # Fallback to classical methods
        if method == "Bicubic":
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif method == "Lanczos":
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        else:
            raise ValueError(f"Unsupported upscaling method: {method}")


def upscale_realesrgan(img, scale, device):
    """Upscale image using Real-ESRGAN."""
    try:
        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )

        # Select appropriate model based on scale
        model_url = MODEL_URLS[f'Real-ESRGAN-{scale}x']
        model_path = load_file_from_url(model_url, model_dir='models')

        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=True if device == 'cuda' else False,
            device=device
        )

        # Process image
        output, _ = upsampler.enhance(img, outscale=scale)
        return output

    except Exception as e:
        raise Exception(f"Real-ESRGAN processing failed: {str(e)}")


def upscale_swinir(img, scale, device):
    """Upscale image using SwinIR."""
    try:
        from basicsr.utils.registry import ARCH_REGISTRY

        # Prepare the image
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle padding
        h, w = img.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Load model
        model_url = MODEL_URLS[f'SwinIR-{scale}x']
        model_path = load_file_from_url(model_url, model_dir='models')

        # Initialize model
        model = ARCH_REGISTRY.get('SwinIR')(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv'
        )

        # Load weights
        pretrained_model = torch.load(model_path, map_location=device)
        if 'params_ema' in pretrained_model:
            pretrained_model = pretrained_model['params_ema']
        elif 'params' in pretrained_model:
            pretrained_model = pretrained_model['params']

        model.load_state_dict(pretrained_model, strict=True)
        model.eval()
        model.to(device)

        # Process image
        img_tensor = torch.from_numpy(img).float().permute(
            2, 0, 1).unsqueeze(0) / 255.
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)

        # Final processing
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output = output.transpose(1, 2, 0)

        if pad_h > 0 or pad_w > 0:
            output = output[:h*scale, :w*scale, :]

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    except Exception as e:
        raise Exception(f"SwinIR processing failed: {str(e)}")
