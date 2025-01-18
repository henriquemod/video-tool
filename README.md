# AI-Powered Multimedia Processing Suite

A powerful desktop application for processing, enhancing, and manipulating images and videos using state-of-the-art AI models and traditional processing techniques.

![Screenshot From 2024-12-15 21-56-26](https://github.com/user-attachments/assets/41f40562-e22d-4b1c-8e4a-3f8674f57a53)

## 🌟 Features

### Video Player

- Professional-grade video playback with frame-accurate navigation
- Frame-by-frame navigation with keyboard shortcuts
- Advanced screenshot capabilities with AI upscaling
- Multiple playback speeds and volume control

### Image Processing

- Advanced AI-powered image upscaling using multiple models:
  - Real-ESRGAN (2x and 4x)
  - SwinIR (2x and 4x)
  - ESRGAN variants for general and anime content
- Traditional upscaling methods:
  - Bicubic (2x, 3x, 4x)
  - Lanczos (2x, 3x, 4x)
- Interactive image cropping with aspect ratio control
- Batch processing capabilities
- Support for multiple image formats

### Video Processing

- Real-time video playback and processing
- High-quality screenshot capture
- Frame-accurate navigation
- Multiple format support (MP4, AVI, MKV)

### AI Enhancement

- Multiple AI models for different use cases:
  - General purpose upscaling
  - Anime/illustration optimization
  - Lightweight processing options
- GPU acceleration support (CUDA and MPS)
- Integrated progress tracking
- Memory-efficient processing

## 🛠️ Technology Stack

- **GUI Framework**: PyQt5
- **Image Processing**: OpenCV, Pillow
- **Video Processing**: OpenCV, PyQt5 Multimedia
- **AI/ML**:
  - TensorFlow
  - PyTorch
  - basicsr
  - Real-ESRGAN
  - SwinIR
- **Utilities**: NumPy
- **Temp File Management**: Custom implementation

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for AI processing)
- Apple Silicon MPS support (for Mac M1/M2)
- FFmpeg (for video processing)

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/henriquemod/video-tool.git
cd video-tool
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the application:

```bash
python main.py
```

### Project Structure

```
├── main.py
├── src/
│   ├── app.py
│   ├── exceptions/ # Custom exception handling
│   │   ├── upscale_error.py # AI upscaling exceptions
│   │   └── video_processing_error.py # Video processing exceptions
│   ├── gui/
│   │   ├── main_window.py # Main application window
│   │   ├── video_player.py # Video playback component
│   │   ├── download_dialog.py # Video download interface
│   │   ├── crop_dialog.py # Image cropping interface
│   │   ├── upscale_dialog.py # AI upscaling interface
│   │   └── resize_dialog.py # Image resizing interface
│   ├── processing/
│   │   └── ai_upscaling.py # AI enhancement implementation
│   └── utils/
│       ├── icon_utils.py # Icon management
│       └── temp_file_manager.py # Temporary file handling
```

## 🔧 Key Features in Detail

### Video Player

- Frame-accurate navigation with keyboard shortcuts
- Multiple playback speeds
- Screenshot capability with AI enhancement
- Volume control and mute option
- Progress bar with time display

### Image Processing

- Multiple AI upscaling models:
  - Real-ESRGAN variants
  - SwinIR models
  - ESRGAN specialized models
- Batch processing with progress tracking
- Interactive cropping with aspect ratio control
- Preview functionality

### AI Upscaling

- Automatic GPU detection (CUDA/MPS)
- Memory-efficient processing
- Multiple model support
- Progress tracking
- Error handling and recovery

## 🚨 Common Issues & Solutions

### Torchvision and BasicSR Compatibility

#### Issue: Incompatibility between newer torchvision and basicsr

When using newer versions of torchvision (0.20.1+), you might encounter this error:

```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

#### Solution:

We implemented a compatibility layer (`src/utils/torchvision_patch.py`) to handle this issue. This was necessary because:

1. The basicsr package (1.4.2) depends on an older torchvision API that was deprecated and removed
2. basicsr hasn't released an update to use the new torchvision APIs yet
3. Downgrading torchvision would prevent us from using newer features and improvements

The compatibility layer:

- Transparently redirects old import paths to new ones
- Doesn't modify any package code
- Will be easy to remove once basicsr updates

This is a temporary solution until basicsr releases an update using the new torchvision APIs. We chose this approach over:

- Forking and maintaining basicsr (too resource-intensive)
- Downgrading torchvision (would miss out on improvements)
- Waiting for an update (would block development)

The implementation can be found in `src/utils/torchvision_patch.py`.

### Torchvision Deprecation Warning

#### Problem: Deprecation Warning from Torchvision

```
UserWarning: The torchvision.transforms.functional_tensor module is deprecated in 0.15 and will be **removed in 0.17**. Please don't rely on it. You probably just need to use APIs in torchvision.transforms.functional or in torchvision.transforms.v2.functional.
```

#### Solution:

This warning appears when a deprecated torchvision module is imported. The module is used by some of our dependencies (confirmed in `basicsr` and potentially others) and will be removed in a future torchvision version. Since this is a dependency-level warning and doesn't affect functionality, we suppress it in the application. The warning is suppressed because:

1. It comes from dependencies that we don't directly control
2. The dependencies are using their latest stable versions
3. Downgrading torchvision would be counterproductive
4. The warning doesn't affect any functionality
5. The dependencies will need to update their code to use the new recommended APIs

The warning must be suppressed before any imports that might trigger it. In our application, this is done at the very top of main.py:

```python
# Suppress warnings before any imports
import warnings
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='torchvision.transforms.functional_tensor'
)

# Rest of the imports follow...
```

Note: This warning indicates that a future version of torchvision will remove this module. When the affected dependencies update to use the new APIs (`torchvision.transforms.functional` or `torchvision.transforms.v2.functional`), we can remove this warning suppression.

### Qt Platform Plugin Issues

#### Problem: Qt XCB Plugin Error

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "[path]/cv2/qt/plugins" even though it was found.
```

#### Solution:

The issue is related to OpenCV's Qt integration. You can resolve it by using the headless version of OpenCV:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

For more details and alternative solutions, see the full discussion at: [instant-ngp#300](https://github.com/NVlabs/instant-ngp/discussions/300#discussioncomment-3179213)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Real-ESRGAN team for the AI models
- SwinIR team for their implementation
- PyQt community
- OpenCV community
- All contributors and users of this project

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers directly.

---

Made with ❤️ by Henrique Souza
