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
│   ├── gui/
│   │   ├── main_window.py      # Main application window
│   │   ├── video_player.py     # Video playback component
│   │   ├── download_dialog.py  # Video download interface
│   │   ├── crop_dialog.py      # Image cropping interface
│   │   ├── upscale_dialog.py   # AI upscaling interface
│   │   └── resize_dialog.py    # Image resizing interface
│   ├── processing/
│   │   └── ai_upscaling.py     # AI enhancement implementation
│   └── utils/
│       ├── icon_utils.py       # Icon management
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
