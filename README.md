# AI-Powered Multimedia Processing Suite

A powerful desktop application for processing, enhancing, and manipulating images and videos using state-of-the-art AI models and traditional processing techniques.

## 🌟 Features

### Image Processing
- Advanced AI-powered image upscaling using ESRGAN/Real-ESRGAN
- Basic image operations (resize, crop, filter)
- Batch processing capabilities
- Support for multiple image formats

### Video Processing
- Real-time video playback and processing
- Video upscaling and enhancement
- Frame interpolation
- Multiple format support (MP4, AVI, MKV)

### AI Enhancement
- Super-resolution upscaling
- Noise reduction and artifact removal
- GPU acceleration support
- Integrated pre-trained models

## 🛠️ Technology Stack

- **GUI Framework**: PyQt/PySide
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Video Processing**: OpenCV, moviepy, imageio
- **AI/ML**: TensorFlow, PyTorch, ONNX
- **Performance**: NumPy, Dask, Numba
- **Packaging**: PyInstaller

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for AI processing)
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

├── main.py
├── src/
│ ├── app.py
│ ├── gui/
│ │ ├── main_window.py
│ │ ├── video_player.py
│ │ ├── download_dialog.py
│ │ ├── crop_dialog.py
│ │ └── upscale_dialog.py
│ └── processing/
│ ├── image_processing.py
│ └── ai_upscaling.py
```

## 🔧 Configuration

The application settings can be configured through the GUI interface or by modifying the configuration files:

- AI model settings
- Processing parameters
- Output preferences
- Performance options

## 🎯 Features in Detail

### Image Upscaling
- Support for multiple AI models
- Custom upscaling factors
- Batch processing with progress tracking
- Preview functionality

### Video Enhancement
- Frame-by-frame processing
- Real-time preview
- Custom output settings
- Progress monitoring

### User Interface
- Intuitive drag-and-drop interface
- Live preview capabilities
- Progress tracking for long operations
- Multi-threaded processing for responsive UI

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ESRGAN/Real-ESRGAN teams for the AI models
- OpenCV community
- PyQt/PySide developers
- All contributors and users of this project

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers directly.

---

Made with ❤️ by Henrique Souza