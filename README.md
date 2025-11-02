# Khmer Character Classification Project 🇰🇭

A machine learning application that classifies Khmer handwritten and printed characters using deep learning. This project features a PyTorch-based neural network classifier and an interactive Streamlit web interface for real-time character recognition.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Details](#model-details)
- [Supported Characters](#supported-characters)
- [File Descriptions](#file-descriptions)

## 🎯 Overview

This project implements a neural network-based classification system for recognizing Khmer (Cambodian) script characters. The system can identify individual Khmer consonants from images, making it useful for:

- **OCR (Optical Character Recognition)** applications for Khmer text
- **Handwriting recognition** systems
- **Document digitization** projects
- **Educational tools** for learning Khmer script
- **Research** in Southeast Asian language processing

The application provides an easy-to-use web interface where users can upload images of Khmer characters and receive instant predictions with confidence scores.

## ✨ Features

- **Real-time Character Classification**: Upload images and get instant predictions
- **Confidence Scoring**: View prediction confidence levels with visual indicators
- **Top-K Predictions**: See the top 3 most likely character matches
- **Interactive Web Interface**: User-friendly Streamlit-based UI
- **Image Preprocessing**: Automatic image normalization and resizing
- **Multiple Character Support**: Trained on 10 Khmer character classes
- **Production-Ready**: Pre-trained model included for immediate use

## 📁 Project Structure

```
khmer-character-classification/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── model/                      # Trained model files
│   ├── khmer_char_model.pth   # PyTorch model weights
│   └── label_encoder.joblib   # Label encoder for classes
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── model_def.py           # Neural network model definition
│   ├── preprocess.py           # Image preprocessing utilities
│   └── inference.py            # Model loading and prediction logic
├── utils/                      # Utility modules
│   └── label_map.py           # Mapping between labels and Khmer characters
├── test_images_khmer/          # Test images for validation
├── fonts/                      # Khmer font files
│   └── NotoSansKhmer.ttf
└── data/                       # Data directory
    └── sample_images/
```

## 🏗️ Architecture

### Model Architecture

The project uses a **feedforward neural network** with the following structure:

- **Input Layer**: 2304 features (48×48 pixel grayscale image flattened)
- **Hidden Layer 1**: 200 neurons with ReLU activation
- **Hidden Layer 2**: 100 neurons with BatchNorm1d and ReLU activation, 20% dropout
- **Hidden Layer 3**: 50 neurons with BatchNorm1d and ReLU activation, 10% dropout
- **Output Layer**: 10 neurons (one per character class)

### Processing Pipeline

1. **Image Upload**: User uploads an image via Streamlit interface
2. **Preprocessing**: 
   - Convert to grayscale
   - Resize to 48×48 pixels
   - Normalize pixel values
   - Flatten to 1D tensor
3. **Model Inference**: 
   - Load pre-trained model and label encoder
   - Forward pass through neural network
   - Apply softmax to get probability distribution
4. **Post-processing**:
   - Extract top prediction and confidence score
   - Map English label to Khmer character
   - Display results with visual feedback

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for the neural network model
- **Streamlit**: Web framework for the interactive UI
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical operations and array handling
- **scikit-learn**: Label encoding utilities
- **joblib**: Model serialization and loading
- **Python 3.x**: Programming language

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd khmer-character-classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist**:
   - Ensure `model/khmer_char_model.pth` exists
   - Ensure `model/label_encoder.joblib` exists

## 📖 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Use the application**:
   - Click "Upload Image" or drag and drop an image file
   - Supported formats: PNG, JPG, JPEG
   - Wait for the classification results
   - View the predicted character, confidence score, and top 3 predictions

### Example Usage Flow

1. Upload a Khmer character image (e.g., from `test_images_khmer/`)
2. View the uploaded image in the interface
3. Review the prediction results:
   - **Predicted Character**: The Khmer character identified
   - **English Label**: The internal label (e.g., "KO", "CHA")
   - **Confidence**: Prediction confidence percentage
   - **Top 3 Predictions**: Alternative character matches

## 🤖 Model Details

### Training Specifications

- **Model Type**: Feedforward Neural Network (Multi-layer Perceptron)
- **Input Size**: 2304 features (48×48 grayscale image)
- **Output Classes**: 10 Khmer characters
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Regularization**: Batch Normalization and Dropout to prevent overfitting

### Model Performance

The model uses:
- **Batch Normalization** in hidden layers for stable training
- **Dropout layers** (20% and 10%) to reduce overfitting
- **Softmax activation** for probability distribution over classes

### Loading and Inference

The model is loaded from saved state dict (`khmer_char_model.pth`) and the label encoder from `label_encoder.joblib`. The model is set to evaluation mode (`model.eval()`) for inference, disabling dropout and batch norm statistics updates.

## 🔤 Supported Characters

The model can classify **10 Khmer consonant characters**. Current label mappings include:

| English Label | Khmer Character | English Label | Khmer Character |
|--------------|-----------------|---------------|-----------------|
| KO | ក | CHA | ឆ |
| KHO | គ | CHHA | ឈ |
| KHA | ខ | CHHO | ជ |
| NGO | ង | DA | ដ |
| TA | ត | NA | ណ |

*Note: The label encoder contains all 10 classes.*

## 📝 File Descriptions

### Core Application Files

- **`app.py`**: Main Streamlit application with UI components, file upload handling, and result display
- **`requirements.txt`**: Python package dependencies list

### Source Code Modules (`src/`)

- **`model_def.py`**: Defines the `NNClassifierModel` class with neural network architecture
- **`preprocess.py`**: Contains `preprocess_image()` function for image normalization and tensor conversion
- **`inference.py`**: Provides `load_model_and_encoder()` and `predict()` functions for model inference

### Utility Modules (`utils/`)

- **`label_map.py`**: Dictionary mapping English label constants to Khmer Unicode characters

### Model Files (`model/`)

- **`khmer_char_model.pth`**: Serialized PyTorch model state dictionary
- **`label_encoder.joblib`**: Scikit-learn label encoder for class-to-index mapping

## 🔍 Code Examples

### Loading and Using the Model Programmatically

```python
from src.inference import load_model_and_encoder, predict
from src.preprocess import preprocess_image
from PIL import Image
from utils.label_map import LABEL_TO_KHMER

# Load model and label encoder
model, le = load_model_and_encoder()

# Preprocess an image
image = Image.open("path/to/character.png")
x = preprocess_image(image)

# Make prediction
label_const, confidence, probs = predict(model, le, x)

# Get Khmer character
khmer_char = LABEL_TO_KHMER.get(label_const)
print(f"Predicted: {khmer_char} ({label_const}) with {confidence*100:.2f}% confidence")
```

### Image Preprocessing

```python
from src.preprocess import preprocess_image
from PIL import Image

image = Image.open("character.png")
tensor = preprocess_image(image)
# Returns: torch.Tensor of shape [1, 2304]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**:
   - Ensure `model/khmer_char_model.pth` and `model/label_encoder.joblib` exist
   - Check file paths are correct

2. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure virtual environment is activated

3. **Low prediction confidence**:
   - Image quality may be poor
   - Character may not be clearly visible
   - Try images with single, centered characters on clean backgrounds

4. **Streamlit not starting**:
   - Check if port 8501 is available
   - Use `streamlit run app.py --server.port 8502` to use a different port

## 🎓 Use Cases

- **Educational Applications**: Learning tools for Khmer script recognition
- **Document Processing**: Digitizing Khmer documents and manuscripts
- **Mobile Apps**: Integration into mobile applications for handwriting recognition
- **Research**: Basis for more complex Khmer OCR systems
- **Accessibility**: Assisting visually impaired users with Khmer text

## 🤝 Contributing

This project is open for contributions! Areas for improvement:
- Expanding character support
- Improving model accuracy
- Adding training scripts
- Enhancing UI/UX
- Supporting full word/sentence recognition

## 📄 License

*Add your license information here*

## 🙏 Acknowledgments

- Khmer script Unicode standard
- PyTorch and Streamlit communities
- Contributors and users of this project

---

**Built with ❤️ using PyTorch & Streamlit**
