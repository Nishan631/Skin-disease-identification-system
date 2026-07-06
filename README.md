# 🏥 Skin Disease Identification System

> An intelligent deep learning solution for accurate detection and classification of common skin diseases using Convolutional Neural Networks (CNN).

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Diseases](#supported-diseases)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project leverages **Convolutional Neural Networks (CNN)** to identify and classify five common skin diseases from medical images. The system combines data augmentation techniques with a multi-layer neural network to achieve robust classification performance.

The application includes:
- ✅ Machine learning model training pipeline
- ✅ Web-based Flask interface for easy predictions
- ✅ Real-time disease classification
- ✅ Data augmentation for improved generalization

## ✨ Features

- **Automated Disease Detection**: Classifies skin conditions with high accuracy
- **Multi-Class Classification**: Identifies 5 different skin disease categories
- **Data Augmentation**: Implements advanced techniques to prevent overfitting
- **Flask Web Interface**: User-friendly interface for predictions
- **Modular Design**: Clean separation of training and inference logic
- **Production Ready**: Serialized model for deployment

## 🔬 Supported Diseases

| Disease | Description |
|---------|-------------|
| **Acne** | Inflammatory skin condition characterized by pimples and blackheads |
| **Melanoma** | Most serious type of skin cancer; early detection is critical |
| **Psoriasis** | Chronic autoimmune skin condition causing red, scaly patches |
| **Rosacea** | Causes facial redness and small, red bumps on the face |
| **Vitiligo** | Loss of skin pigmentation resulting in white patches |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nishan631/Skin-disease-identification-system.git
cd Skin-disease-identification-system
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
Organize your dataset in the following structure:
```
Skin_Diseases/
├── train/
│   ├── Acne/
│   ├── Melanoma/
│   ├── Psoriasis/
│   ├── Rosacea/
│   └── Vitiligo/
└── test/
    ├── Acne/
    ├── Melanoma/
    ├── Psoriasis/
    ├── Rosacea/
    └── Vitiligo/
```

## 📁 Project Structure

```
Skin-disease-identification-system/
├── Skin_disease_detection.ipynb    # Jupyter notebook for exploration & analysis
├── train_model.py                   # Model training script
├── requirements.txt                 # Project dependencies
├── Skin_Diseases/                   # Dataset directory
│   ├── train/                       # Training images
│   └── test/                        # Testing images
├── Flask/                           # Flask application directory
│   ├── app.py                       # Flask application
│   ├── Skin_Diseases.h5            # Trained model (generated)
│   └── templates/                   # HTML templates
└── README.md                        # This file
```

## 🚀 Usage

### Training the Model

To train the CNN model on your dataset:

```bash
python train_model.py
```

**What happens:**
- Loads and preprocesses training/test images (64×64 pixels, RGB)
- Applies data augmentation for better generalization
- Trains the model for 20 epochs
- Saves the trained model as `Flask/Skin_Diseases.h5`
- Displays training metrics and validation accuracy

### Using the Web Interface

After training, start the Flask application:

```bash
cd Flask
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

## 🧠 Model Architecture

The CNN model uses the following architecture:

```
Input Layer (64×64×3)
    ↓
Conv2D (32 filters, 3×3 kernel, ReLU)
    ↓
Conv2D (32 filters, 3×3 kernel, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (128 units, ReLU)
    ↓
Dense (64 units, ReLU)
    ↓
Dense (5 units, Softmax) → Classification Output
```

**Key Design Choices:**
- **Input**: 64×64 RGB images for computational efficiency
- **Convolutional Layers**: Extract spatial features from images
- **MaxPooling**: Reduces dimensionality and computational load
- **Dense Layers**: Learn complex decision boundaries
- **Softmax Activation**: Multi-class probability distribution

## 📊 Dataset

### Dataset Statistics
- **Total Training Images**: 2,205
- **Total Testing Images**: 550
- **Image Size**: 64×64 pixels
- **Format**: RGB color images
- **Classes**: 5 disease categories

### Data Augmentation Techniques
To improve model robustness:
- Shear range: 0.2
- Zoom range: 0.2
- Height shift: 0.2
- Width shift: 0.2
- Horizontal flip: Enabled
- Vertical flip: Enabled

## 📈 Results

The model demonstrates:
- **Training Accuracy**: Progressive improvement over 20 epochs
- **Validation Testing**: Evaluated on 550 test images
- **Class Balancing**: Handles 5 disease classes effectively
- **Generalization**: Data augmentation prevents overfitting

Training curves show steady convergence with validation metrics confirming model reliability.

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **TensorFlow/Keras** | Deep learning framework for CNN |
| **Python** | Programming language |
| **NumPy** | Numerical computations |
| **Pillow** | Image processing |
| **Flask** | Web application framework |
| **h5py** | Model serialization |
| **Jupyter Notebook** | Data exploration & experimentation |

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 🙋 Support

For questions or issues, please:
- Open an [GitHub Issue](https://github.com/Nishan631/Skin-disease-identification-system/issues)
- Contact the repository maintainer

---

**Made with ❤️ by [Nishan631](https://github.com/Nishan631)**

*Last Updated: April, 2026*
