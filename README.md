# Leaffliction 🌿🍂

A computer vision project for **image classification by disease recognition on leaves**. This project implements a complete pipeline for analyzing, augmenting, transforming, training, and predicting plant diseases from leaf images.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface (Gradio)](#web-interface-gradio)
  - [Command Line Tools](#command-line-tools)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

## 🎯 Overview

Leaffliction is a comprehensive machine learning system designed to identify plant diseases from leaf images. The project supports multiple plant types (Apple, Grape) and various disease categories including:

- **Apple**: Black rot, Healthy, Rust, Scab
- **Grape**: Black rot, Esca, Healthy, Leaf spot

The system achieves **>90% accuracy** on validation sets with a minimum of 100 images, using a combination of data augmentation, image transformation, and deep learning techniques.

## ✨ Features

### 1. **Data Distribution Analysis**
- Interactive pie charts and bar plots
- Visualization of dataset balance across categories
- Support for original and augmented image analysis

### 2. **Data Augmentation**
- **6 augmentation techniques**:
  - Rotation (random angles)
  - Gaussian blur
  - Contrast adjustment
  - Scaling/Zoom
  - Illumination changes
  - Projective transformation
- Automatic dataset balancing through oversampling
- Preserves original images alongside augmented versions

### 3. **Image Transformation**
- **6 transformation methods**:
  - Gaussian blur with leaf segmentation
  - Mask generation for leaf isolation
  - ROI (Region of Interest) detection
  - Pseudolandmark generation
  - Disease spot isolation
  - Background removal
- PlantCV integration for advanced plant analysis
- Batch processing with progress tracking

### 4. **Training & Classification**
- Deep learning model training with TensorFlow/Keras
- Automatic train/validation split
- Model persistence (saved as `.zip`)
- Progress tracking with Rich library

### 5. **Prediction**
- Load pre-trained models
- Predict disease class from new leaf images
- Display original and transformed images
- ASCII art visualization in terminal

### 6. **Interactive Web Interface**
- Built with Gradio
- Tabbed interface for each module:
  - Home
  - Distribution
  - Augmentation
  - Transformation
  - Training
  - Prediction
- Real-time visualization
- Dark theme with custom Plotly styling

## 📁 Project Structure

```
Leaffliction/
├── app.py                      # Gradio web interface
├── Distribution.py             # CLI for distribution analysis
├── Augmentation.py             # CLI for data augmentation
├── Transformation.py           # CLI for image transformation
├── train.py                    # Model training script
├── predict.py                  # Prediction script
├── Makefile                    # Build automation
├── requirements.txt            # Python dependencies
├── signature.txt               # Dataset SHA1 hash
├── data/
│   ├── leaves/                 # Original dataset
│   │   ├── Apple_Black_rot/
│   │   ├── Apple_healthy/
│   │   ├── Apple_rust/
│   │   ├── Apple_scab/
│   │   ├── Grape_Black_rot/
│   │   ├── Grape_Esca/
│   │   ├── Grape_healthy/
│   │   └── Grape_spot/
│   └── leaves_preprocessed/    # Processed images
└── srcs/
    ├── DetectionAgent.py       # ML model wrapper
    ├── tab_augmentation.py     # Gradio augmentation tab
    ├── tab_distribution.py     # Gradio distribution tab
    ├── tab_transformation.py   # Gradio transformation tab
    └── tools.py                # Utility functions
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment support

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/jtoulous/Leaffliction.git
cd Leaffliction
```

2. **Automated setup** (recommended)
```bash
make all
```

This will:
- Create a virtual environment in `~/goinfre/venv`
- Install all dependencies from `requirements.txt`
- Unzip the dataset
- Display activation instructions

3. **Manual setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Unzip dataset
make unzip
```

4. **Activate the environment**
```bash
source ~/goinfre/venv/bin/activate
```

## 📖 Usage

### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python app.py
```

Then open your browser to `http://localhost:7860` (or the URL shown in terminal).

#### Available Tabs:
- **Distribution**: Analyze dataset distribution with interactive charts
- **Augmentation**: Apply and visualize data augmentation techniques
- **Transformation**: Apply image transformations for preprocessing
- **Training**: Train classification models (coming soon)
- **Prediction**: Predict diseases on new images (coming soon)

### Command Line Tools

#### 1. Distribution Analysis

Display dataset distribution:

```bash
# All charts (pie + bar)
python Distribution.py --source data/leaves

# Only pie chart
python Distribution.py --source data/leaves --distribution pie

# Include augmented images
python Distribution.py --source data/leaves --all-images
```

#### 2. Data Augmentation

Augment images to balance the dataset:

```bash
# Augment all images with all techniques
python Augmentation.py --source data/leaves --destination data/leaves

# Apply specific augmentation
python Augmentation.py --source data/leaves --augmentation rotation

# Process specific number of images
python Augmentation.py --source data/leaves --range-nb 100

# Process percentage of dataset
python Augmentation.py --source data/leaves --range-percent 50

# Display images during processing
python Augmentation.py --source data/leaves --display

# Set random seed for reproducibility
python Augmentation.py --source data/leaves --seed 42
```

**Available augmentations**: `rotation`, `blur`, `contrast`, `scaling`, `illumination`, `projective`

#### 3. Image Transformation

Apply preprocessing transformations:

```bash
# Transform all images with all methods
python Transformation.py --source data/leaves --destination data/leaves_preprocessed

# Apply specific transformation
python Transformation.py --source data/leaves --transform mask

# Process single image
python Transformation.py --source data/leaves/Apple_healthy/image001.JPG --display

# Process with range limits
python Transformation.py --source data/leaves --range-nb 50 --range-percent 100
```

**Available transformations**: `gaussian_blur`, `mask`, `roi_objects`, `pseudolandmarks`, `spots_isolation`, `background_removal`

#### 4. Model Training

Train a classification model:

```bash
# Basic training
python train.py -imgs_folder data/leaves -save_folder DetectionAgent_1

# Custom epochs
python train.py -imgs_folder data/leaves -epochs 50

# With specific transformations
python train.py -imgs_folder data/leaves -transfo gaussian_blur mask
```

#### 5. Prediction

Predict disease from new images:

```bash
python predict.py image1.jpg image2.jpg -load_folder DetectionAgent_1
```

Output includes:
- ASCII art representation of images
- Transformed image visualizations
- Predicted disease class

## 📊 Dataset

The dataset is organized by plant type and disease category:

```
data/leaves/
├── Apple_Black_rot/     # Apple leaves with black rot disease
├── Apple_healthy/       # Healthy apple leaves
├── Apple_rust/          # Apple leaves with rust disease
├── Apple_scab/          # Apple leaves with scab disease
├── Grape_Black_rot/     # Grape leaves with black rot
├── Grape_Esca/          # Grape leaves with Esca disease
├── Grape_healthy/       # Healthy grape leaves
└── Grape_spot/          # Grape leaves with leaf spot
```

### Dataset Verification

The project includes a `signature.txt` file containing the SHA1 hash of the dataset for integrity verification:

```bash
# Linux
sha1sum data/leaves.zip

# macOS
shasum data/leaves.zip

# Windows
certUtil -hashfile data/leaves.zip sha1
```

## 🎯 Model Performance

The trained model achieves:
- **Accuracy**: >90% on validation set
- **Validation set size**: Minimum 100 images
- **Training/Validation split**: Automatic separation
- **No data leakage**: Validation set completely isolated

## 🔧 Technical Details

### Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, PlantCV
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Gradio
- **CLI**: Rich (progress bars), argparse

### Image Processing Pipeline

1. **Load**: Read images from directory structure
2. **Augment**: Apply augmentation techniques to balance dataset
3. **Transform**: Extract features through transformations
4. **Train**: Deep learning model training
5. **Predict**: Classification of new images

### Custom Features

- **Smart image loading**: Detects file/folder/root directory structures
- **Progress tracking**: Real-time progress bars with Rich
- **Batch processing**: Efficient processing of large datasets
- **Range control**: Process specific subsets via count or percentage
- **Reproducibility**: Seed control for random operations
- **Custom Plotly theme**: Matches Gradio dark theme with gradient colors

## 📦 Requirements

Key dependencies (see `requirements.txt` for complete list):

```
tensorflow>=2.20.0
opencv-python>=4.12.0
gradio>=5.49.1
plantcv>=4.9
pandas>=2.3.3
plotly>=5.0.0
scikit-learn>=1.7.2
matplotlib>=3.10.7
seaborn>=0.13.2
rich>=14.2.0
numpy>=2.2.6
```

## 🧹 Maintenance

### Clean generated files
```bash
make clean  # Remove __pycache__ and processed data
```

### Complete cleanup
```bash
make fclean  # Remove virtual environment and all generated files
```

### Rebuild everything
```bash
make re  # Clean and rebuild from scratch
```

## 📝 License

This is an educational project developed as part of a computer vision curriculum.

## 👥 Authors

- **jtoulous**
- **rsterin**

## 🙏 Acknowledgments

- PlantCV library for plant-specific image analysis
- Gradio for the intuitive web interface framework
- The open-source community for excellent ML/CV tools
