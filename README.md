<div align="center">

# Leaffliction

![Language](https://img.shields.io/badge/language-Python-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/status-completed-brightgreen?style=for-the-badge)

**Leaf disease classification using a CNN trained on the PlantVillage dataset.**

🌐 [leaffliction.rsterin.fr](https://leaffliction.rsterin.fr)

</div>

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)

</details>

## About

A deep learning pipeline that classifies leaf diseases from photographs. The CNN is trained on 8 classes of Apple and Grape leaves (healthy and diseased) from the [PlantVillage](https://plantvillage.psu.edu/) dataset. The project includes image augmentation, transformation preprocessing, distribution analysis, and an interactive Streamlit web app.

This is a [42](https://42.fr) school project.

## Features

- **CNN classifier** — Conv2D architecture with batch normalization and dropout, trained with Adam
- **Image augmentation** — rotation, blur, contrast, scaling, illumination, and projective transforms to balance the dataset
- **Preprocessing pipeline** — background removal, masking, ROI detection, pseudolandmarks, and spot isolation via PlantCV
- **Streamlit web app** — predict, visualize distributions, preview transformations/augmentations, and generate training commands
- **Pretrained model included** — ready-to-use model in `pretrained_agent/`
- **Custom model upload** — import your own trained model via the sidebar

## Tech Stack

| Component  | Technology                                              |
|:-----------|:--------------------------------------------------------|
| Language   | [Python](https://www.python.org/) 3.10+                |
| DL         | [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) |
| CV         | [OpenCV](https://opencv.org/) / [PlantCV](https://plantcv.readthedocs.io/) |
| ML         | [scikit-learn](https://scikit-learn.org/)               |
| Web UI     | [Streamlit](https://streamlit.io/)                      |

## Getting Started

### Prerequisites

| Tool                                    | Version |
|:----------------------------------------|:--------|
| [Python](https://www.python.org/)       | ≥ 3.10  |
| Make                                    | —       |

### Installation

```bash
git clone https://github.com/rsterin/Leaffliction.git
cd Leaffliction
make
```

### Build & Run

> **Full pipeline** (train from scratch)

```bash
python train.py --source data/leaves/images --destination my_model --epochs 10 --transfo gaussian_blur
```

> **Predict**

```bash
python predict.py --source path/to/image.JPG --model pretrained_agent
```

> **Streamlit app**

```bash
streamlit run app.py
```

### Verify

Open [http://localhost:8501](http://localhost:8501) — upload a leaf image in the Prediction tab to classify it.

## Usage

| Command       | Description                                                |
|:--------------|:-----------------------------------------------------------|
| `train.py`    | Train a new model on the dataset                           |
| `predict.py`  | Classify leaf images using a trained model                 |
| `app.py`      | Launch the Streamlit web app                               |

| Flag (train)       | Description                                     |
|:-------------------|:------------------------------------------------|
| `--source`         | Folder with training images (default: `data/leaves`) |
| `--destination`    | Model save folder                               |
| `--epochs`         | Number of training epochs (default: 10)         |
| `--transfo`        | Transformations to apply (e.g. `gaussian_blur`) |

| Flag (predict)     | Description                                     |
|:-------------------|:------------------------------------------------|
| `--source`         | Image path(s) to classify                       |
| `--model`          | Model folder to load (default: `DetectionAgent_1`) |
