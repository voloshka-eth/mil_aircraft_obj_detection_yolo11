# Sokil — Military Aircraft Object Detection with YOLO11

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red)
![Gradio](https://img.shields.io/badge/UI-Gradio-green)
![Status](https://img.shields.io/badge/status-demo%20v1.0-lightgrey)

**Sokil** is a Python-based computer vision project for detecting military aircraft in images using **Ultralytics YOLO11**.

The repository contains a complete experimental pipeline: model training, local inference, a Gradio web interface, PDF report generation, training metric visualization, and YOLO feature map inspection.

> This project is intended for educational, research, and portfolio purposes only.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Training](#training)
- [Running the Web Interface](#running-the-web-interface)
- [Model Inspection](#model-inspection)
- [Training Metrics Visualization](#training-metrics-visualization)
- [Feature Map Visualization](#feature-map-visualization)
- [Configuration Notes](#configuration-notes)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Ethical Use](#ethical-use)
- [Author](#author)

---

## Overview

The goal of this project is to build an end-to-end object detection workflow for aircraft imagery.

The pipeline includes:

1. Preparing a YOLO-compatible dataset.
2. Training a YOLO11 detection model.
3. Running inference on custom images.
4. Serving predictions through a lightweight Gradio interface.
5. Generating PDF detection reports.
6. Visualizing training metrics.
7. Inspecting internal YOLO convolutional feature maps.

---

## Features

- YOLO11-based aircraft object detection.
- Training script for a custom dataset.
- Gradio web interface for image upload and prediction.
- Detection summary with class names and confidence scores.
- Automatic PDF report generation.
- Training metric visualization from `results.csv`.
- YOLO convolutional feature map visualization.
- Clean structure for future deployment improvements.

---

## Tech Stack

- **Python**
- **PyTorch**
- **Ultralytics YOLO11**
- **Gradio**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **ReportLab**
- **Pillow**
- **TorchVision**

---

## Repository Structure

```text
mil_aircraft_obj_detection_yolo11/
│
├── train.py          # YOLO11 model training script
├── interface.py      # Gradio interface and PDF report generation
├── test.py           # Model loading and architecture inspection
├── grafik.py         # Training metrics visualization
├── yolo_layers.py    # YOLO convolutional feature map visualization
├── requirements.txt  # Python dependencies
├── .gitignore        # Ignored local files, datasets, weights, outputs
└── README.md
```

Expected local dataset structure:

```text
Jet Detection/
│
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The dataset, trained weights, generated reports, and experiment outputs are intentionally not included in the repository.

---

## Dataset Format

The project expects a YOLO-format dataset with:

- image files in `.jpg`, `.jpeg`, or `.png` format;
- labels in YOLO bounding box format;
- a `data.yaml` file describing train, validation, test paths, and class names.

Example `data.yaml`:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  - F-16
  - Su-27
  - MiG-29
```

Place the dataset inside:

```text
Jet Detection/
```

The current training script uses:

```python
data='Jet Detection/data.yaml'
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/voloshka-eth/mil_aircraft_obj_detection_yolo11.git
cd mil_aircraft_obj_detection_yolo11
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If dependency installation fails, use a cleaned version of `requirements.txt`:

```txt
ultralytics>=8.2.0
torch>=2.3.0
gradio
reportlab
seaborn
pandas
matplotlib
pillow
torchvision
```

---

## Training

To train the model, run:

```bash
python train.py
```

Current training configuration:

```python
model = YOLO('yolo11s.pt')

results = model.train(
    data='Jet Detection/data.yaml',
    epochs=40,
    imgsz=960,
    batch=8,
    lr0=0.001,
    device=0,
    workers=4,
    project='jet_classifier',
    name='yolo11n_detection',
)
```

Training outputs are saved inside:

```text
jet_classifier/
```

Typical YOLO output structure:

```text
jet_classifier/
└── yolo11n_detection/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    └── ...
```

---

## Running the Web Interface

Before launching the interface, update the model path inside `interface.py`:

```python
model = YOLO('jet_classifier/yolo11n_detection/weights/best.pt')
```

Then start the app:

```bash
python interface.py
```

The Gradio interface allows the user to upload an image and returns:

- image with detected aircraft;
- detection summary;
- downloadable PDF report.

Example output:

```text
Detected objects:
- F-16 (94.31%)
- Su-27 (88.42%)
```

---

## Model Inspection

The `test.py` script loads trained weights and prints detailed model information.

Before running it, update the path:

```python
weights = r'jet_classifier/yolo11n_detection/weights/best.pt'
```

Then run:

```bash
python test.py
```

---

## Training Metrics Visualization

The `grafik.py` script reads YOLO training metrics from `results.csv` and generates plots for:

- training box loss;
- training class loss;
- training DFL loss;
- validation box loss;
- validation class loss;
- validation DFL loss;
- precision;
- recall;
- mAP@0.5;
- mAP@0.5:0.95;
- learning rate schedule.

Before running the script, update:

```python
results_path = 'jet_classifier/yolo11n_detection/results.csv'
output_dir = 'outputs/plots'
```

Then run:

```bash
python grafik.py
```

Generated plots will be saved to the selected output directory.

---

## Feature Map Visualization

The `yolo_layers.py` script registers forward hooks on convolutional layers and visualizes intermediate feature maps.

Use it to better understand how YOLO processes an input image through its internal convolution layers.

Before running it, update the model path and image path:

```python
yolo = YOLO('jet_classifier/yolo11n_detection/weights/best.pt')
img_path = 'path/to/image.jpg'
```

Then run:

```bash
python yolo_layers.py
```

---

## Configuration Notes

Some scripts currently contain placeholder paths such as:

```python
'/path/'
```

Replace them with real local paths before running the project.

Recommended paths:

```python
# Trained model
'jet_classifier/yolo11n_detection/weights/best.pt'

# Training results
'jet_classifier/yolo11n_detection/results.csv'

# Plots output directory
'outputs/plots'
```

The repository `.gitignore` excludes datasets, model weights, generated outputs, and experiment folders. This keeps the repository lightweight and avoids pushing large binary files to GitHub.

---

## Limitations

This is a demo and research-oriented project, not a production-grade defense system.

Current limitations:

- the dataset is not included in the repository;
- trained model weights are not included;
- several scripts require manual path configuration;
- model quality depends on dataset size, labeling quality, and class balance;
- low-resolution, blurry, occluded, or unusual aircraft images may reduce accuracy;
- the Gradio interface currently has no confidence threshold control;
- there is no automated validation report script yet;
- there is no Docker setup yet;
- there is no ONNX or TensorRT export pipeline yet.

---

## Roadmap

Planned improvements:

- [ ] Add a central configuration file for paths and inference settings.
- [ ] Add a clean evaluation script for validation and test metrics.
- [ ] Add example screenshots of the Gradio interface.
- [ ] Add sample inference images.
- [ ] Add Docker support.
- [ ] Add ONNX export.
- [ ] Add batch image inference.
- [ ] Add video inference support.
- [ ] Add confidence threshold control in the UI.
- [ ] Compare YOLO11n, YOLO11s, and YOLO11m.
- [ ] Add automated report generation after training.
- [ ] Add GitHub Actions for basic code checks.

---

## Ethical Use

This repository is provided for educational, academic, and portfolio purposes.

It must not be used for unlawful surveillance, targeting, weapon guidance, or harmful military operations.

Computer vision systems can produce false positives and false negatives. Any real-world usage requires strict validation, human oversight, and compliance with applicable laws and safety standards.

---

## Author

Created by **voloshka-eth**.

GitHub: [@voloshka-eth](https://github.com/voloshka-eth)

---

## Acknowledgements

This project uses the Ultralytics YOLO ecosystem for object detection training and inference.
