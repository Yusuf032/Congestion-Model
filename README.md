# Congestion-Model
Congestion Detection Model — README

Road congestion detection using camera / sensor data. This repository contains the code, models, and instructions to train, evaluate, and deploy a model that classifies road segments or frames as Free-flowing, Moderate congestion, or Heavy congestion.

Table of contents

Overview

Features

Repository structure

Datasets

Model architecture

Preprocessing & augmentation

Training

Evaluation & metrics

Inference / Usage

Deployment

Troubleshooting & tips

License & citation

Credits & acknowledgements

Overview

This project trains an ML model to detect road congestion level from input data (images, video frames, or sensor-derived features). The goal is fast, robust classification for traffic monitoring, dashboards, and control-room alerting.

Supported congestion classes (by default):

0 — Free-flowing

1 — Moderate congestion

2 — Heavy congestion

The code is modular so you can plug in your own dataset, change the model backbone, and switch from frame-based inference to short-window temporal models if needed.

Features

End-to-end pipeline: preprocessing → training → evaluation → inference.

Support for image and video-frame inputs (single-frame CNN and optional temporal model).

Data augmentation, class balancing, and stratified sampling.

Checkpointing and TensorBoard logging.

Example REST API and Streamlit demo for quick deployment.

Repository structure
README.md
data/
  ├─ raw/                   # raw dataset (not tracked)
  ├─ processed/             # preprocessed images / features
src/
  ├─ data/
  │   ├─ dataset.py         # Dataset classes & loaders
  │   ├─ transforms.py      # preprocessing & augmentation
  ├─ models/
  │   ├─ cnn_backbone.py    # CNN backbone(s)
  │   ├─ temporal_model.py  # optional LSTM/Transformer for short windows
  │   └─ classifier.py
  ├─ train.py               # training entrypoint
  ├─ eval.py                # evaluation scripts
  ├─ infer.py               # command-line inference
  ├─ api/
  │   ├─ app.py             # FastAPI example
  └─ demo/
      └─ streamlit_app.py   # Streamlit demo
configs/
  ├─ default.yaml
checkpoints/
notebooks/
  ├─ EDA.ipynb
requirements.txt
Dockerfile

Datasets

This project can work with several kinds of inputs:

Traffic camera frames — annotated frames (bounding boxes not required; frame-level labels are fine).

Video segments — short clips labeled by dominant congestion level.

Sensor-derived features (optional) — vehicle counts, average speed, occupancy.

Expected dataset format (image-based):

data/processed/
  ├─ images/
  │   ├─ free/            # images for class 0
  │   ├─ moderate/        # images for class 1
  │   └─ heavy/           # images for class 2
  └─ labels.csv           # optional: filename,label


If using video, extract frames at a fixed FPS and optionally group by short windows (e.g., 5s) for temporal modeling.

Model architecture

Default configuration uses a lightweight CNN backbone + classifier:

Backbone: MobileNetV2 / ResNet18 (configurable)

Head: Global average pooling → Dense(128) → ReLU → Dropout(0.5) → Dense(3) (softmax)

Optional temporal model:

CNN feature extractor per-frame → sequence of features → BiLSTM or Transformer encoder → Dense classifier

Loss & optimizer:

Loss: Cross-Entropy

Optimizer: Adam (customizable LR and scheduler)

Metrics: Accuracy, Precision, Recall, F1, Confusion matrix

Preprocessing & augmentation

Typical preprocessing steps:

Resize to 224×224 (or backbone-specific size)

Scale pixel values to [0, 1] and apply mean/std normalization (ImageNet stats by default)

Augmentations (train):

Random horizontal flip (careful if camera orientation matters)

Random brightness/contrast

Random crop / scale

Random rotation (small angles)

Cutout or random erasing (optional)

Tip: apply the same normalization at inference as training.

Training

Quickstart (assumes a Python 3.8+ venv with requirements installed):

Install:

pip install -r requirements.txt


Run default training:

python src/train.py --config configs/default.yaml


Common CLI options (examples):

--data-dir path to processed dataset

--backbone backbone model (mobilenetv2, resnet18)

--batch-size e.g., 32

--epochs e.g., 30

--lr initial learning rate

--output-dir where checkpoints/logs are stored

Training considerations:

Use stratified splits to keep class ratios in train/val/test.

If one class is under-represented, use weighted sampling or class weights in loss.

Monitor TensorBoard:

tensorboard --logdir runs/

Evaluation & metrics

Run evaluation on a test set:

python src/eval.py --checkpoint checkpoints/best.pth --data-dir data/processed


Outputs:

Accuracy, precision, recall, F1 per class

Confusion matrix (saved as PNG)

ROC curves (if you convert to one-vs-rest)

Target thresholds:

Aim for balanced precision/recall for the heavy class (high recall preferred if catching congestion is critical).

Inference / Usage

Example CLI inference (single image):

python src/infer.py --checkpoint checkpoints/best.pth --image sample.jpg
# Output: {"label": "heavy", "confidence": 0.87}


Batch inference over a folder:

python src/infer.py --checkpoint checkpoints/best.pth --input-dir data/test_images/ --output results.json


Python inference example (programmatic):

from src.infer import load_model, predict_image
model = load_model("checkpoints/best.pth")
label, conf = predict_image(model, "frame_001.jpg")
print(label, conf)

REST API example (FastAPI)

A minimal FastAPI app src/api/app.py serves prediction requests:

uvicorn src.api.app:app --host 0.0.0.0 --port 8000


POST /predict with form-data field file returns {"label":"moderate","confidence":0.72}.

Streamlit demo

Simple web demo in src/demo/streamlit_app.py. Run:

streamlit run src/demo/streamlit_app.py


The demo allows image upload and shows predicted label, confidence and sample explanations (Grad-CAM visualizations if enabled).

Deployment

Options:

Docker: Use the included Dockerfile for containerized inference / API.

Kubernetes: Deploy API container with autoscaling on CPU/GPU nodes.

Edge device: Use TensorRT / ONNX export for running on NVIDIA Jetson or embedded GPUs.

Export model to ONNX:

python src/export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx

Troubleshooting & tips

Imbalanced classes: try weighted loss or oversample minority classes; use focal loss for extreme imbalance.

Overfitting: increase augmentation, add stronger regularization (dropout, weight decay), or reduce backbone capacity.

False positives for heavy: check camera angle/time-of-day biases; augment with samples across lighting/season.

Low recall on heavy: lower decision threshold for heavy when using softmax probabilities, or use class-specific thresholds.

Temporal smoothing: for video, smoothing predictions over a short window (e.g., majority vote over 5 frames) reduces flicker.

Example config (configs/default.yaml)
data:
  img_size: 224
  batch_size: 32
  data_dir: "data/processed"

model:
  backbone: "mobilenetv2"
  pretrained: true
  dropout: 0.5
  num_classes: 3

train:
  epochs: 30
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: "cosine"

Reproducibility

Seed RNGs for numpy, torch and random in train.py.

Record exact package versions in requirements.txt.

Save training configs and best checkpoint to checkpoints/ alongside model metadata (training date, dataset hash).

License & citation

Default: MIT License (customize if needed).

If you use public datasets, include citations and follow their licensing terms in LICENSES/ or docs/.

Credits & acknowledgements

Built using PyTorch (or your chosen framework), OpenCV for frame extraction, and standard augmentation libs.

Acknowledge dataset providers and municipalities if using city camera feeds.
