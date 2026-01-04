# NeuroVision PRO

AI-powered brain tumor classification system using MRI scans.

## Features

- Brain tumor detection from MRI images
- 4-class classification: Glioma, Meningioma, No Tumor, Pituitary
- Advanced image preprocessing pipeline (Gaussian Sharpening, CLAHE)
- Real-time analysis with confidence scores
- Modern dark-themed UI

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Upload an MRI scan and click "RUN FULL AI ANALYSIS" to get predictions.

## Model

The trained model is automatically downloaded from Google Drive on first run.

## Tech Stack

- Streamlit
- TensorFlow/Keras
- OpenCV
- NumPy
