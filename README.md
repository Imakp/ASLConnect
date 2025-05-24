# ASL Connect

![ASL Recognition System](static\images\ASL.png)

A comprehensive American Sign Language (ASL) recognition and learning platform that breaks communication barriers through real-time sign language recognition.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset Format](#dataset-format)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## Overview

ASL Connect is a web-based platform designed to facilitate communication through American Sign Language. The system uses machine learning to recognize ASL signs in real-time, enabling seamless communication between signers and non-signers through video calls with automatic subtitling.

## Features

- **Real-time ASL Recognition**: Advanced machine learning model detects and interprets ASL signs instantly
- **Interactive Learning Platform**: Learn ASL through guided lessons, practice sessions, and quizzes
- **Video Call Integration**: Connect with others using ASL with real-time sign recognition and subtitling
- **Multi-hand Support**: Recognizes both single-hand and two-hand ASL signs
- **Comprehensive Dashboard**: Track learning progress and access all features from a central interface
- **Responsive Design**: Works seamlessly across different devices and screen sizes
- **Camera Practice Mode**: Practice ASL signs with real-time feedback
- **Quiz System**: Test your ASL knowledge with interactive quizzes

## System Architecture

The system consists of several key components:

1. **Web Interface**: Flask-based web application with responsive UI
2. **ASL Recognition Engine**: Machine learning model trained on hand landmark data
3. **Video Call System**: WebRTC-based video calling with real-time ASL recognition
4. **Learning Management**: Interactive lessons, practice sessions, and quizzes

## Installation

### Prerequisites

- Python 3.11 - 3.12
- pip package manager
- Web browser with WebRTC support

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ASL-demo.git
cd ASL-demo
```

2. Create and activate a virtual environment:

```bash
python -m venv asl_env
source asl_env/bin/activate  # On Windows: asl_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask application:

```bash
python app.py
```

5. Access the application at http://localhost:5000

## Usage

### Web Application

1. Landing Page : Access the main features and learn about the platform
2. Dashboard : Central hub for accessing all features
3. Learning Section : Interactive lessons for learning ASL
4. Practice Mode : Practice ASL signs with real-time feedback
5. Quiz : Test your ASL knowledge
6. Video Call : Connect with others using ASL with real-time recognition

### ASL Recognition System

The core ASL recognition functionality can also be used via command-line:

Data Collection

```bash
python asl_app.py collect --letter A --samples 100
```

Data Analysis

```bash
python asl_app.py analyze --input asl_landmarks_dataset.csv
```

Model Training

```bash
python asl_app.py train --input asl_landmarks_dataset.csv
```

Real-time Recognition

```bash
python asl_app.py run
```

or

```bash
python asl_app.py run --model asl_mlp_multi_hand_model.joblib
```

Model Evaluation

```bash
python asl_app.py evaluate --input test_dataset.csv
```

## Project Structure

```bash
ASLConnect/
├── app.py                      # Main Flask application
├── asl_app.py                  # Command-line interface for ASL recognition
├── requirements.txt            # Python dependencies
├── subtitles.py                # ASL subtitle generation
├── video_call.py               # Video call management
├── asl_modules/                # Core ASL recognition modules
│   ├── data_collection.py      # Data collection utilities
│   ├── preprocessing.py        # Data preprocessing
│   ├── training.py             # Model training
│   ├── inference.py            # Real-time inference
│   └── utils.py                # Utility functions
├── asl_model/                  # Trained models and scalers
│   ├── asl_mlp_multi_hand_model.joblib     # MLP model for multi-hand ASL recognition
│   └── hand_landmarks_scaler.joblib        # Scaler for hand landmark normalization
├── static/                     # Static assets
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript files
│   └── images/                 # Images and icons
└── templates/                  # HTML templates
    ├── landing.html            # Landing page
    ├── dashboard.html          # Dashboard
    ├── login.html              # Login page
    ├── index.html              # Video call page
    ├── quiz.html               # Quiz page
    └── camera_practice.html    # Practice page
```

## Dataset Format

The ASL recognition system uses hand landmark data collected using MediaPipe Hands. The dataset is stored in CSV format with the following structure:

- Single-hand dataset : Contains normalized landmarks for one hand
- Multi-hand dataset : Contains normalized landmarks for both hands with hand count

## Model Architecture

The ASL recognition model uses a Multi-Layer Perceptron (MLP) neural network with the following architecture:

- Input layer: Hand landmark features (normalized x, y, z coordinates)
- Hidden layers: Configurable size (default: 128 neurons)
- Output layer: ASL sign classes
- Activation: ReLU
- Optimizer: Adam with adaptive learning rate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
