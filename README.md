# ASL Recognition System

A modular system for American Sign Language (ASL) recognition using hand landmarks.

## Setup

```bash
python -m venv asl_env
source asl_env/bin/activate
pip install -r requirements.txt
```

## Project Structure
The project follows a modular organization with the following key components:

- `asl_modules/utils.py`: Core utility functions shared across the system
- `asl_modules/data_collection.py`: Hand landmark data collection functionality  
- `asl_modules/preprocessing.py`: Data processing and analysis pipeline
- `asl_modules/training.py`: Model training and performance evaluation
- `asl_modules/inference.py`: Live prediction and visualization system
- `asl_app.py`: Command-line interface and main application entry point

## Usage
### Data Collection
Collect hand landmark data for ASL signs:

```bash
python asl_app.py collect --letter A --samples 100
 ```

Or collect data for multiple letters interactively:

```bash
python asl_app.py collect
 ```

### Data Analysis
Analyze the collected dataset:

```bash
python asl_app.py analyze
 ```

### Model Training
Train the ASL recognition model:

```bash
python asl_app.py train
 ```

### Real-time Recognition
Run real-time ASL recognition using webcam:

```bash
python asl_app.py run
 ```

### Model Evaluation
Evaluate the model on saved data:

```bash
python asl_app.py evaluate
 ```

## Dataset
The dataset consists of hand landmarks collected using MediaPipe Hands.
Each sample contains 21 landmarks with x, y, z coordinates, normalized relative to the wrist position.

## Model Architecture
The system employs a Multi-Layer Perceptron (MLP) classifier with the following architecture:

- Input Layer: 63 features (21 landmarks × 3 coordinates)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 26 neurons (one for each letter) with softmax activation
- Optimizer: Adam
- Loss Function: Categorical Cross-entropy

## System Design
The project follows a modular architecture with these key benefits:

1. **Separation of Concerns**: Each module handles a specific task independently:
   - Data collection and preprocessing
   - Model training and evaluation
   - Real-time inference
   - Utility functions

2. **Reusability**: Common components are shared across modules:
   - Preprocessing pipelines
   - Model configuration
   - Evaluation metrics

3. **Testability**: Modular design enables:
   - Unit testing of individual components
   - Integration testing of module interactions
   - End-to-end system testing

4. **Extensibility**: The system can be enhanced by:
   - Adding new model architectures
   - Implementing additional preprocessing steps
   - Supporting more sign language gestures

5. **User Interface**: Command-line interface provides:
   - Intuitive access to all features
   - Configurable parameters
   - Clear feedback and logging

For usage instructions, refer to the command examples in the sections above.
