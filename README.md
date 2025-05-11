# ASL Recognition System

A modular system for American Sign Language (ASL) recognition using hand landmarks.

## Setup

```bash
python -m venv asl_env
source asl_env/bin/activate
pip install -r requirements.txt
```

## Project Structure
The project is organized into the following modules:

- asl_modules/utils.py : Common utility functions used across modules
- asl_modules/data_collection.py : Functions for collecting hand landmark data
- asl_modules/preprocessing.py : Data loading, preprocessing, and analysis
- asl_modules/training.py : Model training and evaluation
- asl_modules/inference.py : Real-time prediction and visualization
- asl_app.py : Main application script with command-line interface
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

## Model
The system uses an MLPClassifier with one hidden layer of 128 neurons and ReLU activation.

```plaintext

This modular structure provides several benefits:

1. **Separation of Concerns**: Each module has a specific responsibility, making the code easier to understand and maintain.

2. **Reusability**: Functions and classes can be reused across different parts of the application.

3. **Testability**: Each module can be tested independently.

4. **Extensibility**: New features can be added by extending existing modules or adding new ones.

5. **Command-line Interface**: The main script provides a user-friendly interface to access all functionality.

To use this refactored codebase, you can run the main script with different commands as shown in the README.
 ```
```