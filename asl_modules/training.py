"""
Training module for ASL recognition project.
Handles model training, evaluation, and saving.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from .utils import save_model, get_project_root
from .preprocessing import load_and_preprocess_data

def train_mlp_classifier(X_train, y_train, hidden_layer_size=128, max_iter=300, random_state=42):
    """
    Train an MLPClassifier model for ASL sign classification.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features (scaled)
    y_train : numpy.ndarray
        Training labels
    hidden_layer_size : int
        Number of neurons in the hidden layer
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : MLPClassifier
        Trained MLPClassifier model
    training_time : float
        Time taken to train the model (in seconds)
    """
    print(f"Training MLPClassifier with {hidden_layer_size} neurons in the hidden layer...")
    
    # Initialize the MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),  # One hidden layer with specified neurons
        activation='relu',                        # ReLU activation function
        solver='adam',                            # Adam optimizer
        alpha=0.0001,                             # L2 penalty (regularization term)
        batch_size='auto',                        # Batch size for gradient descent
        learning_rate='adaptive',                 # Adaptive learning rate
        max_iter=max_iter,                        # Maximum number of iterations
        random_state=random_state,                # Random seed
        verbose=True                              # Print progress messages
    )
    
    # Train the model and measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Model training completed in {training_time:.2f} seconds")
    
    return model, training_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    
    Parameters:
    -----------
    model : MLPClassifier
        Trained MLPClassifier model
    X_test : numpy.ndarray
        Testing features (scaled)
    y_test : numpy.ndarray
        Testing labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    y_pred : numpy.ndarray
        Predicted labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate precision, recall, and F1-score (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot and save the confusion matrix.
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    save_path : str, optional
        Path to save the confusion matrix plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique classes
    classes = np.unique(np.concatenate((y_test, y_pred)))
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(csv_file="asl_landmarks_dataset.csv", hidden_layer_size=128, max_iter=300, random_state=42):
    """
    Train and evaluate an MLPClassifier model for ASL sign classification.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the hand landmarks data
    hidden_layer_size : int
        Number of neurons in the hidden layer
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : MLPClassifier
        Trained MLPClassifier model
    metrics : dict
        Dictionary containing evaluation metrics
    """
    try:
        # Load and preprocess the data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(csv_file)
        
        # Train the model
        model, training_time = train_mlp_classifier(X_train, y_train, hidden_layer_size, max_iter, random_state)
        
        # Evaluate the model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Plot confusion matrix
        confusion_matrix_path = os.path.join(get_project_root(), 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, save_path=confusion_matrix_path)
        
        # Save the trained model
        save_model(model, 'asl_mlp_model.joblib')
        
        print("\nModel training and evaluation complete!")
        
        return model, metrics
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    train_and_evaluate_model()