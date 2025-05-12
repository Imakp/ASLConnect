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

def train_mlp_classifier(X_train, y_train, hidden_layer_size=128, max_iter=300, random_state=42, is_multi_hand=False):
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
    is_multi_hand : bool
        Flag indicating if the dataset includes multi-hand signs
        
    Returns:
    --------
    model : MLPClassifier
        Trained MLPClassifier model
    training_time : float
        Time taken to train the model (in seconds)
    """
    print(f"Training MLPClassifier with {hidden_layer_size} neurons in the hidden layer...")
    
    # For multi-hand datasets, we might need a more complex architecture
    if is_multi_hand:
        print("Using enhanced architecture for multi-hand sign recognition")
        hidden_layers = (hidden_layer_size, hidden_layer_size // 2)
    else:
        hidden_layers = (hidden_layer_size,)
    
    # Initialize the MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,        # Hidden layer architecture
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

def evaluate_model(model, X_test, y_test, is_multi_hand=False):
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
    is_multi_hand : bool
        Flag indicating if the dataset includes multi-hand signs
        
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
    
    # If multi-hand dataset, analyze performance by hand count
    if is_multi_hand and hasattr(X_test, 'shape') and X_test.shape[1] > 1:
        # Assuming the first column is the hand count
        hand_counts = X_test[:, 0].astype(int)
        unique_counts = np.unique(hand_counts)
        
        print("\nPerformance by Hand Count:")
        for count in unique_counts:
            # Get indices for this hand count
            indices = np.where(hand_counts == count)[0]
            
            if len(indices) > 0:
                # Calculate accuracy for this subset
                subset_accuracy = accuracy_score(y_test[indices], y_pred[indices])
                print(f"  Hand count = {count}: Accuracy = {subset_accuracy:.4f} (Samples: {len(indices)})")
    
    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, save_path=None, is_multi_hand=False):
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
    is_multi_hand : bool
        Flag indicating if the dataset includes multi-hand signs
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
    title = 'Confusion Matrix' + (' (Multi-Hand Model)' if is_multi_hand else '')
    plt.title(title)
    
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
        
        # Check if this is a multi-hand dataset by examining the shape
        # In multi-hand format, we have hand_count as the first feature
        is_multi_hand = X_train.shape[1] > 63  # 21 landmarks * 3 coordinates = 63 features for single hand
        
        if is_multi_hand:
            print("Detected multi-hand dataset format")
            model_filename = 'asl_mlp_multi_hand_model.joblib'
            cm_filename = 'multi_hand_confusion_matrix.png'
        else:
            print("Detected single-hand dataset format")
            model_filename = 'asl_mlp_model.joblib'
            cm_filename = 'confusion_matrix.png'
        
        # Train the model
        model, training_time = train_mlp_classifier(X_train, y_train, hidden_layer_size, max_iter, random_state, is_multi_hand)
        
        # Evaluate the model
        metrics, y_pred = evaluate_model(model, X_test, y_test, is_multi_hand)
        
        # Plot confusion matrix
        confusion_matrix_path = os.path.join(get_project_root(), cm_filename)
        plot_confusion_matrix(y_test, y_pred, save_path=confusion_matrix_path, is_multi_hand=is_multi_hand)
        
        # Save the trained model
        save_model(model, model_filename)
        
        print(f"\nModel training and evaluation complete! Model saved as {model_filename}")
        
        return model, metrics
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def train_unified_model(single_hand_csv, multi_hand_csv=None, combined_csv=None, hidden_layer_size=128, max_iter=300, random_state=42):
    """
    Train a unified model that can handle both single-hand and multi-hand signs.
    
    Parameters:
    -----------
    single_hand_csv : str
        Path to the CSV file containing single-hand data
    multi_hand_csv : str, optional
        Path to the CSV file containing multi-hand data
    combined_csv : str, optional
        Path to save the combined dataset (if None, will generate one)
    hidden_layer_size : int
        Number of neurons in the hidden layer
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : MLPClassifier
        Trained unified model
    metrics : dict
        Dictionary containing evaluation metrics
    """
    from .preprocessing import combine_datasets
    
    try:
        # If combined_csv is not provided, generate one
        if combined_csv is None:
            combined_csv = "unified_asl_dataset.csv"
        
        # If multi_hand_csv is provided, combine datasets
        if multi_hand_csv:
            print(f"Combining single-hand dataset ({single_hand_csv}) and multi-hand dataset ({multi_hand_csv})...")
            combine_datasets(single_hand_csv, multi_hand_csv, combined_csv)
        else:
            # If only single_hand_csv is provided, use it directly
            combined_csv = single_hand_csv
        
        # Train and evaluate the unified model
        print(f"Training unified model on {combined_csv}...")
        model, metrics = train_and_evaluate_model(combined_csv, hidden_layer_size, max_iter, random_state)
        
        return model, metrics
        
    except Exception as e:
        print(f"Error during unified model training: {e}")
        raise

def visualize_training_history(model, save_path=None):
    """
    Visualize the training history of the MLPClassifier.
    
    Parameters:
    -----------
    model : MLPClassifier
        Trained MLPClassifier model
    save_path : str, optional
        Path to save the visualization
    """
    if not hasattr(model, 'loss_curve_'):
        print("Model does not have training history information.")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot loss curve
    plt.plot(model.loss_curve_)
    
    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Training history visualization saved to {save_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    train_and_evaluate_model()