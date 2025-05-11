"""
Preprocessing module for ASL recognition project.
Handles loading, preprocessing, and splitting the dataset.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import save_model, get_project_root

def load_and_preprocess_data(csv_file="asl_landmarks_dataset.csv", test_size=0.2, random_state=42, save_scaler=True):
    """
    Load the collected CSV dataset, preprocess it, and split into training and testing sets.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the hand landmarks data
    test_size : float
        Proportion of the dataset to include in the test split (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    save_scaler : bool
        Whether to save the scaler to disk (default: True)
    
    Returns:
    --------
    X_train : numpy.ndarray
        Training features (scaled)
    X_test : numpy.ndarray
        Testing features (scaled)
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Testing labels
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object for future use
    """
    # Construct the full path to the CSV file
    filepath = os.path.join(get_project_root(), csv_file)
    
    # Check if the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    # Load the dataset
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Display dataset information
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    if save_scaler:
        save_model(scaler, 'hand_landmarks_scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def analyze_dataset(csv_file="asl_landmarks_dataset.csv"):
    """
    Analyze the dataset and print statistics.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the hand landmarks data
    """
    # Construct the full path to the CSV file
    filepath = os.path.join(get_project_root(), csv_file)
    
    # Check if the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    # Load the dataset
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Display dataset information
    print("\nDataset Statistics:")
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {df.shape[1] - 1}")  # Subtract 1 for the label column
    
    # Class distribution
    class_dist = df['label'].value_counts()
    print("\nClass Distribution:")
    for label, count in class_dist.items():
        print(f"  {label}: {count} samples")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    
    # Basic statistics for features
    print("\nFeature Statistics:")
    feature_stats = df.drop('label', axis=1).describe()
    print(f"  Min value: {feature_stats.loc['min'].min()}")
    print(f"  Max value: {feature_stats.loc['max'].max()}")
    print(f"  Mean value: {feature_stats.loc['mean'].mean()}")
    print(f"  Std deviation: {feature_stats.loc['std'].mean()}")
    
    return df

if __name__ == "__main__":
    # Example usage
    analyze_dataset()
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print("Preprocessing complete!")