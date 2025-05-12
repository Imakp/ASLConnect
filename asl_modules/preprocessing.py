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
    
    # Check if this is a multi-hand dataset (has 'hand_count' column)
    is_multi_hand = 'hand_count' in df.columns
    
    # Separate features and labels
    if is_multi_hand:
        print("Detected multi-hand dataset format")
        X = df.drop(['label'], axis=1)
        
        # Extract hand count as a feature
        hand_count = X['hand_count'].values.reshape(-1, 1)
        X = X.drop(['hand_count'], axis=1).values
        
        # Combine hand count with the rest of the features
        X = np.hstack([hand_count, X])
        y = df['label'].values
    else:
        print("Detected single-hand dataset format")
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
    
    # Check if this is a multi-hand dataset
    is_multi_hand = 'hand_count' in df.columns
    
    # Display dataset information
    print("\nDataset Statistics:")
    print(f"Total samples: {df.shape[0]}")
    
    if is_multi_hand:
        print("Dataset type: Multi-hand (supports two-hand signs)")
        print(f"Total features: {df.shape[1] - 2}")  # Subtract 2 for the label and hand_count columns
        
        # Hand count distribution
        hand_dist = df['hand_count'].value_counts()
        print("\nHand Count Distribution:")
        for count, num_samples in hand_dist.items():
            print(f"  {count} hand(s): {num_samples} samples ({num_samples/len(df)*100:.1f}%)")
    else:
        print("Dataset type: Single-hand (one hand only)")
        print(f"Total features: {df.shape[1] - 1}")  # Subtract 1 for the label column
    
    # Class distribution
    class_dist = df['label'].value_counts()
    print("\nClass Distribution:")
    for label, count in class_dist.items():
        print(f"  {label}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    
    # Basic statistics for features
    print("\nFeature Statistics:")
    if is_multi_hand:
        feature_stats = df.drop(['label', 'hand_count'], axis=1).describe()
    else:
        feature_stats = df.drop('label', axis=1).describe()
        
    print(f"  Min value: {feature_stats.loc['min'].min()}")
    print(f"  Max value: {feature_stats.loc['max'].max()}")
    print(f"  Mean value: {feature_stats.loc['mean'].mean()}")
    print(f"  Std deviation: {feature_stats.loc['std'].mean()}")
    
    # If multi-hand, analyze by hand count
    if is_multi_hand:
        print("\nAnalysis by Hand Count:")
        for count in sorted(hand_dist.index):
            subset = df[df['hand_count'] == count]
            print(f"\n  Hand count = {count}:")
            print(f"    Samples: {len(subset)}")
            print(f"    Classes: {subset['label'].nunique()}")
            print(f"    Top classes: {subset['label'].value_counts().head(3).to_dict()}")
    
    return df

def combine_datasets(single_hand_csv, two_hand_csv, output_csv="combined_asl_dataset.csv"):
    """
    Combine single-hand and two-hand datasets into a unified format.
    
    Parameters:
    -----------
    single_hand_csv : str
        Path to the CSV file containing single-hand data
    two_hand_csv : str
        Path to the CSV file containing two-hand data
    output_csv : str
        Path to save the combined dataset
        
    Returns:
    --------
    combined_df : pandas.DataFrame
        Combined dataset
    """
    # Construct full paths
    single_hand_path = os.path.join(get_project_root(), single_hand_csv)
    two_hand_path = os.path.join(get_project_root(), two_hand_csv)
    output_path = os.path.join(get_project_root(), output_csv)
    
    # Check if files exist
    if not os.path.isfile(single_hand_path):
        raise FileNotFoundError(f"Single-hand dataset not found: {single_hand_path}")
    if not os.path.isfile(two_hand_path):
        raise FileNotFoundError(f"Two-hand dataset not found: {two_hand_path}")
    
    # Load datasets
    print(f"Loading single-hand dataset from {single_hand_path}...")
    single_df = pd.read_csv(single_hand_path)
    
    print(f"Loading two-hand dataset from {two_hand_path}...")
    two_df = pd.read_csv(two_hand_path)
    
    # Check if two-hand dataset has the expected format
    if 'hand_count' not in two_df.columns:
        raise ValueError("Two-hand dataset does not have the expected format (missing 'hand_count' column)")
    
    # Convert single-hand dataset to two-hand format
    print("Converting single-hand dataset to two-hand format...")
    
    # Get the number of landmarks in the single-hand dataset
    single_features = single_df.drop('label', axis=1)
    num_landmarks = single_features.shape[1] // 3  # Assuming x, y, z for each landmark
    
    # Create a new DataFrame with the two-hand format
    converted_rows = []
    
    for _, row in single_df.iterrows():
        # Extract label and landmarks
        label = row['label']
        landmarks = row.drop('label').values
        
        # Create a new row with hand_count=1 and landmarks in the right hand position
        new_row = {'label': label, 'hand_count': 1}
        
        # Add zeros for left hand (21 landmarks * 3 coordinates)
        for i in range(21):
            for coord in ['x', 'y', 'z']:
                new_row[f'left_landmark{i}_{coord}'] = 0.0
        
        # Add the single hand landmarks to the right hand position
        for i in range(num_landmarks):
            new_row[f'right_landmark{i}_x'] = landmarks[i*3]
            new_row[f'right_landmark{i}_y'] = landmarks[i*3 + 1]
            new_row[f'right_landmark{i}_z'] = landmarks[i*3 + 2]
        
        converted_rows.append(new_row)
    
    # Create DataFrame from the converted rows
    converted_df = pd.DataFrame(converted_rows)
    
    # Combine the datasets
    print("Combining datasets...")
    combined_df = pd.concat([converted_df, two_df], ignore_index=True)
    
    # Save the combined dataset
    print(f"Saving combined dataset to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined dataset created with {len(combined_df)} samples")
    print(f"  - {len(converted_df)} samples from single-hand dataset")
    print(f"  - {len(two_df)} samples from two-hand dataset")
    
    return combined_df

if __name__ == "__main__":
    # Example usage
    analyze_dataset()
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print("Preprocessing complete!")