"""
Utility functions for ASL recognition project.
Contains common functions used across different modules.
"""

import numpy as np
import os
import joblib
import cv2
import mediapipe as mp

def normalize_landmarks(landmarks, multi_hand=False):
    """
    Normalize the landmarks by:
    1. Centering to the wrist (landmark 0)
    2. Scaling so the maximum distance from the wrist becomes 1
    
    Parameters:
    -----------
    landmarks : list or dict
        List of landmarks with x, y, z coordinates for single hand
        or dictionary with 'Left' and 'Right' keys for two hands
    multi_hand : bool
        Flag indicating if the landmarks are for multiple hands
        
    Returns:
    --------
    normalized_points : numpy.ndarray or dict
        Normalized landmarks as a numpy array for single hand
        or dictionary with 'Left' and 'Right' keys for two hands
    """
    if not multi_hand:
        # Original single-hand normalization
        # Convert to numpy array for easier manipulation
        points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
        
        # Center to the wrist (landmark 0)
        wrist = points[0]
        centered_points = points - wrist
        
        # Find the maximum distance from the wrist
        distances = np.linalg.norm(centered_points, axis=1)
        max_distance = np.max(distances)
        
        # Scale so the maximum distance becomes 1
        if max_distance > 0:  # Avoid division by zero
            normalized_points = centered_points / max_distance
        else:
            normalized_points = centered_points
            
        return normalized_points
    else:
        # Multi-hand normalization
        normalized_hands = {}
        
        # Process each hand separately
        for hand_label, hand_landmarks in landmarks.items():
            if hand_landmarks:
                points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])
                
                # Center to the wrist (landmark 0)
                wrist = points[0]
                centered_points = points - wrist
                
                # Find the maximum distance from the wrist
                distances = np.linalg.norm(centered_points, axis=1)
                max_distance = np.max(distances)
                
                # Scale so the maximum distance becomes 1
                if max_distance > 0:  # Avoid division by zero
                    normalized_points = centered_points / max_distance
                else:
                    normalized_points = centered_points
                    
                normalized_hands[hand_label] = normalized_points
            else:
                normalized_hands[hand_label] = None
                
        return normalized_hands

def save_model(model, filename, base_dir=None):
    """
    Save a model to disk using joblib.
    
    Parameters:
    -----------
    model : object
        Model to save
    filename : str
        Filename to save the model
    base_dir : str, optional
        Base directory (if None, uses the current file's directory)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    filepath = os.path.join(base_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename, base_dir=None):
    """
    Load a model from disk using joblib.
    
    Parameters:
    -----------
    filename : str
        Filename of the model to load
    base_dir : str, optional
        Base directory (if None, uses the current file's directory)
        
    Returns:
    --------
    model : object
        Loaded model
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading model from {filepath}")
    return joblib.load(filepath)

def initialize_mediapipe_hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
    """
    Initialize MediaPipe Hands for hand landmark detection.
    
    Parameters:
    -----------
    max_num_hands : int
        Maximum number of hands to detect (default: 2 for two-hand signs)
    min_detection_confidence : float
        Minimum confidence for hand detection
    min_tracking_confidence : float
        Minimum confidence for hand tracking
        
    Returns:
    --------
    hands : mediapipe.python.solutions.hands.Hands
        MediaPipe Hands object
    mp_hands : module
        MediaPipe hands module
    mp_drawing : module
        MediaPipe drawing module
    mp_drawing_styles : module
        MediaPipe drawing styles module
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    return hands, mp_hands, mp_drawing, mp_drawing_styles

def initialize_webcam(camera_id=0):
    """
    Initialize OpenCV VideoCapture for webcam access.
    
    Parameters:
    -----------
    camera_id : int
        Camera ID (default: 0 for default webcam)
        
    Returns:
    --------
    cap : cv2.VideoCapture
        OpenCV VideoCapture object
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    
    return cap

def get_project_root():
    """
    Get the root directory of the project.
    
    Returns:
    --------
    root_dir : str
        Absolute path to the project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_hand_landmarks(results):
    """
    Process MediaPipe hand detection results to extract and organize landmarks.
    
    Parameters:
    -----------
    results : mediapipe.python.solutions.hands.Hands.process() results
        Results from MediaPipe hand detection
        
    Returns:
    --------
    hand_data : dict
        Dictionary containing:
        - 'multi_hand_landmarks': List of landmarks for each detected hand
        - 'multi_handedness': List of handedness (left/right) for each detected hand
        - 'organized_landmarks': Dictionary with 'Left' and 'Right' keys containing landmarks
        - 'hand_count': Number of hands detected
    """
    hand_data = {
        'multi_hand_landmarks': results.multi_hand_landmarks,
        'multi_handedness': results.multi_handedness,
        'organized_landmarks': {'Left': None, 'Right': None},
        'hand_count': 0
    }
    
    if results.multi_hand_landmarks:
        hand_data['hand_count'] = len(results.multi_hand_landmarks)
        
        # Organize landmarks by handedness (left/right)
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
                hand_data['organized_landmarks'][handedness] = hand_landmarks.landmark
    
    return hand_data