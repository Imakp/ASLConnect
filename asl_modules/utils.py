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
            if hand_landmarks is not None:  # Check if landmarks exist for this hand
                # Convert to numpy array for easier manipulation
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

def get_model_dir():
    """
    Get the directory for storing model files.
    Creates the directory if it doesn't exist.
    
    Returns:
    --------
    model_dir : str
        Absolute path to the model directory
    """
    model_dir = os.path.join(get_project_root(), 'asl_model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

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
        Base directory (if None, uses the model directory)
    """
    if base_dir is None:
        base_dir = get_model_dir()
    
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
        Base directory (if None, uses the model directory)
        
    Returns:
    --------
    model : object
        Loaded model
    """
    if base_dir is None:
        base_dir = get_model_dir()
    
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
        min_tracking_confidence=min_tracking_confidence,
        # Add model complexity parameter to improve accuracy
        model_complexity=1
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

def process_hand_landmarks(results, image_width=640, image_height=480):
    """
    Process MediaPipe hand detection results to extract and organize landmarks.
    
    Parameters:
    -----------
    results : mediapipe.python.solutions.hands.process() results
        Results from MediaPipe hand detection
    image_width : int
        Width of the input image (default: 640)
    image_height : int
        Height of the input image (default: 480)
        
    Returns:
    --------
    hand_data : dict
        Dictionary containing:
        - 'multi_hand_landmarks': List of landmarks for each detected hand
        - 'multi_handedness': List of handedness (left/right) for each detected hand
        - 'organized_landmarks': Dictionary with 'Left' and 'Right' keys containing landmarks
        - 'hand_count': Number of hands detected
        - 'orientation_confidence': Confidence in hand orientation detection
    """
    hand_data = {
        'multi_hand_landmarks': results.multi_hand_landmarks,
        'multi_handedness': results.multi_handedness,
        'organized_landmarks': {'Left': None, 'Right': None},
        'hand_count': 0,
        'orientation_confidence': {'Left': 0.0, 'Right': 0.0},
        'image_dimensions': (image_width, image_height)
    }
    
    if results.multi_hand_landmarks:
        hand_data['hand_count'] = len(results.multi_hand_landmarks)
        
        # Organize landmarks by handedness (left/right)
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness and idx < len(results.multi_handedness):
                # Get MediaPipe's handedness classification
                mp_handedness = results.multi_handedness[idx].classification[0].label
                mp_confidence = results.multi_handedness[idx].classification[0].score
                
                # Verify and potentially correct the handedness
                verified_handedness, verified_confidence = verify_hand_orientation(
                    hand_landmarks.landmark, mp_handedness
                )
                
                # Use the verified handedness
                handedness = verified_handedness
                hand_data['orientation_confidence'][handedness] = verified_confidence
                
                # Store the landmarks under the verified handedness
                hand_data['organized_landmarks'][handedness] = hand_landmarks.landmark
    
    return hand_data


def verify_hand_orientation(landmarks, handedness_label):
    """
    Verify and potentially correct the hand orientation (left/right) based on
    anatomical features rather than just relying on MediaPipe's classification.
    
    Parameters:
    -----------
    landmarks : list
        List of landmarks with x, y, z coordinates
    handedness_label : str
        The handedness label provided by MediaPipe ('Left' or 'Right')
        
    Returns:
    --------
    corrected_label : str
        The corrected handedness label ('Left' or 'Right')
    confidence : float
        Confidence score for the correction (0.0 to 1.0)
    """
    # Convert landmarks to numpy array for easier manipulation
    points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    
    # Extract key points for orientation analysis
    wrist = points[0]
    thumb_cmc = points[1]  # Thumb carpometacarpal joint
    thumb_tip = points[4]
    index_mcp = points[5]  # Index finger metacarpophalangeal joint
    pinky_mcp = points[17]  # Pinky metacarpophalangeal joint
    
    # Vector from wrist to middle of hand
    hand_center = (index_mcp + pinky_mcp) / 2
    wrist_to_center = hand_center - wrist
    
    # Vector from wrist to thumb
    wrist_to_thumb = thumb_tip - wrist
    
    # Cross product to determine which side the thumb is on
    # This works because the cross product gives a vector perpendicular to the plane
    # formed by the two input vectors, and its direction follows the right-hand rule
    cross_product = np.cross(wrist_to_center[:2], wrist_to_thumb[:2])
    
    # The sign of the z-component of the cross product tells us which side the thumb is on
    # For a right hand, the thumb should be on the left side when palm is facing down
    # For a left hand, the thumb should be on the right side when palm is facing down
    thumb_side = 'Right' if cross_product < 0 else 'Left'
    
    # Calculate the angle between the thumb and the hand center
    # This helps determine if the hand is facing up or down
    dot_product = np.dot(wrist_to_center, wrist_to_thumb)
    magnitudes = np.linalg.norm(wrist_to_center) * np.linalg.norm(wrist_to_thumb)
    angle = np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    
    # Determine if the palm is facing up or down
    # This is a simplified heuristic - in a real system, you might need more complex logic
    palm_facing = 'up' if thumb_tip[1] < wrist[1] else 'down'
    
    # Determine the likely handedness based on anatomical features
    anatomical_handedness = 'Right' if (palm_facing == 'down' and thumb_side == 'Left') or \
                                      (palm_facing == 'up' and thumb_side == 'Right') else 'Left'
    
    # Compare with MediaPipe's classification
    if anatomical_handedness == handedness_label:
        # If they agree, high confidence
        confidence = 0.9
    else:
        # If they disagree, use anatomical features but with lower confidence
        confidence = 0.7
    
    return anatomical_handedness, confidence