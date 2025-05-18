import base64
import cv2
import numpy as np
from asl_modules.utils import initialize_mediapipe_hands, process_hand_landmarks, normalize_landmarks
from asl_modules.utils import load_model
import time

class ASLSubtitleGenerator:
    """
    Class to generate subtitles from ASL signs in video frames.
    """
    
    def __init__(self, model_filename='asl_mlp_multi_hand_model.joblib', scaler_filename='hand_landmarks_scaler.joblib'):
        """
        Initialize the ASL subtitle generator.
        
        Parameters:
        -----------
        model_filename : str
            Filename of the trained model
        scaler_filename : str
            Filename of the fitted scaler
        """
        # Load model and scaler
        self.model = load_model(model_filename)
        self.scaler = load_model(scaler_filename)
        
        # Initialize MediaPipe Hands
        self.hands, self.mp_hands, _, _ = initialize_mediapipe_hands(max_num_hands=2)
        
        # For smoothing predictions
        self.recent_predictions = []
        self.max_recent = 5
        
        # For rate limiting
        self.last_process_time = 0
        self.process_interval = 0.2  # Process at most 5 frames per second
        
    def process_frame(self, frame_data):
        """
        Process a video frame for ASL recognition.
        
        Parameters:
        -----------
        frame_data : str
            Base64-encoded image data
                
        Returns:
        --------
        prediction : str or None
            Predicted ASL sign, or None if no valid prediction
        confidence : float
            Confidence score for the prediction
        """
        # Rate limiting to prevent overloading
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return None, 0.0
        
        self.last_process_time = current_time
        
        try:
            # Decode base64 image
            encoded_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, 0.0
            
            # Resize image to reduce processing time
            scale_percent = 50  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            image = cv2.resize(image, (width, height))
                
            # Get image dimensions
            image_height, image_width = image.shape[:2]
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Process hand landmarks with image dimensions
            hand_data = process_hand_landmarks(results, image_width, image_height)
            
            # Check if hands are detected
            if hand_data['hand_count'] > 0:
                # Predict ASL sign
                prediction, confidence = self.predict_asl_sign(hand_data['organized_landmarks'])
                
                # Add to recent predictions for smoothing
                if confidence >= 0.7:  # Confidence threshold
                    self.recent_predictions.append(prediction)
                    if len(self.recent_predictions) > self.max_recent:
                        self.recent_predictions.pop(0)
                    
                    # Get most common prediction from recent ones
                    if self.recent_predictions:
                        from collections import Counter
                        counter = Counter(self.recent_predictions)
                        smoothed_prediction = counter.most_common(1)[0][0]
                        return smoothed_prediction, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, 0.0
        
    def predict_asl_sign(self, landmarks):
        """
        Predict the ASL sign from hand landmarks.
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary with 'Left' and 'Right' keys containing landmarks
            
        Returns:
        --------
        prediction : str
            Predicted ASL sign
        confidence : float
            Confidence score for the prediction
        """
        # Normalize the landmarks
        normalized_hands = normalize_landmarks(landmarks, multi_hand=True)
        
        # Prepare features vector with both hands
        feature_vector = np.zeros((1, 21 * 3 * 2))  # 21 landmarks, 3 coordinates, 2 hands
        
        # Add left hand features if available
        if normalized_hands.get('Left') is not None:
            left_features = normalized_hands['Left'].flatten()
            feature_vector[0, :left_features.size] = left_features
        
        # Add right hand features if available
        if normalized_hands.get('Right') is not None:
            right_features = normalized_hands['Right'].flatten()
            feature_vector[0, 21*3:21*3+right_features.size] = right_features
        
        # Add hand count as a feature
        hand_count = sum(1 for hand in normalized_hands.values() if hand is not None)
        
        # Create the final feature vector with hand count
        final_features = np.hstack([np.array([[hand_count]]), feature_vector])
        
        # Scale the features
        scaled_features = self.scaler.transform(final_features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(scaled_features)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence