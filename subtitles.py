import base64
import cv2
import numpy as np
from asl_modules.utils import initialize_mediapipe_hands, process_hand_landmarks, normalize_landmarks
from asl_modules.utils import load_model, get_project_root
import time
import os
import threading
import traceback

class ASLSubtitleGenerator:
    """
    Class to generate subtitles from ASL signs in video frames.
    """
    
    # Add these methods to your ASLSubtitleGenerator class
    
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
        print("Initializing ASL Subtitle Generator...")
        
        # Initialize MediaPipe Hands
        self.hands, self.mp_hands, _, _ = initialize_mediapipe_hands(max_num_hands=2)
        
        # For smoothing predictions
        self.recent_predictions = []
        self.max_recent = 5
        
        # For rate limiting
        self.last_process_time = 0
        self.process_interval = 0.2  # Process at most 5 frames per second
        
        # Load model and scaler in a separate thread to avoid blocking
        self.model = None
        self.scaler = None
        self.model_ready = False
        
        # Start loading models in a separate thread
        self._load_models_async(model_filename, scaler_filename)
    
    def _load_models_async(self, model_filename, scaler_filename):
        """Load models in a separate thread"""
        def load_models():
            try:
                print("Loading ML models...")
                # Get absolute paths to model files
                model_dir = os.path.join(get_project_root(), 'asl_model')
                model_path = os.path.join(model_dir, model_filename)
                scaler_path = os.path.join(model_dir, scaler_filename)
                
                print(f"Looking for model at: {model_path}")
                print(f"Looking for scaler at: {scaler_path}")
                
                if not os.path.exists(model_path):
                    print(f"ERROR: Model file not found: {model_path}")
                    return
                
                if not os.path.exists(scaler_path):
                    print(f"ERROR: Scaler file not found: {scaler_path}")
                    return
                
                print(f"Loading model from {model_path}")
                self.model = load_model(model_filename)
                
                print(f"Loading scaler from {scaler_path}")
                self.scaler = load_model(scaler_filename)
                
                self.model_ready = True
                print("ML models loaded successfully!")
            except Exception as e:
                print(f"Error loading models: {e}")
                print(traceback.format_exc())
        
        # Start loading in a separate thread
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        
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
        # Check if models are loaded
        if not self.model_ready:
            print("Models not yet loaded, skipping frame processing")
            return None, 0.0
            
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
                print("Failed to decode image")
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
            
            # Check if hands are detected
            if not results.multi_hand_landmarks:
                return None, 0.0
                
            # Process hand landmarks with image dimensions
            hand_data = process_hand_landmarks(results, image_width, image_height)
            
            # Debug print
            print(f"Detected {hand_data['hand_count']} hands")
            
            # Check if hands are detected
            if hand_data['hand_count'] > 0:
                # Predict ASL sign
                prediction, confidence = self.predict_asl_sign(hand_data['organized_landmarks'])
                
                # Debug print
                print(f"Raw prediction: {prediction}, confidence: {confidence:.2f}")
                
                # Add to recent predictions for smoothing
                if confidence >= 0.6:  # Lower threshold to catch more signs
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
            print(traceback.format_exc())
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
        # Check if models are loaded
        if not self.model_ready:
            return None, 0.0
            
        try:
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
            
            # After getting a prediction, check for special commands
            if prediction and confidence > 0.8:
                # Check for special command gestures
                if prediction == "MUTE":
                    return {"command": "mute", "text": "Mute/Unmute"}, confidence
                elif prediction == "END":
                    return {"command": "end_call", "text": "End Call"}, confidence
                elif prediction == "CLEAR":
                    return {"command": "clear_history", "text": "Clear History"}, confidence
                else:
                    # Regular ASL sign
                    return prediction, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            print(traceback.format_exc())
            return None, 0.0

    def add_to_sentence(self, prediction, confidence):
        """
        Add a prediction to the current sentence if it's stable.
        
        Parameters:
        -----------
        prediction : str
            Predicted ASL sign
        confidence : float
            Confidence score for the prediction
            
        Returns:
        --------
        sentence : str or None
            Current sentence if updated, None otherwise
        """
        current_time = time.time()
        
        # Reset sentence if timeout occurred
        if current_time - self.last_sentence_update > self.sentence_timeout and self.current_sentence:
            sentence = " ".join(self.current_sentence)
            self.current_sentence = []
            self.last_prediction = None
            self.prediction_count = 0
            return f"{sentence}."  # Return the completed sentence
        
        # Update last activity time
        self.last_sentence_update = current_time
        
        # Check if this is a repeat of the last prediction
        if prediction == self.last_prediction:
            self.prediction_count += 1
            
            # Add to sentence after seeing the same prediction multiple times
            if self.prediction_count == 3 and prediction not in self.current_sentence[-3:]:
                self.current_sentence.append(prediction)
                self.prediction_count = 0
                return " ".join(self.current_sentence)
        else:
            # New prediction
            self.last_prediction = prediction
            self.prediction_count = 1
        
        # Limit sentence length
        if len(self.current_sentence) > self.max_sentence_length:
            sentence = " ".join(self.current_sentence)
            self.current_sentence = []
            return f"{sentence}."  # Return the completed sentence
        
        return None