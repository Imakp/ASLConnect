"""
Inference module for ASL recognition project.
Handles real-time prediction and visualization.
"""

import cv2
import numpy as np
import time
from collections import Counter
from .utils import normalize_landmarks, load_model, initialize_mediapipe_hands, initialize_webcam, get_project_root

def predict_asl_sign(landmarks, model, scaler):
    """
    Predict the ASL sign from hand landmarks using the loaded model and scaler.
    
    Parameters:
    -----------
    landmarks : list or numpy.ndarray
        Hand landmarks from MediaPipe (21 landmarks with x, y, z coordinates)
    model : MLPClassifier
        Trained model for ASL sign classification
    scaler : StandardScaler
        Fitted scaler for feature normalization
        
    Returns:
    --------
    prediction : str
        Predicted ASL sign (letter)
    confidence : float
        Confidence score for the prediction (probability)
    """
    # Normalize the landmarks
    normalized_landmarks = normalize_landmarks(landmarks)
    
    # Flatten the landmarks to match the input format expected by the model
    flattened_landmarks = normalized_landmarks.flatten().reshape(1, -1)
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(flattened_landmarks)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    # Get prediction probability (confidence)
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence

def run_asl_recognition(model_filename='asl_mlp_model.joblib', scaler_filename='hand_landmarks_scaler.joblib'):
    """
    Run real-time ASL recognition using webcam feed.
    
    Parameters:
    -----------
    model_filename : str
        Filename of the trained model
    scaler_filename : str
        Filename of the fitted scaler
    """
    try:
        # Load the model and scaler
        model = load_model(model_filename)
        scaler = load_model(scaler_filename)
        print("Model and scaler loaded successfully!")
        
        # Initialize MediaPipe Hands
        hands, mp_hands, mp_drawing, mp_drawing_styles = initialize_mediapipe_hands()
        
        # Initialize webcam
        cap = initialize_webcam()
        
        # Get webcam dimensions for UI layout
        _, frame = cap.read()
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
        else:
            frame_height, frame_width = 480, 640  # Default values
        
        # Create a separate window for the prediction display
        cv2.namedWindow('ASL Recognition', cv2.WINDOW_NORMAL)
        
        print("Starting ASL Recognition. Press 'q' to quit.")
        
        # For FPS calculation
        prev_frame_time = 0
        new_frame_time = 0
        
        # For smoothing predictions (simple moving average)
        recent_predictions = []
        max_recent = 5  # Number of recent predictions to consider
        
        # For tracking prediction history
        prediction_history = []
        max_history = 10  # Maximum number of letters to show in history
        
        # For UI elements
        confidence_threshold = 0.7  # Minimum confidence to display prediction
        
        # For letter display
        letter_display_size = min(frame_width, frame_height) // 4
        letter_background_color = (50, 50, 50)
        letter_text_color = (255, 255, 255)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Flip the image horizontally and convert BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            # Process the image
            image.flags.writeable = False
            results = hands.process(image)
            
            # Draw on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Create a copy for the UI overlay
            display_image = image.copy()
            
            # Create UI elements
            # 1. Top bar with FPS and app title
            cv2.rectangle(display_image, (0, 0), (frame_width, 40), (50, 50, 50), -1)
            cv2.putText(display_image, "ASL Recognition App", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"FPS: {int(fps)}", (frame_width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 2. Bottom bar for instructions
            cv2.rectangle(display_image, (0, frame_height - 40), (frame_width, frame_height), (50, 50, 50), -1)
            cv2.putText(display_image, "Press 'q' to quit | 'c' to clear history", (10, frame_height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 3. Right sidebar for prediction history
            sidebar_width = 150
            cv2.rectangle(display_image, (frame_width - sidebar_width, 40), 
                         (frame_width, frame_height - 40), (50, 50, 50), -1)
            cv2.putText(display_image, "History:", (frame_width - sidebar_width + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display prediction history
            for i, pred in enumerate(prediction_history[-max_history:]):
                y_pos = 100 + i * 30
                cv2.putText(display_image, pred, (frame_width - sidebar_width + 20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Check if hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        display_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Predict ASL sign
                    prediction, confidence = predict_asl_sign(
                        hand_landmarks.landmark, model, scaler
                    )
                    
                    # Add to recent predictions for smoothing
                    if confidence >= confidence_threshold:
                        recent_predictions.append(prediction)
                        if len(recent_predictions) > max_recent:
                            recent_predictions.pop(0)
                    
                    # Get most common prediction from recent ones
                    if recent_predictions:
                        counter = Counter(recent_predictions)
                        smoothed_prediction = counter.most_common(1)[0][0]
                        
                        # Add to history if it's different from the last one
                        if not prediction_history or prediction_history[-1] != smoothed_prediction:
                            prediction_history.append(smoothed_prediction)
                    else:
                        smoothed_prediction = prediction
                    
                    # Display large letter in center-left of screen if confidence is high enough
                    if confidence >= confidence_threshold:
                        # Create a background rectangle for the letter
                        letter_x = 50
                        letter_y = frame_height // 2 - letter_display_size // 2
                        cv2.rectangle(display_image, 
                                     (letter_x, letter_y), 
                                     (letter_x + letter_display_size, letter_y + letter_display_size), 
                                     letter_background_color, -1)
                        
                        # Display the letter
                        cv2.putText(display_image, smoothed_prediction, 
                                   (letter_x + letter_display_size // 4, letter_y + letter_display_size * 3 // 4), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, letter_text_color, 8)
                        
                        # Display confidence below the letter
                        cv2.putText(display_image, f"Confidence: {confidence:.2f}", 
                                   (letter_x, letter_y + letter_display_size + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display hand position guide
                    cv2.putText(display_image, "Position hand in center of frame", 
                               (frame_width // 2 - 150, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                # Display "No hand detected" when no hand is found
                cv2.putText(display_image, "No hand detected", (frame_width // 2 - 100, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Clear recent predictions when no hand is detected
                recent_predictions = []
            
            # Show the image
            cv2.imshow('ASL Recognition', display_image)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear history
                prediction_history = []
        
        # Release resources
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("ASL recognition stopped.")
        
    except Exception as e:
        print(f"Error during ASL recognition: {e}")
        import traceback
        traceback.print_exc()

def predict_from_saved_landmarks(csv_file, model=None, scaler=None):
    """
    Make predictions on saved landmark data from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing hand landmarks
    model : MLPClassifier, optional
        Trained model (if None, will be loaded from disk)
    scaler : StandardScaler, optional
        Fitted scaler (if None, will be loaded from disk)
        
    Returns:
    --------
    y_pred : numpy.ndarray
        Predicted labels
    accuracy : float
        Accuracy of the predictions
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model = load_model('asl_mlp_model.joblib')
        scaler = load_model('hand_landmarks_scaler.joblib')
    
    # Load the CSV file
    filepath = os.path.join(get_project_root(), csv_file)
    df = pd.read_csv(filepath)
    
    # Extract features and true labels
    X = df.drop('label', axis=1).values
    y_true = df['label'].values
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nPredictions on {csv_file}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return y_pred, accuracy

if __name__ == "__main__":
    # Example usage
    run_asl_recognition()