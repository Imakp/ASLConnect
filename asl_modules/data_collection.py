"""
Data collection module for ASL recognition project.
Handles collecting hand landmark data from webcam feed.
"""

import cv2
import numpy as np
import os
import csv
import time
from .utils import normalize_landmarks, initialize_mediapipe_hands, initialize_webcam, get_project_root, process_hand_landmarks

def save_to_csv(landmarks_list, labels, filename="asl_landmarks_dataset.csv"):
    """
    Save the normalized landmarks along with their labels to a CSV file
    
    Parameters:
    -----------
    landmarks_list : list
        List of normalized landmarks (can contain single hand or both hands data)
    labels : list
        List of labels corresponding to the landmarks
    filename : str
        Name of the CSV file to save the data
        
    Returns:
    --------
    saved_count : int
        Number of samples saved
    """
    filepath = os.path.join(get_project_root(), filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers if file is new
        if not file_exists:
            # Create header: label, hand_count, left_landmark0_x, left_landmark0_y, ...
            headers = ['label', 'hand_count']
            
            # Headers for left hand
            for i in range(21):  # 21 landmarks per hand
                headers.extend([f'left_landmark{i}_x', f'left_landmark{i}_y', f'left_landmark{i}_z'])
            
            # Headers for right hand
            for i in range(21):
                headers.extend([f'right_landmark{i}_x', f'right_landmark{i}_y', f'right_landmark{i}_z'])
                
            writer.writerow(headers)
        
        # Write data rows
        for landmarks, label in zip(landmarks_list, labels):
            # Check if this is multi-hand data (dictionary) or single-hand data (numpy array)
            if isinstance(landmarks, dict):
                # Multi-hand data
                hand_count = sum(1 for hand in landmarks.values() if hand is not None)
                
                # Initialize row with label and hand count
                row = [label, hand_count]
                
                # Add left hand data if available
                if landmarks.get('Left') is not None:
                    row.extend(landmarks['Left'].flatten().tolist())
                else:
                    # Add zeros for missing left hand
                    row.extend([0.0] * (21 * 3))
                
                # Add right hand data if available
                if landmarks.get('Right') is not None:
                    row.extend(landmarks['Right'].flatten().tolist())
                else:
                    # Add zeros for missing right hand
                    row.extend([0.0] * (21 * 3))
            else:
                # Single-hand data (backward compatibility)
                row = [label, 1]
                # Add the single hand data to the right hand columns
                row.extend([0.0] * (21 * 3))  # Zeros for left hand
                row.extend(landmarks.flatten().tolist())  # Single hand data as right hand
            
            writer.writerow(row)
    
    print(f"Data saved to {filepath}")
    return len(landmarks_list)

def collect_asl_data(asl_letter=None, num_samples=None, output_file="asl_landmarks_dataset.csv", multi_hand=False):
    """
    Collect ASL hand landmark data using webcam.
    
    Parameters:
    -----------
    asl_letter : str, optional
        ASL letter to collect data for (if None, will prompt user)
    num_samples : int, optional
        Number of samples to collect (if None, will prompt user)
    output_file : str
        Name of the CSV file to save the data
    multi_hand : bool
        Flag indicating if collecting data for multi-hand signs
    """
    # Initialize MediaPipe Hands with support for two hands
    hands, mp_hands, mp_drawing, mp_drawing_styles = initialize_mediapipe_hands(max_num_hands=2)
    
    # Initialize webcam
    cap = initialize_webcam()
    
    # Get user input for the ASL letter and number of samples if not provided
    if asl_letter is None:
        print("Enter the ASL letter/sign to collect data for (e.g., 'A', 'B', 'MEET', etc.):")
        asl_letter = input().strip().upper()
    
    if num_samples is None:
        print(f"Enter the number of samples to collect for sign '{asl_letter}':")
        try:
            num_samples = int(input().strip())
        except ValueError:
            print("Invalid input. Using default of 50 samples.")
            num_samples = 50
    
    # Ask if this is a two-handed sign if multi_hand flag is not set
    requires_two_hands = multi_hand
    if not multi_hand:
        print(f"Is '{asl_letter}' a two-handed sign? (y/n)")
        requires_two_hands = input().strip().lower() == 'y'
    
    # Lists to store collected data
    all_landmarks = []
    all_labels = []
    
    # Counter for collected samples
    collected_samples = 0
    
    # Delay between captures (in seconds)
    capture_delay = 0.5
    last_capture_time = 0
    
    # Auto-capture mode flag
    auto_capture = False
    
    print("\nInstructions:")
    print("- Position your hand(s) in the webcam view making the ASL sign for '" + asl_letter + "'")
    print("- Press 'c' to manually capture a sample")
    print("- Press 'a' to toggle auto-capture mode (captures every 0.5 seconds)")
    print("- Press 'q' to quit and save the collected data")
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            # Process the image
            image.flags.writeable = False
            results = hands.process(image)
            
            # Draw on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process hand landmarks
            hand_data = process_hand_landmarks(results)
            
            # Display progress and instructions
            cv2.putText(image, f"Collecting: {asl_letter} ({collected_samples}/{num_samples})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            auto_status = "ON" if auto_capture else "OFF"
            cv2.putText(image, f"Auto-capture: {auto_status}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check if required hands are detected
            hands_detected = hand_data['hand_count'] > 0
            correct_hands = True
            
            if requires_two_hands and hand_data['hand_count'] < 2:
                correct_hands = False
            
            # Draw landmarks if hands are detected
            if hand_data['multi_hand_landmarks']:
                for hand_landmarks in hand_data['multi_hand_landmarks']:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Display hand detection status and orientation confidence
            status_color = (0, 255, 0) if hands_detected and correct_hands else (0, 0, 255)
            
            if not hands_detected:
                status_text = "No Hands Detected"
            elif requires_two_hands and hand_data['hand_count'] < 2:
                status_text = f"Need Both Hands ({hand_data['hand_count']}/2)"
            else:
                status_text = f"Hands Detected: {hand_data['hand_count']}"
                
            cv2.putText(image, status_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display orientation confidence
            y_pos = 120
            for hand in ['Left', 'Right']:
                if hand_data['organized_landmarks'][hand] is not None:
                    confidence = hand_data['orientation_confidence'][hand]
                    confidence_text = f"{hand} hand: {confidence:.2f}"
                    cv2.putText(image, confidence_text, 
                               (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    y_pos += 30
            
            # Show the image
            cv2.imshow('ASL Data Collection', image)
            
            # Handle auto-capture
            current_time = time.time()
            can_capture = hands_detected and correct_hands
            
            if auto_capture and can_capture and (current_time - last_capture_time) >= capture_delay:
                if collected_samples < num_samples:
                    # Normalize landmarks based on whether we need one or two hands
                    if requires_two_hands:
                        # For two-handed signs, use the organized landmarks
                        normalized_landmarks = normalize_landmarks(
                            hand_data['organized_landmarks'], 
                            multi_hand=True
                        )
                    else:
                        # For single-handed signs, use the organized landmarks with proper handedness
                        if hand_data['multi_hand_landmarks'] and len(hand_data['multi_hand_landmarks']) > 0:
                            # Use the organized landmarks which should have the correct handedness
                            normalized_landmarks = normalize_landmarks(
                                hand_data['organized_landmarks'],
                                multi_hand=True
                            )
                        else:
                            continue  # Skip if no hand landmarks
                    
                    all_landmarks.append(normalized_landmarks)
                    all_labels.append(asl_letter)
                    
                    # Update counters
                    collected_samples += 1
                    last_capture_time = current_time
                    
                    print(f"Auto-captured sample {collected_samples}/{num_samples}")
                    
                    # Check if we've collected all samples
                    if collected_samples >= num_samples:
                        print(f"All {num_samples} samples collected for '{asl_letter}'!")
                        auto_capture = False
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c') and can_capture and collected_samples < num_samples:
                # Manual capture - normalize landmarks based on whether we need one or two hands
                if requires_two_hands:
                    # For two-handed signs, use the organized landmarks
                    normalized_landmarks = normalize_landmarks(
                        hand_data['organized_landmarks'], 
                        multi_hand=True
                    )
                else:
                    # For single-handed signs, use the organized landmarks with proper handedness
                    if hand_data['multi_hand_landmarks'] and len(hand_data['multi_hand_landmarks']) > 0:
                        # Use the organized landmarks which should have the correct handedness
                        normalized_landmarks = normalize_landmarks(
                            hand_data['organized_landmarks'],
                            multi_hand=True
                        )
                    else:
                        continue  # Skip if no hand landmarks
                
                all_landmarks.append(normalized_landmarks)
                all_labels.append(asl_letter)
                collected_samples += 1
                print(f"Manually captured sample {collected_samples}/{num_samples}")
                
                # Check if we've collected all samples
                if collected_samples >= num_samples:
                    print(f"All {num_samples} samples collected for '{asl_letter}'!")
            elif key == ord('a'):
                # Toggle auto-capture
                auto_capture = not auto_capture
                print(f"Auto-capture {'enabled' if auto_capture else 'disabled'}")
    
    finally:
        # Save collected data
        if all_landmarks:
            saved_count = save_to_csv(all_landmarks, all_labels, output_file)
            print(f"Saved {saved_count} samples for ASL sign '{asl_letter}'")
        else:
            print("No data was collected.")
        
        # Release resources
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        
    return collected_samples

def collect_multiple_asl_letters():
    """
    Collect data for multiple ASL letters in sequence.
    """
    print("ASL Data Collection Tool")
    print("========================")
    
    while True:
        # Collect data for one letter
        collect_asl_data()
        
        # Ask if user wants to collect more data
        print("\nDo you want to collect data for another letter? (y/n)")
        response = input().strip().lower()
        
        if response != 'y':
            break
    
    print("Data collection complete!")

if __name__ == "__main__":
    collect_multiple_asl_letters()