"""
Data collection module for ASL recognition project.
Handles collecting hand landmark data from webcam feed.
"""

import cv2
import numpy as np
import os
import csv
import time
from .utils import normalize_landmarks, initialize_mediapipe_hands, initialize_webcam, get_project_root

def save_to_csv(landmarks_list, labels, filename="asl_landmarks_dataset.csv"):
    """
    Save the normalized landmarks along with their labels to a CSV file
    
    Parameters:
    -----------
    landmarks_list : list
        List of normalized landmarks
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
            # Create header: label, landmark0_x, landmark0_y, landmark0_z, landmark1_x, ...
            headers = ['label']
            for i in range(21):  # 21 landmarks
                headers.extend([f'landmark{i}_x', f'landmark{i}_y', f'landmark{i}_z'])
            writer.writerow(headers)
        
        # Write data rows
        for landmarks, label in zip(landmarks_list, labels):
            # Flatten the landmarks array and convert to list
            flattened = landmarks.flatten().tolist()
            # Combine label with flattened landmarks
            row = [label] + flattened
            writer.writerow(row)
    
    print(f"Data saved to {filepath}")
    return len(landmarks_list)

def collect_asl_data(asl_letter=None, num_samples=None, output_file="asl_landmarks_dataset.csv"):
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
    """
    # Initialize MediaPipe Hands
    hands, mp_hands, mp_drawing, mp_drawing_styles = initialize_mediapipe_hands(max_num_hands=1)
    
    # Initialize webcam
    cap = initialize_webcam()
    
    # Get user input for the ASL letter and number of samples if not provided
    if asl_letter is None:
        print("Enter the ASL letter to collect data for (e.g., 'A', 'B', etc.):")
        asl_letter = input().strip().upper()
    
    if num_samples is None:
        print(f"Enter the number of samples to collect for letter '{asl_letter}':")
        try:
            num_samples = int(input().strip())
        except ValueError:
            print("Invalid input. Using default of 50 samples.")
            num_samples = 50
    
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
    print("- Position your hand in the webcam view making the ASL sign for letter '" + asl_letter + "'")
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
            
            # Display progress and instructions
            cv2.putText(image, f"Collecting: {asl_letter} ({collected_samples}/{num_samples})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            auto_status = "ON" if auto_capture else "OFF"
            cv2.putText(image, f"Auto-capture: {auto_status}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check if hand is detected
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Display hand detection status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Hand Detected" if hand_detected else "No Hand Detected"
            cv2.putText(image, status_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show the image
            cv2.imshow('ASL Data Collection', image)
            
            # Handle auto-capture
            current_time = time.time()
            if auto_capture and hand_detected and (current_time - last_capture_time) >= capture_delay:
                if collected_samples < num_samples:
                    # Normalize and save landmarks
                    normalized_landmarks = normalize_landmarks(results.multi_hand_landmarks[0].landmark)
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
            elif key == ord('c') and hand_detected and collected_samples < num_samples:
                # Manual capture
                normalized_landmarks = normalize_landmarks(results.multi_hand_landmarks[0].landmark)
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
            print(f"Saved {saved_count} samples for ASL letter '{asl_letter}'")
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