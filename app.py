# Add these two lines at the very top of the file
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import base64
import cv2
import numpy as np
from subtitles import ASLSubtitleGenerator
from video_call import VideoCallManager
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize managers
# Load models in a separate thread to avoid blocking the main thread
subtitle_generator = None
video_manager = VideoCallManager()

# Create a thread pool for handling blocking operations
thread_pool = eventlet.greenpool.GreenPool(size=10)

# Flag to track if initialization is complete
init_complete = False
init_failed = False
init_error = None
init_timeout = 60  # Timeout in seconds

# Initialize the subtitle generator in a separate thread
def init_subtitle_generator():
    global subtitle_generator, init_complete, init_failed, init_error
    try:
        print("Starting ASL Subtitle Generator initialization...")
        subtitle_generator = ASLSubtitleGenerator()
        
        # Wait for model to load with timeout
        start_time = time.time()
        while not subtitle_generator.model_ready:
            time.sleep(0.5)
            # Check for timeout
            if time.time() - start_time > init_timeout:
                init_failed = True
                init_error = "Initialization timed out after {} seconds".format(init_timeout)
                print(init_error)
                return
                
        init_complete = True
        print("ASL Subtitle Generator initialization complete!")
    except Exception as e:
        init_failed = True
        init_error = str(e)
        print(f"ASL Subtitle Generator initialization failed: {e}")

# Start initialization in a background thread
init_thread = threading.Thread(target=init_subtitle_generator)
init_thread.daemon = True
init_thread.start()

@app.route('/')
def index():
    """Serve the main video call page"""
    return render_template('index.html')

@socketio.on('join')
def on_join(data):
    """Handle user joining a room"""
    room = data['room']
    join_room(room)
    video_manager.add_user_to_room(room, request.sid)
    emit('user_joined', {'room': room}, room=room)
    
    # Inform client about initialization status
    emit('init_status', {
        'complete': init_complete,
        'failed': init_failed,
        'error': init_error
    })

@socketio.on('leave')
def on_leave(data):
    """Handle user leaving a room"""
    room = data['room']
    leave_room(room)
    video_manager.remove_user_from_room(room, request.sid)
    emit('user_left', {'room': room}, room=room)

@socketio.on('offer')
def handle_offer(data):
    """Handle WebRTC offer"""
    room = data['room']
    emit('offer', data, room=room, skip_sid=request.sid)

@socketio.on('answer')
def handle_answer(data):
    """Handle WebRTC answer"""
    room = data['room']
    emit('answer', data, room=room, skip_sid=request.sid)

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    """Handle ICE candidate"""
    room = data['room']
    emit('ice_candidate', data, room=room, skip_sid=request.sid)

def process_frame_async(frame_data, room, user_id):
    """Process frame in a separate thread to avoid blocking the main loop"""
    global subtitle_generator
    
    # Check if subtitle generator is initialized
    if subtitle_generator is None or not subtitle_generator.model_ready:
        # If not ready yet, return without processing
        return
    
    prediction, confidence = subtitle_generator.process_frame(frame_data)
    
    if prediction and confidence > 0.6:  # Lower threshold to catch more signs
        # Emit subtitle to all users in room including sender
        socketio.emit('subtitle', {
            'text': prediction,
            'confidence': confidence,
            'user_id': user_id
        }, room=room)
        
        # Log successful predictions for debugging
        print(f"ASL Prediction: {prediction} (Confidence: {confidence:.2f})")

@socketio.on('frame')
def handle_frame(data):
    """Process video frame for ASL recognition"""
    frame_data = data['frame']
    room = data['room']
    user_id = request.sid
    
    # Offload the processing to a green thread to avoid blocking
    thread_pool.spawn_n(process_frame_async, frame_data, room, user_id)

@socketio.on('check_init_status')
def check_init_status():
    """Check if initialization is complete"""
    emit('init_status', {
        'complete': init_complete,
        'failed': init_failed,
        'error': init_error
    })

if __name__ == '__main__':
    # Use PORT environment variable provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)