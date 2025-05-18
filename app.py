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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize managers
# Load models in a separate thread to avoid blocking the main thread
subtitle_generator = None
video_manager = VideoCallManager()

# Create a thread pool for handling blocking operations
thread_pool = eventlet.greenpool.GreenPool(size=10)

# Initialize the subtitle generator in a separate thread
def init_subtitle_generator():
    global subtitle_generator
    subtitle_generator = ASLSubtitleGenerator()

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
    if subtitle_generator is None:
        # If not ready yet, return without processing
        return
    
    prediction, confidence = subtitle_generator.process_frame(frame_data)
    
    if prediction and confidence > 0.7:
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

if __name__ == '__main__':
    # Use PORT environment variable provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)