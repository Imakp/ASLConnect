from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import base64
import cv2
import numpy as np
from subtitles import ASLSubtitleGenerator
from video_call import VideoCallManager

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize managers
subtitle_generator = ASLSubtitleGenerator()
video_manager = VideoCallManager()

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

@socketio.on('frame')
def handle_frame(data):
    """Process video frame for ASL recognition"""
    frame_data = data['frame']
    room = data['room']
    user_id = request.sid
    
    # Process frame for ASL recognition
    prediction, confidence = subtitle_generator.process_frame(frame_data)
    
    # Use the same confidence threshold as in inference.py (0.7 instead of 0.85)
    if prediction and confidence > 0.7:
        # Emit subtitle to all users in room including sender (for debugging)
        emit('subtitle', {
            'text': prediction,
            'confidence': confidence,
            'user_id': user_id  # Include user ID so client knows which video to show subtitle on
        }, room=room)
        
        # Log successful predictions for debugging
        print(f"ASL Prediction: {prediction} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)