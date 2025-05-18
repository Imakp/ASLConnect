document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const webrtc = new WebRTCHandler(socket);
    let frameInterval = null;
    
    // UI elements
    const joinBtn = document.getElementById('joinBtn');
    const leaveBtn = document.getElementById('leaveBtn');
    const muteBtn = document.getElementById('muteBtn');
    const roomInput = document.getElementById('roomId');
    
    // Initialize local video stream
    webrtc.initializeLocalStream().then(success => {
        if (!success) {
            alert('Failed to access camera and microphone');
        }
    });
    
    // Join room
    joinBtn.addEventListener('click', async () => {
        const roomId = roomInput.value.trim();
        if (roomId) {
            await webrtc.joinRoom(roomId);
            joinBtn.disabled = true;
            leaveBtn.disabled = false;
            startFrameCapture();
        }
    });
    
    // Leave room
    leaveBtn.addEventListener('click', () => {
        webrtc.leaveRoom();
        joinBtn.disabled = false;
        leaveBtn.disabled = true;
        stopFrameCapture();
    });
    
    // Toggle mute
    muteBtn.addEventListener('click', () => {
        const isMuted = !webrtc.toggleMute();
        muteBtn.textContent = isMuted ? 'Unmute' : 'Mute';
    });
    
    // Handle subtitles
    socket.on('subtitle', (data) => {
        const { text, confidence, user_id } = data;
        
        // If user_id is provided, find the specific video element
        if (user_id) {
            // Find the video element for this user
            const videoElement = document.querySelector(`[data-user-id="${user_id}"]`) || 
                                document.getElementById('remoteVideo');
            
            if (videoElement) {
                // Find or create subtitle element for this video
                let subtitleElement = videoElement.parentElement.querySelector('.subtitles');
                if (!subtitleElement) {
                    subtitleElement = document.createElement('div');
                    subtitleElement.className = 'subtitles';
                    videoElement.parentElement.appendChild(subtitleElement);
                }
                
                // Update subtitle text
                subtitleElement.textContent = `${text} (${(confidence * 100).toFixed(0)}%)`;
                
                // Make subtitle visible
                subtitleElement.classList.add('visible');
                
                // Hide subtitle after 3 seconds
                setTimeout(() => {
                    subtitleElement.classList.remove('visible');
                }, 3000);
            }
        } else {
            // Fallback to the old method if no user_id is provided
            showSubtitle(text, confidence);
        }
    });
    
    function showSubtitle(text, confidence) {
        const subtitles = document.getElementById('subtitles');
        if (subtitles) {
            subtitles.textContent = `${text} (${(confidence * 100).toFixed(1)}%)`;
            subtitles.classList.add('visible');
            
            // Hide subtitle after 3 seconds
            setTimeout(() => {
                subtitles.classList.remove('visible');
            }, 3000);
        }
    }
    
    function startFrameCapture() {
        const video = document.getElementById('localVideo');
        
        frameInterval = setInterval(() => {
            if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
                sendFrameForASLRecognition(video);
            }
        }, 500); // Process 2 frames per second instead of 5
    }
    
    function sendFrameForASLRecognition(videoElement) {
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        
        // Reduce resolution to save bandwidth and processing
        const scaleFactor = 0.5;
        canvas.width = videoElement.videoWidth * scaleFactor || 320;
        canvas.height = videoElement.videoHeight * scaleFactor || 240;
        
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame to the canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert the canvas to a base64-encoded image with reduced quality
        const frameData = canvas.toDataURL('image/jpeg', 0.6);
        
        // Send the frame to the server
        socket.emit('frame', {
            room: webrtc.roomId,
            frame: frameData
        });
    }
    
    function stopFrameCapture() {
        if (frameInterval) {
            clearInterval(frameInterval);
            frameInterval = null;
        }
    }
});