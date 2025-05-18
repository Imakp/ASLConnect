document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const webrtc = new WebRTCHandler(socket);
    let frameInterval = null;
    let isInitialized = false;
    
    // UI elements
    const joinBtn = document.getElementById('joinBtn');
    const leaveBtn = document.getElementById('leaveBtn');
    const muteBtn = document.getElementById('muteBtn');
    const roomInput = document.getElementById('roomId');
    const statusElement = document.createElement('div');
    statusElement.className = 'status-message';
    document.querySelector('.controls').appendChild(statusElement);
    
    // Initialize local video stream
    webrtc.initializeLocalStream().then(success => {
        if (!success) {
            alert('Failed to access camera and microphone');
        }
    });
    
    // Check initialization status
    function checkInitStatus() {
        socket.emit('check_init_status');
    }
    
    // Check status every 2 seconds
    const statusInterval = setInterval(checkInitStatus, 2000);
    
    // Handle initialization status updates
    socket.on('init_status', (data) => {
        isInitialized = data.complete;
        if (isInitialized) {
            statusElement.textContent = 'ASL Recognition Ready';
            statusElement.style.color = 'green';
            clearInterval(statusInterval);
        } else {
            statusElement.textContent = 'Loading ASL Recognition...';
            statusElement.style.color = 'orange';
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
        
        console.log(`Received subtitle: ${text} (${confidence}) for user ${user_id}`);
        
        // If user_id is provided, find the specific video element
        if (user_id) {
            // Find the video element for this user
            let videoElement = document.querySelector(`[data-user-id="${user_id}"]`);
            
            // If not found by data attribute, try to determine if it's local or remote
            if (!videoElement) {
                if (user_id === socket.id) {
                    videoElement = document.getElementById('localVideo');
                } else {
                    videoElement = document.getElementById('remoteVideo');
                }
            }
            
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
    
    // Start capturing frames for ASL recognition
    function startFrameCapture() {
        if (frameInterval) {
            clearInterval(frameInterval);
        }
        
        // Capture frames every 200ms (5fps)
        frameInterval = setInterval(() => {
            if (!isInitialized) {
                return; // Skip if not initialized
            }
            
            captureAndSendFrame();
        }, 200);
    }
    
    // Stop capturing frames
    function stopFrameCapture() {
        if (frameInterval) {
            clearInterval(frameInterval);
            frameInterval = null;
        }
    }
    
    // Capture and send a frame for ASL recognition
    function captureAndSendFrame() {
        const localVideo = document.getElementById('localVideo');
        if (!localVideo.srcObject) return;
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // Set canvas dimensions to match video
        canvas.width = localVideo.videoWidth;
        canvas.height = localVideo.videoHeight;
        
        // Draw the current video frame to the canvas
        context.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to base64 image
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        // Send the frame to the server
        socket.emit('frame', {
            frame: imageData,
            room: webrtc.roomId
        });
    }
});