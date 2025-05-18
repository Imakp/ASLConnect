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
        console.log('Received init status:', data);
        isInitialized = data.complete;
        
        if (isInitialized) {
            statusElement.textContent = 'ASL Recognition Ready';
            statusElement.style.color = 'green';
            clearInterval(statusInterval);
        } else if (data.failed) {
            statusElement.textContent = 'ASL Recognition Failed: ' + (data.error || 'Unknown error');
            statusElement.style.color = 'red';
            clearInterval(statusInterval);
            console.error('ASL Recognition initialization failed:', data.error);
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
    // Handle special commands
    socket.on('subtitle', function(data) {
        if (typeof data.text === 'object' && data.text.command) {
            // Handle command
            const command = data.text.command;
            
            if (command === 'mute') {
                document.getElementById('muteBtn').click();
            } else if (command === 'end_call') {
                document.getElementById('leaveBtn').click();
            } else if (command === 'clear_history') {
                document.querySelector('.subtitles-history').innerHTML = '';
            }
            
            // Display command text
            displaySubtitle(data.text.text, data.confidence);
        } else {
            // Regular subtitle
            displaySubtitle(data.text, data.confidence);
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
    
    // Add a confidence threshold slider to the UI
    const confidenceSlider = document.createElement('input');
    confidenceSlider.type = 'range';
    confidenceSlider.min = '0.5';
    confidenceSlider.max = '0.95';
    confidenceSlider.step = '0.05';
    confidenceSlider.value = '0.7';
    confidenceSlider.id = 'confidenceThreshold';
    
    const sliderLabel = document.createElement('label');
    sliderLabel.htmlFor = 'confidenceThreshold';
    sliderLabel.textContent = 'Recognition Sensitivity: ';
    
    // Add to controls div
    document.querySelector('.controls').appendChild(sliderLabel);
    document.querySelector('.controls').appendChild(confidenceSlider);
    
    // Send threshold to server when changed
    confidenceSlider.addEventListener('change', function() {
        socket.emit('set_threshold', {
            threshold: parseFloat(this.value),
            room: currentRoom
        });
    });
});

// Create subtitle history container
const subtitlesHistory = document.createElement('div');
subtitlesHistory.className = 'subtitles-history';
document.querySelector('.video-wrapper:nth-child(2)').appendChild(subtitlesHistory);

// Update subtitle display function
function displaySubtitle(text, confidence) {
    const subtitles = document.getElementById('subtitles');
    subtitles.textContent = `${text} (${Math.round(confidence * 100)}%)`;
    subtitles.classList.add('active');
    
    // Add to history
    const historyEntry = document.createElement('p');
    historyEntry.textContent = text;
    subtitlesHistory.prepend(historyEntry);
    
    // Limit history size
    if (subtitlesHistory.children.length > 10) {
        subtitlesHistory.removeChild(subtitlesHistory.lastChild);
    }
    
    // Hide subtitle after 3 seconds
    setTimeout(() => {
        subtitles.classList.remove('active');
    }, 3000);
}