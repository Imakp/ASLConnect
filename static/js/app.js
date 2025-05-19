document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const webrtc = new WebRTCHandler(socket);
    let frameInterval = null;
    let callTimer = null;
    let callStartTime = null;
    
    // UI elements
    const joinBtn = document.getElementById('joinBtn');
    const leaveBtn = document.getElementById('leaveBtn');
    const muteBtn = document.getElementById('muteBtn');
    const roomInput = document.getElementById('roomId');
    const roomDisplay = document.getElementById('roomDisplay');
    const statusMessage = document.getElementById('statusMessage');
    const callTimerElement = document.getElementById('callTimer');
    const connectingOverlay = document.getElementById('connectingOverlay');
    const localMuted = document.getElementById('localMuted');
    const messagesContent = document.getElementById('messagesContent');
    
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
            // Update UI
            statusMessage.textContent = 'Connecting to room...';
            roomDisplay.textContent = roomId;
            connectingOverlay.style.display = 'flex';
            
            // Join room
            await webrtc.joinRoom(roomId);
            joinBtn.disabled = true;
            leaveBtn.disabled = false;
            startFrameCapture();
            startCallTimer();
        }
    });
    
    // Leave room
    leaveBtn.addEventListener('click', () => {
        webrtc.leaveRoom();
        joinBtn.disabled = false;
        leaveBtn.disabled = true;
        stopFrameCapture();
        stopCallTimer();
        
        // Update UI
        roomDisplay.textContent = 'Not Connected';
        statusMessage.textContent = 'Call ended';
        connectingOverlay.style.display = 'none';
    });
    
    // Toggle mute
    muteBtn.addEventListener('click', () => {
        const isMuted = !webrtc.toggleMute();
        muteBtn.innerHTML = isMuted ? 
            '<i class="fas fa-microphone-slash"></i><span>Unmute</span>' : 
            '<i class="fas fa-microphone"></i><span>Mute</span>';
        
        if (isMuted) {
            muteBtn.classList.add('active');
            localMuted.classList.add('active');
        } else {
            muteBtn.classList.remove('active');
            localMuted.classList.remove('active');
        }
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
                const subtitleTextElement = subtitleElement.querySelector('.subtitle-text') || subtitleElement;
                subtitleTextElement.textContent = `${text} (${(confidence * 100).toFixed(0)}%)`;
                
                // Make subtitle visible
                subtitleElement.classList.add('visible');
                
                // Hide subtitle after 3 seconds
                setTimeout(() => {
                    subtitleElement.classList.remove('visible');
                }, 3000);
                
                // Add message to the messages container
                // If the video element is remoteVideo, it's a received message
                // Otherwise, it's a sent message
                const messageType = videoElement.id === 'remoteVideo' ? 'received' : 'sent';
                addMessageToChat(text, confidence, messageType);
            }
        } else {
            // Fallback to the old method if no user_id is provided
            showSubtitle(text, confidence);
            
            // Determine if this is from local or remote user
            // Since we don't have user_id, we'll assume it's from remote
            addMessageToChat(text, confidence, 'received');
        }
    });
    
    // Function to add message to chat
    function addMessageToChat(text, confidence, type) {
        // Don't add empty messages or messages with very low confidence
        if (!text || text.trim() === '' || confidence < 0.2) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        
        messageElement.innerHTML = `
            <div class="message-content">${text}</div>
            <div class="message-confidence">${(confidence * 100).toFixed(0)}% confidence</div>
        `;
        
        messagesContent.appendChild(messageElement);
        messagesContent.scrollTop = messagesContent.scrollHeight;
        
        // Limit the number of messages to prevent performance issues
        while (messagesContent.children.length > 50) {
            messagesContent.removeChild(messagesContent.firstChild);
        }
    }
    
    // Add event to capture local sign language
    function sendFrameForASLRecognition(videoElement) {
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame to the canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert the canvas to a base64-encoded image
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send the frame to the server
        socket.emit('frame', {
            room: webrtc.roomId,
            frame: frameData,
            isLocal: true  // Flag to indicate this is from local user
        });
    }
    
    // Handle local ASL recognition result
    socket.on('asl-result', (data) => {
        const { text, confidence, isLocal } = data;
        
        // Show subtitle on the appropriate video
        if (isLocal) {
            // Show on local video
            const localVideo = document.getElementById('localVideo');
            if (localVideo) {
                let subtitleElement = localVideo.parentElement.querySelector('.subtitles');
                if (!subtitleElement) {
                    subtitleElement = document.createElement('div');
                    subtitleElement.className = 'subtitles';
                    localVideo.parentElement.appendChild(subtitleElement);
                }
                
                const subtitleTextElement = subtitleElement.querySelector('.subtitle-text') || subtitleElement;
                subtitleTextElement.textContent = `${text} (${(confidence * 100).toFixed(0)}%)`;
                
                subtitleElement.classList.add('visible');
                
                setTimeout(() => {
                    subtitleElement.classList.remove('visible');
                }, 3000);
                
                // Add to messages as sent
                addMessageToChat(text, confidence, 'sent');
            }
        } else {
            // Show on remote video (already handled by subtitle event)
            // But we can add to messages here as well
            addMessageToChat(text, confidence, 'received');
        }
    });
    
    // Handle peer connection events
    socket.on('user-connected', () => {
        connectingOverlay.style.display = 'none';
        statusMessage.textContent = 'Peer connected';
    });
    
    socket.on('user-disconnected', () => {
        statusMessage.textContent = 'Peer disconnected';
        connectingOverlay.style.display = 'flex';
        connectingOverlay.querySelector('p').textContent = 'Peer disconnected. Waiting for someone to join...';
    });
    
    // Add these new event listeners to fix the connection status issue
    socket.on('ice-candidate', () => {
        // When ICE candidates are exchanged, it means connection is being established
        // Hide the connecting overlay if remote video is playing
        checkRemoteVideoStatus();
    });
    
    socket.on('answer', () => {
        // When an answer is received, connection is being established
        // Hide the connecting overlay if remote video is playing
        checkRemoteVideoStatus();
    });
    
    // Add event listener for remote video
    const remoteVideo = document.getElementById('remoteVideo');
    if (remoteVideo) {
        remoteVideo.addEventListener('playing', () => {
            // When remote video starts playing, hide the connecting overlay
            connectingOverlay.style.display = 'none';
            statusMessage.textContent = 'Connected to peer';
        });
    }
    
    function checkRemoteVideoStatus() {
        const remoteVideo = document.getElementById('remoteVideo');
        if (remoteVideo && remoteVideo.readyState >= 3 && !remoteVideo.paused) {
            // If remote video is ready and playing, hide the connecting overlay
            connectingOverlay.style.display = 'none';
            statusMessage.textContent = 'Connected to peer';
        }
        
        // Set a timeout to check again in case the video takes time to start
        setTimeout(() => {
            if (remoteVideo && remoteVideo.readyState >= 3 && !remoteVideo.paused) {
                connectingOverlay.style.display = 'none';
                statusMessage.textContent = 'Connected to peer';
            }
        }, 2000);
    }
    
    function showSubtitle(text, confidence) {
        const subtitles = document.getElementById('subtitles');
        if (subtitles) {
            const subtitleTextElement = subtitles.querySelector('.subtitle-text') || subtitles;
            subtitleTextElement.textContent = `${text} (${(confidence * 100).toFixed(1)}%)`;
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
        }, 200); // Process 5 frames per second
    }
    
    function sendFrameForASLRecognition(videoElement) {
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame to the canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert the canvas to a base64-encoded image
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
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
    
    function startCallTimer() {
        callStartTime = new Date();
        callTimer = setInterval(updateCallTimer, 1000);
    }
    
    function updateCallTimer() {
        if (!callStartTime) return;
        
        const now = new Date();
        const diff = Math.floor((now - callStartTime) / 1000);
        const minutes = Math.floor(diff / 60).toString().padStart(2, '0');
        const seconds = (diff % 60).toString().padStart(2, '0');
        
        callTimerElement.textContent = `${minutes}:${seconds}`;
    }
    
    function stopCallTimer() {
        if (callTimer) {
            clearInterval(callTimer);
            callTimer = null;
        }
        callStartTime = null;
        callTimerElement.textContent = '00:00';
    }
});