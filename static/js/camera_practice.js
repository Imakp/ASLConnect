document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const practiceItems = document.querySelectorAll('.practice-item');
    const practiceTabs = document.querySelectorAll('.practice-tab');
    const tabContents = document.querySelectorAll('.practice-tab-content');
    const startCameraBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureImage');
    const cameraFeed = document.getElementById('cameraFeed');
    const referenceSign = document.getElementById('referenceSign');
    const currentLetterEl = document.getElementById('currentLetter');
    const feedbackMessage = document.querySelector('.feedback-message');
    const accuracyFill = document.querySelector('.accuracy-fill');
    const accuracyValue = document.querySelector('.accuracy-value');
    
    let stream = null;
    let currentLetter = 'A';
    let socket = null;
    let frameInterval = null;
    let recognitionActive = false;
    
    // Initialize
    updateReferenceImage('A');
    initializeSocket();
    
    // Fix letter data attributes in HTML
    const letterItems = document.querySelectorAll('#letters-content .practice-item');
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    
    letterItems.forEach((item, index) => {
        if (index < letters.length) {
            item.setAttribute('data-letter', letters[index]);
            item.querySelector('span').textContent = letters[index];
            item.querySelector('img').alt = `ASL ${letters[index]}`;
        }
    });
    
    // Initialize Socket.IO connection
    function initializeSocket() {
        // Check if socket.io is loaded
        if (typeof io === 'undefined') {
            console.error('Socket.IO is not loaded. Make sure to include the socket.io client script.');
            feedbackMessage.textContent = 'Error: Socket.IO not loaded. Please refresh the page.';
            return;
        }
        
        // Initialize socket connection
        socket = io();
        console.log('Socket connection initialized');
        
        // Listen for ASL recognition results from the server
        socket.on('asl-result', (data) => {
            if (!recognitionActive) return;
            
            const { text, confidence, isLocal } = data;
            
            // Only process if this is a local prediction (from this user)
            if (isLocal) {
                console.log(`Received prediction: ${text} with confidence ${confidence}`);
                
                // Check if the prediction matches the current letter
                if (text === currentLetter) {
                    // Only update accuracy meter if the prediction matches the current letter
                    updateAccuracy(Math.round(confidence * 100));
                    
                    feedbackMessage.textContent = `Great! Your sign for "${currentLetter}" was recognized with ${Math.round(confidence * 100)}% confidence.`;
                    
                    // Add success class if confidence is high enough
                    if (confidence > 0.8) {
                        feedbackMessage.className = 'feedback-message feedback-success';
                    } else {
                        feedbackMessage.className = 'feedback-message feedback-warning';
                    }
                } else if (text) {
                    // If prediction doesn't match, set accuracy to 0
                    updateAccuracy(0);
                    
                    feedbackMessage.textContent = `Your sign was recognized as "${text}" instead of "${currentLetter}". Try adjusting your hand position.`;
                    feedbackMessage.className = 'feedback-message feedback-error';
                } else {
                    // No sign detected
                    updateAccuracy(0);
                    
                    feedbackMessage.textContent = `No sign detected. Make sure your hand is visible in the frame.`;
                    feedbackMessage.className = 'feedback-message';
                }
            }
        });
    }
    
    // Tab switching
    practiceTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            practiceTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show corresponding content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(`${tabId}-content`).classList.add('active');
            
            // Select first item in the tab
            const firstItem = document.querySelector(`#${tabId}-content .practice-item`);
            if (firstItem) {
                selectItem(firstItem);
            }
        });
    });
    
    // Item selection
    practiceItems.forEach(item => {
        item.addEventListener('click', function() {
            selectItem(this);
        });
    });
    
    function selectItem(item) {
        // Remove active class from all items
        practiceItems.forEach(i => i.classList.remove('active'));
        
        // Add active class to clicked item
        item.classList.add('active');
        
        // Update reference image
        const letter = item.getAttribute('data-letter');
        updateReferenceImage(letter);
    }
    
    function updateReferenceImage(letter) {
        currentLetter = letter;
        currentLetterEl.textContent = letter;
        
        // Update reference image based on letter
        const isNumber = !isNaN(parseInt(letter));
        const folder = isNumber ? 'numbers' : 'alphabets';
        const filename = isNumber ? `Sign ${letter}.jpeg` : `${letter}_test.jpg`;
        
        referenceSign.src = `/static/images/public/${folder}/${filename}`;
        
        // Update feedback message
        feedbackMessage.textContent = `Practice signing the ${isNumber ? 'number' : 'letter'} "${letter}". Position your hand in the center of the frame.`;
        feedbackMessage.className = 'feedback-message';
        
        // Reset accuracy meter
        updateAccuracy(0);
    }
    
    // Camera functionality
    startCameraBtn.addEventListener('click', async function() {
        try {
            if (stream) {
                // Stop camera if already running
                stopCamera();
                return;
            }
            
            // Start camera
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } 
            });
            
            cameraFeed.srcObject = stream;
            startCameraBtn.innerHTML = '<i class="fas fa-video-slash"></i> Stop Camera';
            captureBtn.disabled = false;
            
            feedbackMessage.textContent = `Camera started. Position your hand to sign "${currentLetter}".`;
            
            // Start continuous frame capture
            recognitionActive = true;
            startFrameCapture();
            
        } catch (err) {
            console.error('Error accessing camera:', err);
            feedbackMessage.textContent = `Error accessing camera: ${err.message}`;
            feedbackMessage.className = 'feedback-message feedback-error';
        }
    });
    
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            cameraFeed.srcObject = null;
            startCameraBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
            captureBtn.disabled = true;
            feedbackMessage.textContent = `Select a letter or number and start the camera to begin practice.`;
            feedbackMessage.className = 'feedback-message';
            
            // Stop frame capture
            recognitionActive = false;
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
        }
    }
    
    // Start continuous frame capture
    function startFrameCapture() {
        // Clear any existing interval
        if (frameInterval) {
            clearInterval(frameInterval);
        }
        
        console.log('Starting continuous frame capture');
        
        // Capture frames at regular intervals
        frameInterval = setInterval(() => {
            if (!recognitionActive) return;
            
            // Capture and send frame for processing
            captureAndSendFrame();
        }, 500); // 2 frames per second to avoid overloading
    }
    
    // Capture and send frame for ASL recognition
    function captureAndSendFrame() {
        if (!cameraFeed || !cameraFeed.srcObject || !socket) return;
        
        try {
            // Create a canvas to capture the frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            canvas.width = cameraFeed.videoWidth;
            canvas.height = cameraFeed.videoHeight;
            
            // Draw the current video frame to the canvas
            ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
            
            // Convert the canvas to a base64-encoded image
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send the frame to the server with the target letter
            socket.emit('frame', {
                room: 'practice-room-' + socket.id, // Create a unique room for practice
                frame: frameData,
                isLocal: true,
                targetLetter: currentLetter
            });
            
        } catch (error) {
            console.error('Error capturing or sending frame:', error);
        }
    }
    
    // Update accuracy meter
    function updateAccuracy(value) {
        accuracyFill.style.width = `${value}%`;
        accuracyValue.textContent = `${value}%`;
        
        // Change color based on accuracy
        if (value > 80) {
            accuracyFill.style.backgroundColor = 'var(--success)';
            accuracyValue.style.color = 'var(--success)';
        } else if (value > 50) {
            accuracyFill.style.backgroundColor = 'var(--warning)';
            accuracyValue.style.color = 'var(--warning)';
        } else {
            accuracyFill.style.backgroundColor = 'var(--danger)';
            accuracyValue.style.color = 'var(--danger)';
        }
    }
    
    // Remove the capture button click event since we're doing continuous capture
    captureBtn.addEventListener('click', function() {
        // We'll keep this for manual capture if needed
        captureAndSendFrame();
    });
});