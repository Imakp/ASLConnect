document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startQuizBtn = document.getElementById('startQuizBtn');
    const nextQuizBtn = document.getElementById('nextQuizBtn');
    const quizCamera = document.getElementById('quizCamera');
    const cameraOverlay = document.getElementById('cameraOverlay');
    const quizLetter = document.getElementById('quizLetter');
    const timeLeft = document.getElementById('timeLeft');
    const currentScore = document.getElementById('currentScore');
    const feedbackMessage = document.getElementById('feedbackMessage');
    const predictionFeedback = document.getElementById('predictionFeedback');
    
    // Quiz state
    let stream = null;
    let quizActive = false;
    let timerInterval = null;
    let secondsRemaining = 120;
    let currentLetterIndex = -1;
    let score = 10;
    let recognitionActive = false;
    let currentPrediction = '';
    let currentConfidence = 0;
    let confidenceThreshold = 0.90; // Increased threshold to 90% for stricter validation
    let socket = null;
    let frameInterval = null;
    let attemptCount = 0;
    let maxAttempts = 3;
    
    // ASL letters to quiz
    const aslLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
    
    // Initialize Socket.IO connection for real model predictions
    function initializeSocket() {
        // Check if socket.io is loaded
        if (typeof io === 'undefined') {
            console.error('Socket.IO is not loaded. Make sure to include the socket.io client script.');
            feedbackMessage.textContent = 'Error: Socket.IO not loaded. Please refresh the page.';
            feedbackMessage.className = 'feedback-message feedback-error';
            return;
        }
        
        // Initialize socket connection
        socket = io();
        console.log('Socket connection initialized');
        
        // Listen for ASL recognition results from the server
        socket.on('asl-result', (data) => {
            console.log('Received ASL result:', data);
            if (!quizActive) return;
            
            const { text, confidence, isLocal } = data;
            
            // Only process if this is a local prediction (from this user)
            if (isLocal) {
                // Update current prediction and confidence
                currentPrediction = text;
                currentConfidence = confidence;
                
                // Get current letter to recognize
                const targetLetter = aslLetters[currentLetterIndex];
                
                // Update prediction feedback
                updatePredictionFeedback(text, confidence, targetLetter);
                
                // Check if the prediction matches the target letter with sufficient confidence
                if (text === targetLetter && confidence >= confidenceThreshold) {
                    // Correct prediction with high confidence
                    handleCorrectPrediction();
                } else if (text === targetLetter) {
                    // Correct letter but confidence too low
                    feedbackMessage.textContent = `Almost there! Keep signing "${targetLetter}" clearly. (${Math.round(confidence * 100)}% confident)`;
                    feedbackMessage.className = 'feedback-message feedback-warning';
                    attemptCount++;
                    checkAttempts();
                } else if (text) {
                    // Incorrect prediction
                    feedbackMessage.textContent = `That looks like "${text}". Try signing "${targetLetter}" instead.`;
                    feedbackMessage.className = 'feedback-message feedback-error';
                    attemptCount++;
                    checkAttempts();
                } else {
                    // No valid prediction
                    feedbackMessage.textContent = 'No valid sign detected. Please try again.';
                    feedbackMessage.className = 'feedback-message feedback-warning';
                }
            }
        });
        
        // Handle connection events
        socket.on('connect', () => {
            console.log('Connected to server with ID:', socket.id);
        });
        
        socket.on('connect_error', (error) => {
            console.error('Socket connection error:', error);
            feedbackMessage.textContent = 'Connection error. Please refresh the page.';
            feedbackMessage.className = 'feedback-message feedback-error';
        });
    }
    
    // Check if max attempts reached
    function checkAttempts() {
        if (attemptCount >= maxAttempts) {
            feedbackMessage.textContent = `Maximum attempts reached. The correct sign was "${aslLetters[currentLetterIndex]}".`;
            feedbackMessage.className = 'feedback-message feedback-error';
            nextQuizBtn.style.display = 'block';
            recognitionActive = false;
        }
    }
    
    // Start the quiz
    async function startQuiz() {
        // Reset quiz state
        quizActive = true;
        secondsRemaining = 120;
        score = 10;
        currentScore.textContent = score;
        currentPrediction = '';
        currentConfidence = 0;
        attemptCount = 0;
        
        // Update UI
        startQuizBtn.style.display = 'none';
        nextQuizBtn.style.display = 'none';
        feedbackMessage.textContent = 'Quiz started! Sign the letter shown.';
        feedbackMessage.className = 'feedback-message';
        
        // Start camera
        const cameraStarted = await startCamera();
        if (!cameraStarted) {
            return; // Exit if camera failed to start
        }
        
        // Select random letter
        selectRandomLetter();
        
        // Start timer
        startTimer();
        
        // Initialize Socket if not already done
        if (!socket) {
            initializeSocket();
        }
        
        // Start frame capture for ASL recognition
        recognitionActive = true;
        startFrameCapture();
    }
    
    // Start the camera
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            quizCamera.srcObject = stream;
            cameraOverlay.style.display = 'none';
            
            return true;
        } catch (error) {
            console.error("Error accessing camera:", error);
            feedbackMessage.textContent = 'Could not access camera. Please check permissions.';
            feedbackMessage.className = 'feedback-message feedback-error';
            return false;
        }
    }
    
    // Stop the camera
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            quizCamera.srcObject = null;
            cameraOverlay.style.display = 'flex';
        }
    }
    
    // Select a random letter for the quiz
    function selectRandomLetter() {
        // Get a random letter that's different from the current one
        let newIndex;
        do {
            newIndex = Math.floor(Math.random() * aslLetters.length);
        } while (newIndex === currentLetterIndex && aslLetters.length > 1);
        
        currentLetterIndex = newIndex;
        const letter = aslLetters[currentLetterIndex];
        
        // Update UI
        quizLetter.textContent = letter;
        
        // Reset attempt counter
        attemptCount = 0;
    }
    
    // Start the timer
    function startTimer() {
        updateTimerDisplay();
        
        timerInterval = setInterval(() => {
            secondsRemaining--;
            
            // Update score based on time
            if (secondsRemaining % 12 === 0 && secondsRemaining > 0) {
                score = Math.max(1, Math.floor(secondsRemaining / 12) + 1);
                currentScore.textContent = score;
            }
            
            updateTimerDisplay();
            
            // Check if time is up
            if (secondsRemaining <= 0) {
                endQuiz(false);
            }
        }, 1000);
    }
    
    // Update the timer display
    function updateTimerDisplay() {
        const minutes = Math.floor(secondsRemaining / 60);
        const seconds = secondsRemaining % 60;
        timeLeft.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Visual indication when time is running low
        if (secondsRemaining <= 30) {
            timeLeft.style.color = 'var(--danger)';
        } else if (secondsRemaining <= 60) {
            timeLeft.style.color = 'var(--warning)';
        } else {
            timeLeft.style.color = '';
        }
    }
    
    // Start frame capture for ASL recognition
    function startFrameCapture() {
        // Clear any existing interval
        if (frameInterval) {
            clearInterval(frameInterval);
        }
        
        console.log('Starting frame capture');
        
        // Capture frames at regular intervals
        frameInterval = setInterval(() => {
            if (!quizActive || !recognitionActive) return;
            
            // Capture and send frame for processing
            captureAndSendFrame();
        }, 200); // 5 frames per second
    }
    
    // Capture and send frame for ASL recognition
    function captureAndSendFrame() {
        if (!quizCamera || !quizCamera.srcObject || !socket) return;
        
        try {
            // Create a canvas to capture the frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            canvas.width = quizCamera.videoWidth;
            canvas.height = quizCamera.videoHeight;
            
            // Draw the current video frame to the canvas
            ctx.drawImage(quizCamera, 0, 0, canvas.width, canvas.height);
            
            // Convert the canvas to a base64-encoded image
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Get the current target letter
            const targetLetter = aslLetters[currentLetterIndex];
            
            // Send the frame to the server with the target letter
            socket.emit('frame', {
                room: 'quiz-room-' + socket.id, // Create a unique room for quiz
                frame: frameData,
                isLocal: true,
                targetLetter: targetLetter
            });
            
            console.log(`Frame sent for letter ${targetLetter}`);
        } catch (error) {
            console.error('Error capturing or sending frame:', error);
        }
    }
    
    // Update prediction feedback display
    function updatePredictionFeedback(prediction, confidence, targetLetter) {
        if (!prediction) {
            predictionFeedback.textContent = '';
            return;
        }
        
        // Format the confidence as a percentage
        const confidencePercent = Math.round(confidence * 100);
        
        // Create feedback text
        let feedbackText = `Detected: ${prediction} (${confidencePercent}%)`;
        
        // Add visual indication if it matches the target
        if (prediction === targetLetter) {
            predictionFeedback.className = 'prediction-feedback correct';
            feedbackText += ' ✓';
        } else {
            predictionFeedback.className = 'prediction-feedback incorrect';
            feedbackText += ' ✗';
        }
        
        predictionFeedback.textContent = feedbackText;
    }
    
    // Handle correct prediction
    function handleCorrectPrediction() {
        // Stop recognition temporarily
        recognitionActive = false;
        
        // Update UI
        feedbackMessage.textContent = `Great job! You correctly signed "${aslLetters[currentLetterIndex]}"`;
        feedbackMessage.className = 'feedback-message feedback-success';
        
        // Show next button
        nextQuizBtn.style.display = 'block';
        
        // Add points to score based on confidence
        const bonusPoints = Math.round(currentConfidence * 2);
        score += bonusPoints;
        currentScore.textContent = score;
    }
    
    // Move to next letter
    function nextLetter() {
        // Hide next button
        nextQuizBtn.style.display = 'none';
        
        // Select a new letter
        selectRandomLetter();
        
        // Reset feedback
        feedbackMessage.textContent = `Sign the letter "${aslLetters[currentLetterIndex]}"`;
        feedbackMessage.className = 'feedback-message';
        predictionFeedback.textContent = '';
        
        // Resume recognition
        recognitionActive = true;
    }
    
    // End the quiz
    function endQuiz(completed = true) {
        // Stop timer and recognition
        clearInterval(timerInterval);
        clearInterval(frameInterval);
        recognitionActive = false;
        quizActive = false;
        
        // Update UI
        if (completed) {
            feedbackMessage.textContent = `Quiz completed! Your final score is ${score}.`;
        } else {
            feedbackMessage.textContent = `Time's up! Your final score is ${score}.`;
        }
        feedbackMessage.className = 'feedback-message feedback-success';
        
        // Show start button to try again
        startQuizBtn.style.display = 'block';
        startQuizBtn.textContent = 'Try Again';
        nextQuizBtn.style.display = 'none';
        
        // Stop camera
        stopCamera();
    }
    
    // Event listeners
    startQuizBtn.addEventListener('click', startQuiz);
    nextQuizBtn.addEventListener('click', nextLetter);
    
    // Initialize socket connection when page loads
    initializeSocket();
});