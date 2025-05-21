document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startQuizBtn = document.getElementById('startQuizBtn');
    const nextQuizBtn = document.getElementById('nextQuizBtn');
    const quizCamera = document.getElementById('quizCamera');
    const cameraOverlay = document.getElementById('cameraOverlay');
    const quizLetter = document.getElementById('quizLetter');
    const referenceImage = document.getElementById('referenceImage');
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
    let handLandmarker = null;
    let lastPredictionTime = 0;
    let currentPrediction = '';
    let currentConfidence = 0;
    let confidenceThreshold = 0.90; // Increased threshold to 90% for stricter validation
    let socket = null;
    let frameInterval = null;
    
    // ASL letters to quiz
    const aslLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
    
    // Initialize Socket.IO connection for real model predictions
    function initializeSocket() {
        socket = io();
        
        // Listen for ASL recognition results from the server
        socket.on('asl-result', (data) => {
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
                } else if (text) {
                    // Incorrect prediction
                    feedbackMessage.textContent = `That looks like "${text}". Try signing "${targetLetter}" instead.`;
                    feedbackMessage.className = 'feedback-message feedback-error';
                } else {
                    // No valid prediction
                    feedbackMessage.textContent = 'No valid sign detected. Please try again.';
                    feedbackMessage.className = 'feedback-message feedback-warning';
                }
            }
        });
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
        
        // Update UI
        startQuizBtn.style.display = 'none';
        nextQuizBtn.style.display = 'none';
        feedbackMessage.textContent = 'Quiz started! Sign the letter shown.';
        feedbackMessage.className = 'feedback-message';
        
        // Start camera
        await startCamera();
        
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
        
        // Update reference image
        referenceImage.src = `/static/images/public/alphabets/${letter}_test.jpg`;
        referenceImage.alt = `ASL ${letter}`;
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
        
        // Capture frames at regular intervals
        frameInterval = setInterval(() => {
            if (!quizActive || !recognitionActive) return;
            
            // Capture and send frame for processing
            captureAndSendFrame();
        }, 200); // 5 frames per second
    }
    
    // Capture and send frame for ASL recognition
    function captureAndSendFrame() {
        if (!quizCamera || !quizCamera.srcObject) return;
        
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = quizCamera.videoWidth;
        canvas.height = quizCamera.videoHeight;
        
        // Draw the current video frame to the canvas
        ctx.drawImage(quizCamera, 0, 0, canvas.width, canvas.height);
        
        // Convert the canvas to a base64-encoded image
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send the frame to the server for ASL recognition
        if (socket) {
            socket.emit('frame', {
                room: 'quiz-room', // Use a fixed room for quiz
                frame: frameData,
                isLocal: true,
                targetLetter: aslLetters[currentLetterIndex] // Send the target letter for validation
            });
        }
    }
    
    // Update the prediction feedback display
    function updatePredictionFeedback(prediction, confidence, targetLetter) {
        if (predictionFeedback) {
            if (prediction) {
                const confidencePercent = Math.round(confidence * 100);
                const isCorrect = prediction === targetLetter;
                const confidenceClass = confidence >= confidenceThreshold ? 'high-confidence' : 'low-confidence';
                const correctClass = isCorrect ? 'correct-prediction' : 'incorrect-prediction';
                
                predictionFeedback.innerHTML = `
                    <div class="prediction ${correctClass} ${confidenceClass}">
                        <span class="prediction-letter">${prediction}</span>
                        <span class="prediction-confidence">${confidencePercent}%</span>
                    </div>
                `;
                predictionFeedback.style.display = 'block';
            } else {
                predictionFeedback.innerHTML = '<div class="prediction">No hand detected</div>';
                predictionFeedback.style.display = 'block';
            }
        }
    }
    
    // Handle correct prediction
    function handleCorrectPrediction() {
        // Only proceed if quiz is still active
        if (!quizActive) return;
        
        // Stop recognition
        recognitionActive = false;
        if (frameInterval) {
            clearInterval(frameInterval);
            frameInterval = null;
        }
        
        // Update UI
        feedbackMessage.textContent = `Correct! You signed "${aslLetters[currentLetterIndex]}" correctly.`;
        feedbackMessage.className = 'feedback-message feedback-success';
        
        // Show next button
        nextQuizBtn.style.display = 'flex';
        
        // Update score based on remaining time
        currentScore.textContent = score;
    }
    
    // End the quiz
    function endQuiz(completed) {
        quizActive = false;
        clearInterval(timerInterval);
        
        if (recognitionActive) {
            recognitionActive = false;
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
        }
        
        // Update UI
        if (completed) {
            feedbackMessage.textContent = 'Quiz completed! Well done.';
            feedbackMessage.className = 'feedback-message feedback-success';
        } else {
            feedbackMessage.textContent = 'Time\'s up! Quiz ended.';
            feedbackMessage.className = 'feedback-message feedback-warning';
        }
        
        // Show start button to restart
        startQuizBtn.style.display = 'flex';
        startQuizBtn.querySelector('span').textContent = 'Restart Quiz';
        
        // Hide next button
        nextQuizBtn.style.display = 'none';
        
        // Stop camera
        stopCamera();
    }
    
    // Event listeners
    startQuizBtn.addEventListener('click', startQuiz);
    
    nextQuizBtn.addEventListener('click', function() {
        // Move to next letter
        selectRandomLetter();
        
        // Reset UI
        nextQuizBtn.style.display = 'none';
        feedbackMessage.textContent = `Sign the letter "${aslLetters[currentLetterIndex]}".`;
        feedbackMessage.className = 'feedback-message';
        
        // Restart recognition
        recognitionActive = true;
        startFrameCapture();
    });
});