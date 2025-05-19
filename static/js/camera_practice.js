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
    
    // Initialize
    updateReferenceImage('A');
    
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
    }
    
    // Camera functionality
    startCameraBtn.addEventListener('click', async function() {
        try {
            if (stream) {
                // Stop camera if already running
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                startCameraBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
                captureBtn.disabled = true;
                cameraFeed.srcObject = null;
                feedbackMessage.textContent = `Select a letter or number and start the camera to begin practice.`;
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
            
            feedbackMessage.textContent = `Camera started. Position your hand to sign "${currentLetter}" and click Capture.`;
            
        } catch (err) {
            console.error('Error accessing camera:', err);
            feedbackMessage.textContent = `Error accessing camera: ${err.message}`;
        }
    });
    
    // Capture image
    captureBtn.addEventListener('click', function() {
        if (!stream) return;
        
        // Simulate recognition with random accuracy
        const accuracy = Math.floor(Math.random() * 101);
        updateAccuracy(accuracy);
        
        if (accuracy > 80) {
            feedbackMessage.textContent = `Great job! Your sign for "${currentLetter}" was recognized with high accuracy.`;
        } else if (accuracy > 50) {
            feedbackMessage.textContent = `Good attempt! Your sign for "${currentLetter}" was recognized, but could use some improvement.`;
        } else {
            feedbackMessage.textContent = `Keep practicing! Your sign for "${currentLetter}" was not clearly recognized. Try adjusting your hand position.`;
        }
    });
    
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
});