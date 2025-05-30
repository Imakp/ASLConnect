
// Content for different tabs
const tabContents = {
    alphabet: [
        // Page 1
        [
            { letter: 'A', image: '/static/images/public/alphabets/A_test.jpg' },
            { letter: 'B', image: '/static/images/public/alphabets/B_test.jpg' },
            { letter: 'C', image: '/static/images/public/alphabets/C_test.jpg' },
            { letter: 'D', image: '/static/images/public/alphabets/D_test.jpg' },
            { letter: 'E', image: '/static/images/public/alphabets/E_test.jpg' },
            { letter: 'F', image: '/static/images/public/alphabets/F_test.jpg' },
            { letter: 'G', image: '/static/images/public/alphabets/G_test.jpg' },
            { letter: 'H', image: '/static/images/public/alphabets/H_test.jpg' },
            { letter: 'I', image: '/static/images/public/alphabets/I_test.jpg' },
            { letter: 'J', image: '/static/images/public/alphabets/J_test.jpg' },
            { letter: 'K', image: '/static/images/public/alphabets/K_test.jpg' },
            { letter: 'L', image: '/static/images/public/alphabets/L_test.jpg' }
        ],
        // Page 2
        [
            { letter: 'M', image: '/static/images/public/alphabets/M_test.jpg' },
            { letter: 'N', image: '/static/images/public/alphabets/N_test.jpg' },
            { letter: 'O', image: '/static/images/public/alphabets/O_test.jpg' },
            { letter: 'P', image: '/static/images/public/alphabets/P_test.jpg' },
            { letter: 'Q', image: '/static/images/public/alphabets/Q_test.jpg' },
            { letter: 'R', image: '/static/images/public/alphabets/R_test.jpg' },
            { letter: 'S', image: '/static/images/public/alphabets/S_test.jpg' },
            { letter: 'T', image: '/static/images/public/alphabets/T_test.jpg' },
            { letter: 'U', image: '/static/images/public/alphabets/U_test.jpg' },
            { letter: 'V', image: '/static/images/public/alphabets/V_test.jpg' },
            { letter: 'W', image: '/static/images/public/alphabets/W_test.jpg' },
            { letter: 'X', image: '/static/images/public/alphabets/X_test.jpg' }
        ],
        // Page 3
        [
            { letter: 'Y', image: '/static/images/public/alphabets/Y_test.jpg' },
            { letter: 'Z', image: '/static/images/public/alphabets/Z_test.jpg' }
        ]
    ],
    numbers: [
        // Page 1
        [
            { letter: '0', image: '/static/images/public/numbers/Sign 0.jpeg' },
            { letter: '1', image: '/static/images/public/numbers/Sign 1.jpeg' },
            { letter: '2', image: '/static/images/public/numbers/Sign 2.jpeg' },
            { letter: '3', image: '/static/images/public/numbers/Sign 3.jpeg' },
            { letter: '4', image: '/static/images/public/numbers/Sign 4.jpeg' },
            { letter: '5', image: '/static/images/public/numbers/Sign 5.jpeg' },
            { letter: '6', image: '/static/images/public/numbers/Sign 6.jpeg' },
            { letter: '7', image: '/static/images/public/numbers/Sign 7.jpeg' },
            { letter: '8', image: '/static/images/public/numbers/Sign 8.jpeg' },
            { letter: '9', image: '/static/images/public/numbers/Sign 9.jpeg' },
            // { letter: '10', image: 'num10.png' }
        ]
    ],
    'common-phrases': [
        // Page 1
        [
            { letter: 'Thank You', image: '/static/images/public/phrases/thanku.png' },
            { letter: 'Please', image: '/static/images/public/phrases/please.jpeg' },
            { letter: 'Sorry', image: '/static/images/public/phrases/sorry.png' },
            { letter: 'Help', image: '/static/images/public/phrases/help.webp' },
            { letter: 'Yes', image: '/static/images/public/phrases/yes.webp' },
            { letter: 'No', image: '/static/images/public/phrases/no.gif' },
            { letter: 'Love', image: '/static/images/public/phrases/love.jpg' },
            { letter: 'Friend', image: '/static/images/public/phrases/friend.webp' }
        ]
    ],
    greetings: [
        // Page 1
        [
            { letter: 'Hello', image: '/static/images/public/greetings/hello.jpg' },
            { letter: 'Good Morning', image: '/static/images/public/greetings/morning.webp' },
            { letter: 'Good Afternoon', image: '/static/images/public/greetings/afternoon.jpg' },
            { letter: 'Good Night', image: '/static/images/public/greetings/night.jpg' },
            { letter: 'How are you?', image: '/static/images/public/greetings/how_are_u.png' },
            { letter: 'Nice to meet you', image: '/static/images/public/greetings/nice-to-meet.jpg' },
            { letter: 'Welcome', image: '/static/images/public/greetings/welcome.webp' },
            { letter: 'Goodbye', image: '/static/images/public/greetings/goodbye.webp' }
        ]
    ],
    resources: [
        // Page 1
        [
            {
                letter: "ASL Alphabet A-Z",
                image: "/static/images/public/resources/alphabet.png",
                description: "Learn the complete ASL alphabet from A to Z with clear demonstrations.",
                videoUrl: "https://www.youtube.com/embed/7suKo9kCTus?si=jf8DIScjknRcB72H"
            },
            {
                letter: "ASL Numbers 1-20",
                image: "/static/images/public/resources/number.jpg",
                description: "Learn how to sign numbers 1-20 in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/SP6maLr2DDI?si=2VI4fplH0lAIO6gs"
            },
            {
                letter: "ASL Colors",
                image: "/static/images/public/resources/colour.jpg",
                description: "Learn how to sign different colors in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/Dsh0YWwfn-M?si=KG7FdH1V_-Ws6g1R"
            },
            {
                letter: "ASL Family Signs",
                image: "/static/images/public/resources/family.jpg",
                description: "Learn signs for family members like mother, father, sister, brother, etc.",
                videoUrl: "https://www.youtube.com/embed/RdCnFf8BslA?si=Ub6ARX1sWYnFGuFX"
            },
            {
                letter: "ASL Emotions & Feelings",
                image: "/static/images/public/resources/emotion.png",
                description: "Learn how to express emotions and feelings in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/L-EuYYqMsuU?si=sy45_tlhcIdl9KxF"
            },
            {
                letter: "ASL Common Phrases",
                image: "/static/images/public/resources/phrase.jpg",
                description: "Learn everyday phrases and expressions in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/jCUaMdXsI7w?si=T6ZtFeYaPYkLk8BU"
            }
        ],
        // Page 2
        [
            {
                letter: "ASL Animals",
                image: "/static/images/public/resources/animal.jpg",
                description: "Learn how to sign different animals in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/3V8kP6scQpY?si=fNQrv6Ece8H6zl3x"
            },
            {
                letter: "ASL Food Signs",
                image: "/static/images/public/resources/food.jpg",
                description: "Learn signs for different foods and drinks in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/ghQr2hzdx8k?si=jXEbrAbCY7RqoEGq"
            },
            {
                letter: "ASL Weather Signs",
                image: "/static/images/public/resources/weather.jpg",
                description: "Learn how to sign weather conditions in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/xfhvTHsceTU?si=88inPxCu1IOMwKKp"
            },
            {
                letter: "ASL Conversation Practice",
                image: "/static/images/public/resources/conversation.jpg",
                description: "Practice basic conversations in American Sign Language.",
                videoUrl: "https://www.youtube.com/embed/lGrssQASztg?si=ZTS0DX49talQiK3b"
            }
        ]
    ]
};

document.addEventListener('DOMContentLoaded', function() {
    // Navigation functionality - Show/hide content sections
    const navLinks = document.querySelectorAll('.sidebar-nav a[data-section]');
    const quickLinks = document.querySelectorAll('.quick-card-btn[data-section]');
    const contentSections = document.querySelectorAll('.content-section');
    
    // Quiz data - This would normally come from a backend
    const quizData = {
        score: 85,
        completed: 12,
        recent: [
            { name: 'Alphabet Quiz', score: 90, date: '2023-11-15' },
            { name: 'Numbers Quiz', score: 80, date: '2023-11-10' },
            { name: 'Common Phrases', score: 85, date: '2023-11-05' }
        ]
    };
    
    // Update quiz stats in the dashboard
    function updateQuizStats() {
        const scoreElement = document.querySelector('.dashboard-stats .stat-info h3:first-of-type');
        const completedElement = document.querySelector('.dashboard-stats .stat-info h3:last-of-type');
        
        if (scoreElement && completedElement) {
            scoreElement.textContent = quizData.score + '%';
            completedElement.textContent = quizData.completed;
        }
    }
    
    // Call the function to update stats
    updateQuizStats();
    
    // Function to activate a section
    function activateSection(sectionId) {
        // Hide all sections
        contentSections.forEach(section => {
            section.classList.remove('active');
            section.style.display = 'none';
        });
        
        // Show the selected section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            targetSection.style.display = 'block';
            
            // Update navigation active state
            navLinks.forEach(link => {
                if (link.getAttribute('data-section') === sectionId) {
                    link.parentElement.classList.add('active');
                } else {
                    link.parentElement.classList.remove('active');
                }
            });
        }
    }
    
    // Set initial state - show dashboard section
    activateSection('dashboard');
    
    // Add click event to sidebar navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('data-section');
            activateSection(sectionId);
        });
    });
    
    // Add click event to quick access buttons
    quickLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('data-section');
            activateSection(sectionId);
        });
    });
    
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const alphabetGrid = document.querySelector('.alphabet-grid');
    const alphabetTitle = document.querySelector('.alphabet-title');
    const paginationContainer = document.querySelector('.alphabet-pagination');
    
    let currentTab = 'alphabet';
    let currentPage = 0;
    
    // Function to render alphabet cards
    function renderCards(tab, page) {
        if (!tabContents[tab] || !tabContents[tab][page]) {
            alphabetGrid.innerHTML = '<p>Content coming soon!</p>';
            return;
        }
        
        const cards = tabContents[tab][page];
        alphabetGrid.innerHTML = '';
        
        cards.forEach(item => {
            const card = document.createElement('div');
            card.className = 'alphabet-card';
            
            if (tab === 'resources') {
                card.className = 'alphabet-card resource-card';
                
                card.innerHTML = `
                    <div class="alphabet-image resource-image">
                        <img src="${item.image}" alt="${item.letter}">
                    </div>
                    <div class="alphabet-letter resource-title">${item.letter}</div>
                    <div class="alphabet-action">Click to watch</div>
                `;
                
                // Add click event to show modal with video
                card.addEventListener('click', function() {
                    showLetterModal(item.letter, item.image, item.description, item.videoUrl);
                });
            } else {
                card.innerHTML = `
                    <div class="alphabet-image">
                        <img src="${item.image}" alt="ASL ${item.letter}">
                    </div>
                    <div class="alphabet-letter">${item.letter}</div>
                    <div class="alphabet-action">Click to learn</div>
                `;
                
                // Add click event to show modal
                card.addEventListener('click', function() {
                    showLetterModal(item.letter, item.image);
                });
            }
            
            alphabetGrid.appendChild(card);
        });
    }
    
    // Function to update pagination
    function updatePagination(tab) {
        if (!tabContents[tab] || tabContents[tab].length <= 1) {
            paginationContainer.style.display = 'none';
            return;
        }
        
        paginationContainer.style.display = 'flex';
        paginationContainer.innerHTML = '';
        
        for (let i = 0; i < tabContents[tab].length; i++) {
            const btn = document.createElement('button');
            btn.className = 'pagination-btn' + (i === currentPage ? ' active' : '');
            btn.textContent = i + 1;
            
            btn.addEventListener('click', function() {
                currentPage = i;
                renderCards(currentTab, currentPage);
                
                // Update active state
                document.querySelectorAll('.pagination-btn').forEach((b, idx) => {
                    b.classList.toggle('active', idx === i);
                });
            });
            
            paginationContainer.appendChild(btn);
        }
    }
    
    // Function to show letter modal when a card is clicked
    function showLetterModal(letter, image, description, videoUrl) {
        // Create modal elements
        const modal = document.createElement('div');
        modal.className = 'alphabet-modal';
        
        const modalContent = document.createElement('div');
        modalContent.className = videoUrl ? 'alphabet-modal-content resource-modal' : 'alphabet-modal-content';
        
        const closeBtn = document.createElement('span');
        closeBtn.className = 'close-modal';
        closeBtn.innerHTML = '&times;';
        closeBtn.onclick = function() {
            document.body.removeChild(modal);
        };
        
        const letterTitle = document.createElement('h2');
        letterTitle.className = 'modal-letter-title';
        letterTitle.textContent = letter;
        
        const letterImageContainer = document.createElement('div');
        letterImageContainer.className = 'modal-letter-image';
        
        // Check if it's a video resource
        if (videoUrl) {
            const videoFrame = document.createElement('iframe');
            videoFrame.src = videoUrl;
            videoFrame.width = "100%";
            videoFrame.height = "400";
            videoFrame.frameBorder = "0";
            videoFrame.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
            videoFrame.allowFullscreen = true;
            
            letterImageContainer.appendChild(videoFrame);
        } else {
            const letterImage = document.createElement('img');
            letterImage.src = image;
            letterImage.alt = `ASL ${letter}`;
            
            letterImageContainer.appendChild(letterImage);
        }
        
        const letterDescription = document.createElement('p');
        letterDescription.className = 'modal-letter-description';
        
        if (description) {
            letterDescription.textContent = description;
        } else {
            letterDescription.textContent = `This is how you sign "${letter}" in American Sign Language. Practice by mimicking the hand position shown in the image or watch a video demonstration.`;
        }
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'modal-buttons';
        
        // Only show practice buttons for non-resource items
        if (!videoUrl) {
            const practiceBtn = document.createElement('button');
            practiceBtn.className = 'modal-practice-btn';
            practiceBtn.innerHTML = '<i class="fas fa-camera"></i> Practice with Camera';
            practiceBtn.onclick = function() {
                alert('Camera practice feature coming soon!');
            };
            
            const videoBtn = document.createElement('button');
            videoBtn.className = 'modal-video-btn';
            videoBtn.innerHTML = '<i class="fas fa-play-circle"></i> Learn by Video';
            videoBtn.onclick = function() {
                alert('Video tutorial coming soon!');
            };
            
            buttonContainer.appendChild(videoBtn);
            buttonContainer.appendChild(practiceBtn);
        }
        
        modalContent.appendChild(closeBtn);
        modalContent.appendChild(letterTitle);
        modalContent.appendChild(letterImageContainer);
        modalContent.appendChild(letterDescription);
        modalContent.appendChild(buttonContainer);
        
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
        
        // Add active class after a small delay to trigger animation
        setTimeout(() => {
            modal.classList.add('active');
        }, 10);
        
        // Close modal when clicking outside of content
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }
    
    // Add click event listeners to all alphabet cards
    document.addEventListener('DOMContentLoaded', function() {
        const alphabetCards = document.querySelectorAll('.alphabet-card');
        
        alphabetCards.forEach(card => {
            card.addEventListener('click', function() {
                const letter = this.querySelector('.alphabet-letter').textContent;
                const image = this.querySelector('.alphabet-image img').src;
                showLetterModal(letter, image);
            });
        });
    });
    
    // Initialize with alphabet tab
    renderCards(currentTab, currentPage);
    updatePagination(currentTab);
    
    // Tab button click handlers
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Update content based on selected tab
            currentTab = this.getAttribute('data-tab');
            currentPage = 0;
            
            // Update title
            const tabName = currentTab.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase());
            alphabetTitle.textContent = `ASL ${tabName}`;
            
            // Render cards for the selected tab
            renderCards(currentTab, currentPage);
            updatePagination(currentTab);
        });
    });
    
    // Search functionality
    const searchInput = document.getElementById('alphabetSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            const cards = document.querySelectorAll('.alphabet-card');
            cards.forEach(card => {
                const letter = card.querySelector('.alphabet-letter').textContent.toLowerCase();
                
                if (letter.includes(searchTerm)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }
    
    // Video call room functionality
    const createRoomBtn = document.getElementById('createRoomBtn');
    const joinRoomBtn = document.getElementById('joinRoomBtn');
    const roomIdInput = document.getElementById('roomIdInput');
    
    if (createRoomBtn && joinRoomBtn && roomIdInput) {
        // Generate a random room ID
        createRoomBtn.addEventListener('click', function() {
            const randomRoomId = Math.random().toString(36).substring(2, 8);
            window.location.href = `/video-call?room=${randomRoomId}`;
        });
        
        // Join an existing room
        joinRoomBtn.addEventListener('click', function() {
            const roomId = roomIdInput.value.trim();
            if (roomId) {
                window.location.href = `/video-call?room=${roomId}`;
            } else {
                alert('Please enter a valid Room ID');
            }
        });
        
        // Join room when pressing Enter in the input field
        roomIdInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                joinRoomBtn.click();
            }
        });
    }
    
    // Handle join buttons in recent rooms
    const joinButtons = document.querySelectorAll('.join-btn');
    joinButtons.forEach(button => {
        button.addEventListener('click', function() {
            const roomName = this.parentElement.querySelector('.room-details h4').textContent;
            const roomId = roomName.toLowerCase().replace(/\s+/g, '-');
            window.location.href = `/video-call?room=${roomId}`;
        });
    });
});