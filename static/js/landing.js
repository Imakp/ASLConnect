document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const hamburger = document.querySelector('.hamburger');
    const body = document.querySelector('body');
    
    // Create mobile nav if it doesn't exist
    if (!document.querySelector('.mobile-nav')) {
        const mobileNav = document.createElement('div');
        mobileNav.className = 'mobile-nav';
        
        const closeMenu = document.createElement('div');
        closeMenu.className = 'close-menu';
        closeMenu.innerHTML = '&times;';
        
        const navLinks = document.querySelector('.nav-links').cloneNode(true);
        navLinks.className = 'mobile-nav-links';
        
        const ctaButton = document.querySelector('.cta-button').cloneNode(true);
        
        mobileNav.appendChild(closeMenu);
        mobileNav.appendChild(navLinks);
        mobileNav.appendChild(ctaButton);
        
        body.appendChild(mobileNav);
        
        // Toggle mobile navigation
        hamburger.addEventListener('click', function() {
            mobileNav.classList.add('active');
            body.style.overflow = 'hidden';
        });
        
        // Close mobile navigation
        closeMenu.addEventListener('click', function() {
            mobileNav.classList.remove('active');
            body.style.overflow = 'auto';
        });
        
        // Close mobile navigation when clicking a link
        const mobileNavLinks = mobileNav.querySelectorAll('a');
        mobileNavLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileNav.classList.remove('active');
                body.style.overflow = 'auto';
            });
        });
    }
    
    // Header scroll effect
    const header = document.querySelector('header');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
    
    // Scroll animation
    const fadeElements = document.querySelectorAll('.features-container, .steps-container, .demo-container, .benefits-container, .contact-container');
    fadeElements.forEach(element => {
        element.classList.add('fade-in');
    });
    
    // Intersection Observer for scroll animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, {
        threshold: 0.1
    });
    
    fadeElements.forEach(element => {
        observer.observe(element);
    });
    
    // Contact form submission
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form values
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const message = document.getElementById('message').value;
            
            // Simple validation
            if (!name || !email || !message) {
                showFormMessage('Please fill in all fields', 'error');
                return;
            }
            
            // Simulate form submission
            const submitButton = contactForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.textContent = 'Sending...';
            
            // Simulate API call with timeout
            setTimeout(() => {
                showFormMessage('Message sent successfully! We\'ll get back to you soon.', 'success');
                contactForm.reset();
                submitButton.disabled = false;
                submitButton.textContent = 'Send Message';
            }, 1500);
        });
    }
    
    // Function to show form submission message
    function showFormMessage(message, type) {
        // Remove any existing message
        const existingMessage = document.querySelector('.form-message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Create new message
        const messageElement = document.createElement('div');
        messageElement.className = `form-message ${type}`;
        messageElement.textContent = message;
        
        // Add to DOM
        contactForm.appendChild(messageElement);
        
        // Remove after 5 seconds
        setTimeout(() => {
            messageElement.remove();
        }, 5000);
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const headerHeight = document.querySelector('header').offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add animation to demo video
    const demoVideo = document.querySelector('.demo-video video');
    if (demoVideo) {
        demoVideo.addEventListener('mouseover', function() {
            this.setAttribute('controls', true);
        });
        
        demoVideo.addEventListener('mouseout', function() {
            if (!this.paused) return;
            this.removeAttribute('controls');
        });
    }
    
    // Add placeholder image if demo video is not available
    const videoElement = document.querySelector('.demo-video video');
    if (videoElement) {
        videoElement.addEventListener('error', function() {
            const img = document.createElement('img');
            img.src = '/static/images/demo-placeholder.jpg';
            img.alt = 'ASL Recognition Demo';
            img.className = 'demo-placeholder';
            
            this.parentNode.replaceChild(img, this);
        });
    }
});