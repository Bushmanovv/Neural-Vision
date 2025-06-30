// Navigation between sections
function showSection(sectionId) {
    document.querySelectorAll('.page-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
}

// Floating glowing particles
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 8 + 's';
        particle.style.animationDuration = (Math.random() * 3 + 5) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Navbar shadow on scroll
function handleNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 100) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
}

// Intersection Observer for entrance animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Train models by calling backend training endpoint
async function trainModels() {
    const trainBtn = document.getElementById('train-btn');
    const originalText = trainBtn.innerHTML;

    try {
        // Update button to show loading state
        trainBtn.innerHTML = '🔄 Training Models...';
        trainBtn.disabled = true;
        trainBtn.style.opacity = '0.7';
        trainBtn.style.cursor = 'not-allowed';

        // Show loading on all metric values
        document.querySelectorAll('[id$="-accuracy"], [id$="-precision"], [id$="-recall"], [id$="-f1"]').forEach(el => {
            el.innerText = 'Training...';
            el.style.color = '#feca57';
        });

        console.log('🚀 Starting model training...');

        // Call backend training endpoint
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`Training failed! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log('✅ Training completed:', result);

        // Show success message briefly
        trainBtn.innerHTML = '✅ Training Complete!';
        trainBtn.style.background = 'linear-gradient(45deg, #00ff88, #00ff88)';

        // Wait a moment then load the new metrics
        setTimeout(async () => {
            await loadModelMetrics();

            // Reset button after metrics are loaded
            setTimeout(() => {
                trainBtn.innerHTML = originalText;
                trainBtn.disabled = false;
                trainBtn.style.opacity = '1';
                trainBtn.style.cursor = 'pointer';
                trainBtn.style.background = 'linear-gradient(45deg, #00f5ff, #0080ff)';
            }, 1000);
        }, 1500);

    } catch (err) {
        console.error('❌ Training failed:', err);

        // Show error state
        trainBtn.innerHTML = '❌ Training Failed';
        trainBtn.style.background = 'linear-gradient(45deg, #ff6b6b, #ff8e8e)';

        // Show error on metrics
        document.querySelectorAll('[id$="-accuracy"], [id$="-precision"], [id$="-recall"], [id$="-f1"]').forEach(el => {
            el.innerText = 'Error';
            el.style.color = '#ff6b6b';
        });

        // Reset button after 3 seconds
        setTimeout(() => {
            trainBtn.innerHTML = originalText;
            trainBtn.disabled = false;
            trainBtn.style.opacity = '1';
            trainBtn.style.cursor = 'pointer';
            trainBtn.style.background = 'linear-gradient(45deg, #00f5ff, #0080ff)';
        }, 3000);
    }
}

// Load model metrics from Flask API and display in stats cards
async function loadModelMetrics() {
    try {
        console.log('📡 Fetching model metrics...');
        const response = await fetch('/metrics');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('📊 Received metrics data:', data);

        // Map your backend model names to frontend IDs
        const modelMapping = {
            'Naive Bayes': 'Naive-Bayes',
            'Random Forest': 'Random-Forest',
            'Neural Network (MLP)': 'Neural-Network-(MLP)',
            'MLP': 'Neural-Network-(MLP)', // Alternative name mapping
            'MLPClassifier': 'Neural-Network-(MLP)' // Another alternative
        };

        for (const [model, stats] of Object.entries(data)) {
            // Use mapping or fallback to replacing spaces with dashes
            const safeId = modelMapping[model] || model.replace(/\s+/g, '-');

            console.log(`🔄 Processing ${model} -> ${safeId}`);

            // Check if elements exist before updating
            const accuracyEl = document.getElementById(`${safeId}-accuracy`);
            const precisionEl = document.getElementById(`${safeId}-precision`);
            const recallEl = document.getElementById(`${safeId}-recall`);
            const f1El = document.getElementById(`${safeId}-f1`);

            if (accuracyEl) {
                accuracyEl.innerText = parseFloat(stats.accuracy).toFixed(1) + '%';
                accuracyEl.style.color = '#4ecdc4'; // Reset to original color
            } else {
                console.warn(`⚠️ Element not found: ${safeId}-accuracy`);
            }

            if (precisionEl) {
                precisionEl.innerText = parseFloat(stats.precision).toFixed(1) + '%';
                precisionEl.style.color = '#45b7d1';
            } else {
                console.warn(`⚠️ Element not found: ${safeId}-precision`);
            }

            if (recallEl) {
                recallEl.innerText = parseFloat(stats.recall).toFixed(1) + '%';
                recallEl.style.color = '#96ceb4';
            } else {
                console.warn(`⚠️ Element not found: ${safeId}-recall`);
            }

            if (f1El) {
                f1El.innerText = parseFloat(stats.f1_score || stats.f1).toFixed(1) + '%';
                f1El.style.color = '#feca57';
            } else {
                console.warn(`⚠️ Element not found: ${safeId}-f1`);
            }

            // Log confusion matrix if available
            if (stats.confusion_matrix) {
                console.log(`📊 ${model} Confusion Matrix:`);
                console.table(stats.confusion_matrix);
            }
        }

        console.log('✅ Metrics loaded successfully');

    } catch (err) {
        console.error('❌ Failed to load metrics:', err);

        // Show error message on all metric elements
        document.querySelectorAll('[id$="-accuracy"], [id$="-precision"], [id$="-recall"], [id$="-f1"]').forEach(el => {
            el.innerText = 'Error';
            el.style.color = '#ff6b6b';
        });
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Neural Vision...');

    // Create visual effects
    createParticles();

    // Setup animations for stats cards and matrix cards
    const animatedElements = document.querySelectorAll('.stats-card, .matrix-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Load initial metrics from backend
    loadModelMetrics();

    console.log('✅ Neural Vision initialized successfully');
});

// Setup scroll listener for navbar effects
window.addEventListener('scroll', handleNavbarScroll);