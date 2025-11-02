// JavaScript Code

// Navigation Handling
document.addEventListener('DOMContentLoaded', () => {
    const navbarLinks = document.querySelectorAll('.nav-link');
    navbarLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const targetPath = link.getAttribute('href');
            navigateTo(targetPath);
        });
    });
    
    // Mobile Menu Toggle
    const mobileMenuButton = document.querySelector('.mobile-menu-button');
    mobileMenuButton.addEventListener('click', () => {
        document.querySelector('.navbar-links').classList.toggle('visible');
    });

    // Smooth Scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Handle CTA Button
    document.querySelector('.cta-button').addEventListener('click', () => {
        openModal('contact-modal');
    });
});

// Navigation function
function navigateTo(path) {
    window.history.pushState({}, '', path);
    handleRouting();
}

// Routing Handling
function handleRouting() {
    const path = window.location.pathname;
    // Handle page rendering based on `path`
    // Example: renderHomePage(), renderServicesPage(), etc.
}

// Modal handling
function openModal(targetId) {
    const modal = document.getElementById(targetId);
    if (modal) {
        modal.classList.add('open');
        modal.querySelector('.close').addEventListener('click', () => {
            modal.classList.remove('open');
        });
    }
}

// FeatureList Component Rendering
function renderFeatureList() {
    const features = [
        { name: "Giải pháp AI", icon: "🤖" },
        { name: "Cloud Hosting", icon: "☁️" },
        { name: "Bảo mật mạng", icon: "🔒" }
    ];
    const featureListContainer = document.querySelector('.feature-list');
    featureListContainer.innerHTML = features.map(feature => `
        <div class="feature-card">
            <span class="feature-icon">${feature.icon}</span>
            <span class="feature-name">${feature.name}</span>
        </div>
    `).join('');
}

// Submit Handle
document.querySelectorAll('.contact-form').forEach(form => {
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const formData = new FormData(form);
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const result = await response.json();
            alert('Form submitted successfully');
        } catch (error) {
            console.error('There was a problem with the form submission:', error);
            alert('Form submission failed');
        }
    });
});

// Initial Render
function render() {
    renderFeatureList();
    handleRouting();
}

window.onpopstate = handleRouting;
render();