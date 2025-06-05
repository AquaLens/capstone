// Get the current page path
const currentPath = window.location.pathname.replace(/\/$/, '');

// Find all navigation links in the header
const navLinks = document.querySelectorAll('.nav-links a');

// Loop through each navigation link to check if it matches current page
navLinks.forEach(link => {
    // Extract the path from each link's href 
    const linkPath = new URL(link.href).pathname.replace(/\/$/, '');
    
    // If the link path matches the current page path add active class
    if (linkPath === currentPath) {
        link.classList.add('active');
    }
});