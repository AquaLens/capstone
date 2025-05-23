console.log("header.js loaded");

// Remove DOMContentLoaded wrapper since the DOM is already loaded when this runs
const currentPath = window.location.pathname.replace(/\/$/, '');
console.log("Current Path:", currentPath);

const navLinks = document.querySelectorAll('.nav-links a');
console.log("Found nav links:", navLinks.length);

navLinks.forEach(link => {
  const linkPath = new URL(link.href).pathname.replace(/\/$/, '');
  console.log(`Comparing: ${linkPath} === ${currentPath}`);
  
  if (linkPath === currentPath) {
    link.classList.add('active');
    console.log("Added 'active' class to:", link);
  }
});