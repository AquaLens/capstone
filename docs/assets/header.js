console.log("global.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const currentPath = window.location.pathname.replace(/\/$/, '');
  console.log("Current Path:", currentPath);

  const navLinks = document.querySelectorAll('.nav-links a');

  navLinks.forEach(link => {
    const linkPath = new URL(link.href).pathname.replace(/\/$/, '');
    console.log(`Comparing: ${linkPath} === ${currentPath}`);
    
    if (linkPath === currentPath) {
      link.classList.add('active');
      console.log("Added 'active' class to:", link);
    }
  });
});
