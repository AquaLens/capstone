// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", () => {
  // Get the current path and normalize it (remove trailing slash)
  const currentPath = window.location.pathname.replace(/\/$/, '');
  console.log("Current Path:", currentPath); // Debugging: Log the current path

  // Select all navigation links
  const navLinks = document.querySelectorAll('.nav-links a');

  // Loop through each link and check if its href matches the current path
  navLinks.forEach(link => {
    const linkPath = link.getAttribute('href').replace(/\/$/, ''); // Normalize href
    if (linkPath === currentPath) {
      link.classList.add('active'); // Add the 'active' class to the matching link
      console.log("Added 'active' class to:", link); // Debugging: Log the active link
    }
  });
});