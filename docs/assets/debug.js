console.log("header.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  // Handle both with and without trailing slashes
  const currentPath = window.location.pathname.replace(/\/$/, '') || '/';
  console.log("=== NAVIGATION DEBUGGING ===");
  console.log("Raw pathname:", window.location.pathname);
  console.log("Processed currentPath:", currentPath);

  const navLinks = document.querySelectorAll('.nav-links a');
  console.log("Found", navLinks.length, "navigation links");

  navLinks.forEach((link, index) => {
    console.log(`\n--- Processing Link ${index} ---`);
    console.log("Link href:", link.href);
    console.log("Link text:", link.textContent.trim());
    
    // Skip external links
    if (link.href.startsWith('http') && !link.href.includes(window.location.hostname)) {
      console.log("→ Skipping external link");
      return;
    }

    try {
      const linkUrl = new URL(link.href);
      const linkPath = linkUrl.pathname.replace(/\/$/, '') || '/';
      
      console.log("Link pathname:", linkUrl.pathname);
      console.log("Processed linkPath:", linkPath);
      console.log(`Match check: "${linkPath}" === "${currentPath}"`, linkPath === currentPath);
      
      // Remove active class first
      link.classList.remove('active');
      
      if (linkPath === currentPath) {
        link.classList.add('active');
        console.log("✓ ACTIVE CLASS ADDED!");
        
        // Force style update
        link.style.color = '#ffffff';
        link.style.fontWeight = 'bold';
        
      } else {
        console.log("✗ No match");
      }
      
      console.log("Final classes:", link.className);
      
    } catch (error) {
      console.error("Error processing link:", link.href, error);
    }
  });
  
  console.log("=== END NAVIGATION DEBUGGING ===");
});