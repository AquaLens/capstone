console.log("header.js loaded successfully");

document.addEventListener("DOMContentLoaded", function() {
  console.log("DOM loaded, starting navigation script");
  
  try {
    var currentPath = window.location.pathname;
    if (currentPath.endsWith('/') && currentPath.length > 1) {
      currentPath = currentPath.slice(0, -1);
    }
    
    console.log("Current path:", currentPath);
    
    var navLinks = document.querySelectorAll('.nav-links a');
    console.log("Found " + navLinks.length + " navigation links");
    
    for (var i = 0; i < navLinks.length; i++) {
      var link = navLinks[i];
      console.log("Processing link " + i + ": " + link.href);
      
      // Skip external links (LinkedIn)
      if (link.href.indexOf('linkedin.com') !== -1 || 
          (link.href.indexOf('http') === 0 && link.href.indexOf(window.location.hostname) === -1)) {
        console.log("Skipping external link");
        continue;
      }
      
      var linkPath = link.getAttribute('href');
      if (linkPath.endsWith('/') && linkPath.length > 1) {
        linkPath = linkPath.slice(0, -1);
      }
      
      console.log("Comparing: '" + linkPath + "' with '" + currentPath + "'");
      
      // Remove active class first
      link.classList.remove('active');
      
      if (linkPath === currentPath) {
        link.classList.add('active');
        console.log("MATCH! Added active class to: " + link.textContent);
        
        // Force styling as backup
        link.style.color = '#ffffff';
        link.style.fontWeight = 'bold';
      }
    }
    
  } catch (error) {
    console.error("Error in navigation script:", error);
  }
  
  console.log("Navigation script completed");
});