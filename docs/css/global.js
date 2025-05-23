// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", () => {
  // Get the current path and normalize it (remove trailing slash)
  const currentPath = window.location.pathname.replace(/\/$/, '');
  console.log("Current Path:", currentPath); // Debugging: Log the current path

  // Map paths to their corresponding link IDs
  const linkMap = {
    '/': 'home-link',
    '/projects': 'projects-link',
    '/data': 'data-link',
    '/about': 'about-link',
  };

  // Find the active link ID based on the current path
  const activeLinkId = linkMap[currentPath];
  console.log("Active Link ID:", activeLinkId); // Debugging: Log the active link ID

  // If an active link ID is found, add the 'active' class to the corresponding link
  if (activeLinkId) {
    const activeLink = document.getElementById(activeLinkId);
    if (activeLink) {
      activeLink.classList.add('active');
      console.log("Added 'active' class to:", activeLink); // Debugging: Log the active link element
    } else {
      console.warn("No element found with ID:", activeLinkId); // Debugging: Warn if the element is not found
    }
  } else {
    console.warn("No matching path found in linkMap for:", currentPath); // Debugging: Warn if no matching path
  }
});