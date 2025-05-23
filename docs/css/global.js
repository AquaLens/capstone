// Get the current path
const currentPath = window.location.pathname.replace(/\/$/, '');

// Map paths to their corresponding link IDs
const linkMap = {
  '/': 'home-link',
  '/projects': 'projects-link',
  '/data': 'data-link',
  '/about': 'about-link',
};

// Find the active link and add a class
const activeLinkId = linkMap[currentPath];
if (activeLinkId) {
  const activeLink = document.getElementById(activeLinkId);
  if (activeLink) {
    activeLink.classList.add('active');
  }
}