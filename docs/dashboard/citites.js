let cityCache = {
  "paris": { country: "france", lat: 48.8566, lng: 2.3522 },
  "berlin": { country: "germany", lat: 52.5200, lng: 13.4050 }
};

try {
  const savedCache = JSON.parse(localStorage.getItem('cityCache')) || {};
  cityCache = { ...cityCache, ...savedCache };
} catch (e) {
  console.log('Using default city cache');
}