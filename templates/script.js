// Get DOM elements
const liveSentimentBtn = document.getElementById("live-sentiment-btn");
const imageSentimentBtn = document.getElementById("image-sentiment-btn");
const liveSentimentContainer = document.getElementById("live-sentiment-container");
const imageSentimentContainer = document.getElementById("image-sentiment-container");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const startButton = document.getElementById("start-button");
const sentimentText = document.getElementById("sentiment-text");

// Set up event listeners
liveSentimentBtn.addEventListener("click", showLiveSentiment);
imageSentimentBtn.addEventListener("click", showImageSentiment);
startButton.addEventListener("click", analyzeImage);

// Show live sentiment detection section
function showLiveSentiment() {
  liveSentimentBtn.classList.add("active");
  imageSentimentBtn.classList.remove("active");
  liveSentimentContainer.style.display = "block";
  imageSentimentContainer.style.display = "none";
}

// Show image sentiment detection section
function showImageSentiment() {
  imageSentimentBtn.classList.add("active");
  liveSentimentBtn.classList.remove("active");
  imageSentimentContainer.style.display = "block";
  liveSentimentContainer.style.display = "none";
}
