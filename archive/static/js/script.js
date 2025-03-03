document.addEventListener("DOMContentLoaded", () => {
  const videoFeed = document.getElementById("videoFeed");
  const startButton = document.getElementById("startButton");
  const metrics = document.getElementById("metrics");
  const neckAngle = document.getElementById("neckAngle");
  const torsoAngle = document.getElementById("torsoAngle");
  const postureStatus = document.getElementById("postureStatus");

  let isStreaming = false;

  // Set default placeholder image
  videoFeed.src = "/static/images/placeholder.svg";

  // Function to update metrics
  function updateMetrics(data) {
    neckAngle.textContent = `${data.neck_angle.toFixed(1)}°`;
    torsoAngle.textContent = `${data.torso_angle.toFixed(1)}°`;

    postureStatus.textContent =
      data.posture.charAt(0).toUpperCase() + data.posture.slice(1);
    postureStatus.className =
      data.posture === "good" ? "status-good" : "status-bad";
  }

  async function startDetection() {
    try {
      // Start the video feed
      videoFeed.src = `/video_feed?t=${Date.now()}`;
      isStreaming = true;
      startButton.textContent = "Stop Detection";
      startButton.classList.remove("bg-indigo-600", "hover:bg-indigo-700");
      startButton.classList.add("bg-red-600", "hover:bg-red-700");
      metrics.classList.remove("hidden");

      // Handle video feed errors
      videoFeed.onerror = () => {
        console.error("Video feed error");
        stopDetection();
      };

      // Start metrics polling
      pollMetrics();
    } catch (error) {
      console.error("Failed to start detection:", error);
      stopDetection();
    }
  }

  async function stopDetection() {
    try {
      // Reset video feed
      videoFeed.src = "/static/images/placeholder.svg";
      isStreaming = false;
      startButton.textContent = "Start Detection";
      startButton.classList.remove("bg-red-600", "hover:bg-red-700");
      startButton.classList.add("bg-indigo-600", "hover:bg-indigo-700");
      metrics.classList.add("hidden");

      // Notify backend to stop camera
      const response = await fetch("/stop_camera");
      if (!response.ok) {
        throw new Error("Failed to stop camera");
      }
    } catch (error) {
      console.error("Error stopping detection:", error);
    }
  }

  async function pollMetrics() {
    if (!isStreaming) return;

    try {
      const response = await fetch("/metrics");
      const data = await response.json();
      updateMetrics(data);
    } catch (error) {
      console.error("Error polling metrics:", error);
    }

    if (isStreaming) {
      setTimeout(pollMetrics, 200); // Poll every 200ms
    }
  }

  startButton.addEventListener("click", async () => {
    if (!isStreaming) {
      await startDetection();
    } else {
      await stopDetection();
    }
  });

  // Handle page unload
  window.addEventListener("beforeunload", () => {
    if (isStreaming) {
      fetch("/stop_camera").catch(console.error);
    }
  });
});
