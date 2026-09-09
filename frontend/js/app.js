const API_URL = window.RIDGEVISION_API_URL
  || (window.location.protocol === "file:"
    ? "http://127.0.0.1:8000/predict"
    : `${window.location.origin}/predict`);
const classLabels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"];

const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const previewImage = document.querySelector("#previewImage");
const predictButton = document.querySelector("#predictButton");
const apiStatus = document.querySelector("#apiStatus");
const predictedClass = document.querySelector("#predictedClass");
const confidence = document.querySelector("#confidence");
const probabilities = document.querySelector("#probabilities");
const heatmapImage = document.querySelector("#heatmapImage");
const heatmapPlaceholder = document.querySelector("#heatmapPlaceholder");
const pipelineSteps = [...document.querySelectorAll(".pipeline-step")];

let selectedFile = null;

function renderEmptyProbabilities() {
  probabilities.innerHTML = classLabels
    .map(
      (label) => `
        <div class="probability-row">
          <div class="mb-2 flex items-center justify-between text-sm">
            <span class="font-semibold">${label}</span>
            <span class="text-slate-500">0.00%</span>
          </div>
          <div class="probability-bar"><div class="probability-fill" style="width: 0%"></div></div>
        </div>
      `,
    )
    .join("");
}

function setStatus(text, tone = "neutral") {
  apiStatus.textContent = text;
  apiStatus.className = "rounded px-2 py-1 text-xs";
  const tones = {
    neutral: "bg-slate-100 text-slate-600",
    busy: "bg-cyan-100 text-cyan-800",
    ok: "bg-emerald-100 text-emerald-800",
    error: "bg-rose-100 text-rose-800",
  };
  apiStatus.classList.add(...tones[tone].split(" "));
}

function setPipelineActive(active) {
  pipelineSteps.forEach((step, index) => {
    step.classList.toggle("active", active && index < 5);
  });
}

function setFile(file) {
  selectedFile = file;
  predictButton.disabled = !file;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("border-cyan-500", "bg-cyan-50");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("border-cyan-500", "bg-cyan-50");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("border-cyan-500", "bg-cyan-50");
  setFile(event.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  setStatus("running", "busy");
  setPipelineActive(true);

  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Prediction failed");
    }

    const result = await response.json();
    predictedClass.textContent = result.predicted_class;
    confidence.textContent = `${result.confidence.toFixed(2)}%`;
    renderProbabilities(result.all_probabilities);

    heatmapImage.src = result.tier1_attention_b64 || result.grad_cam_b64;
    heatmapImage.classList.remove("hidden");
    heatmapPlaceholder.classList.add("hidden");
    const isAccepted = result.decision_status !== "PREDICTION_WITHHELD";
    setStatus(isAccepted ? "complete" : "withheld (ambiguity)", isAccepted ? "ok" : "busy");

  } catch (error) {
    setStatus("error", "error");
    alert(error.message);
  } finally {
    predictButton.disabled = false;
    setTimeout(() => setPipelineActive(false), 700);
  }
});

function renderProbabilities(values) {
  probabilities.innerHTML = classLabels
    .map((label) => {
      const value = values[label] || 0;
      return `
        <div class="probability-row">
          <div class="mb-2 flex items-center justify-between text-sm">
            <span class="font-semibold">${label}</span>
            <span class="text-slate-500">${value.toFixed(2)}%</span>
          </div>
          <div class="probability-bar"><div class="probability-fill" style="width: ${value}%"></div></div>
        </div>
      `;
    })
    .join("");
}

renderEmptyProbabilities();
