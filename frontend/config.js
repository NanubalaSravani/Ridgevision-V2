window.RIDGEVISION_API_URL =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000/predict"
    : `${window.location.origin}/predict`;

