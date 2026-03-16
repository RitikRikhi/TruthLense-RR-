const API_BASE = "http://localhost:5000/api";

const claimInput = document.getElementById("claim-input");
const analyzeBtn = document.getElementById("analyze-btn");
const statusText = document.getElementById("status-text");
const scoreBadge = document.getElementById("score-badge");
const verdictText = document.getElementById("verdict-text");
const scoreCaption = document.getElementById("score-caption");
const explanationList = document.getElementById("explanation-list");
const sourcesList = document.getElementById("sources-list");

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  statusText.textContent = isLoading ? "Analyzing claim with TruthLense..." : "";
}

function formatScoreBadge(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    scoreBadge.textContent = "–";
    scoreBadge.classList.remove("medium", "low");
    return;
  }

  const s = Math.round(score);
  scoreBadge.textContent = `${s}`;
  scoreBadge.classList.remove("medium", "low");

  if (s < 40) {
    scoreBadge.classList.add("low");
  } else if (s < 70) {
    scoreBadge.classList.add("medium");
  }
}

function renderExplanation(explanation) {
  explanationList.innerHTML = "";

  if (!Array.isArray(explanation) || explanation.length === 0) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No detailed explanation available for this claim.";
    explanationList.appendChild(li);
    return;
  }

  explanation.forEach((line) => {
    const li = document.createElement("li");
    const clean = String(line).replace(/^[-*]\s*/, "");
    const lower = clean.toLowerCase();
    const isNegative =
      lower.includes("false") ||
      lower.includes("mislead") ||
      lower.includes("suspicious") ||
      lower.includes("not supported");

    if (isNegative) li.classList.add("negative");
    li.textContent = clean;
    explanationList.appendChild(li);
  });
}

function renderSources(sources) {
  sourcesList.innerHTML = "";

  if (!Array.isArray(sources) || sources.length === 0) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No external sources were used or available for this claim.";
    sourcesList.appendChild(li);
    return;
  }

  sources.forEach((s) => {
    const li = document.createElement("li");
    li.className = "source-item";

    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = s.title || s.source || "Source";

    const meta = document.createElement("div");
    meta.className = "source-meta";

    const domain = document.createElement("span");
    domain.className = "pill";
    domain.textContent = s.source || "web";

    const cred = document.createElement("span");
    cred.className = "pill pill-cred";
    if (typeof s.credibility === "number") {
      cred.textContent = `Credibility: ${Math.round(s.credibility)}/100`;
    } else {
      cred.textContent = "Credibility: –";
    }

    meta.appendChild(domain);
    meta.appendChild(cred);

    if (s.url) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = s.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open source";
      meta.appendChild(link);
    }

    li.appendChild(title);
    li.appendChild(meta);
    sourcesList.appendChild(li);
  });
}

async function analyzeClaim() {
  const text = claimInput.value.trim();
  if (!text) {
    statusText.textContent = "Please enter a claim to analyze.";
    return;
  }

  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `Request failed with ${res.status}`);
    }

    const data = await res.json();

    const score = typeof data.credibility_score === "number" ? data.credibility_score : null;
    formatScoreBadge(score);

    const verdict = data.prediction || data.verdict || "Unknown";
    verdictText.textContent = verdict;

    if (score === null) {
      scoreCaption.textContent = "Model could not compute a numeric credibility score.";
    } else if (score < 30) {
      scoreCaption.textContent = "Low credibility — likely false or heavily misleading.";
    } else if (score < 60) {
      scoreCaption.textContent = "Mixed evidence — needs human judgment and more context.";
    } else {
      scoreCaption.textContent = "High credibility — well supported by external evidence.";
    }

    renderExplanation(data.explanation);
    renderSources(data.sources);

    statusText.textContent = `Done in ${data.processing_time_seconds ?? "?"}s`;
  } catch (e) {
    console.error(e);
    statusText.textContent = `Error: ${e.message}`;
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener("click", analyzeClaim);

claimInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyzeClaim();
  }
});

