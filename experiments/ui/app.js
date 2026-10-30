"use strict";

const element = id => document.getElementById(id);
const selected = name => document.querySelector(`input[name="${name}"]:checked`).value;
let configuration = null;
let prepared = null;
let busy = false;
let activeJob = null;
let completedRuns = [];
let photoPreviewURL = null;

if (new URLSearchParams(location.search).has("token")) history.replaceState({}, "", location.pathname);

function showError(message) {
  element("error").textContent = message;
  element("error").hidden = !message;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(Array.isArray(detail) ? detail.map(item => item.msg).join("; ") : detail || "The server could not complete the request.");
  }
  return payload;
}

function updateControls() {
  const distance = Number(element("distance").value);
  element("step-value").textContent = Number(element("step-size").value).toFixed(2);
  element("distance-value").textContent = distance.toFixed(2);
  element("similarity-value").textContent = (1 - distance).toFixed(2);
  const acceptable = prepared && prepared.similarity !== null && prepared.similarity >= 1 - distance;
  element("run").disabled = busy || !acceptable;
  element("run-hint").textContent = !prepared ? selected("mode") === "image" ? "Align & project the selected photo before generating." : "Prepare a starting face first." : acceptable ? `${Number(element("duration").value) * 24} frames · the reference stays fixed` : "The projected center does not meet this boundary.";
  const warning = prepared && !acceptable;
  element("projection-warning").hidden = !warning;
  element("projection-warning").textContent = warning ? "This reconstruction does not match the uploaded face closely enough at the selected threshold. Try more projection steps, a better photo, or a looser boundary before generating." : "";
  element("policy-description").textContent = configuration?.policies[selected("policy")].description || "";
}

function describeMode(mode) {
  element("seed-options").hidden = mode !== "seed";
  element("upload-options").hidden = mode !== "image";
  element("prepare-label").textContent = mode === "image" ? "Align & project photo" : "Prepare starting face";
  element("mode-description").textContent = {
    mean: "Start at the generator’s learned average W vector. The center stays fixed.",
    seed: `Map a reproducible Gaussian seed into shared W with truncation ψ = ${configuration?.seed_truncation_psi ?? 0.75}, then keep that face as the reference.`,
    image: "Align and project your photo. Identity is measured against the aligned upload, not just its reconstruction."
  }[mode];
}

function invalidateCenter() {
  prepared = null;
  for (const id of ["reference", "reconstruction"]) {
    element(`${id}-image`).hidden = true;
    element(`${id}-image`).removeAttribute("src");
    element(`${id}-image`).onerror = null;
    element(`${id}-placeholder`).hidden = false;
  }
  element("projection-score").hidden = true;
  element("download-projection").hidden = true;
  element("ready-badge").textContent = "Prepare a starting face";
  element("center-label").textContent = "Not prepared";
  element("center-title").textContent = "Prepare your new starting point.";
  element("center-description").textContent = "The previous starting point is cleared. Prepare this selection before generating a trajectory.";
  element("reference-caption").textContent = "Fixed reference";
  if (selected("mode") === "image" && photoPreviewURL) {
    element("reference-image").src = photoPreviewURL;
    element("reference-image").hidden = false;
    element("reference-placeholder").hidden = true;
    element("reference-caption").textContent = "Selected photo · not aligned yet";
    element("center-label").textContent = "Selected photo · not projected";
    element("center-title").textContent = "Photo selected. Ready to project.";
    element("center-description").textContent = "Click Align & project photo in the sidebar. After alignment and projection, this preview becomes the aligned identity reference and the generated reconstruction appears beside it.";
    element("ready-badge").textContent = "Photo selected · projection needed";
    element("reference-image").onerror = () => showError("This browser cannot preview the selected image. Choose a readable JPEG, PNG, or WebP photo.");
  }
  updateControls();
}

function changeMode() {
  showError("");
  describeMode(selected("mode"));
  invalidateCenter();
}

function setBusy(value) {
  busy = value;
  element("controls").disabled = value;
  element("progress-card").hidden = !value;
  element("cancel").disabled = value && !activeJob;
  element("cancel").textContent = "Cancel";
  updateControls();
}

function displayCenter(center) {
  prepared = center;
  document.querySelector(`input[name="mode"][value="${center.mode}"]`).checked = true;
  describeMode(center.mode);
  if (center.mode === "seed" && center.seed !== null && center.seed !== undefined) element("anchor-seed").value = center.seed;
  element("device").textContent = center.device;
  element("center-label").textContent = center.label;
  element("center-title").textContent = center.mode === "image" ? "Your face, projected into W." : center.mode === "seed" ? "A reproducible starting identity." : "The learned mean is your center.";
  element("center-description").textContent = center.mode === "image" ? "Left: the aligned uploaded photo, used as the fixed identity reference. Right: its generated reconstruction, where the ball starts. Projection stays in shared W; synthesis noise is held fixed." : "The generated center also serves as the fixed identity reference. The ball moves along two principal axes of W; the center stays pinned.";
  if (center.mode === "seed") element("center-description").textContent += ` Seed truncation ψ = ${center.truncation_psi ?? 1}; PCA axes remain untruncated.`;
  element("reference-caption").textContent = center.mode === "image" ? "Aligned upload · fixed reference" : "Fixed reference";
  for (const [id, url] of [["reference", center.reference_url], ["reconstruction", center.reconstruction_url]]) {
    element(`${id}-image`).onerror = null;
    element(`${id}-image`).src = url;
    element(`${id}-image`).hidden = false;
    element(`${id}-placeholder`).hidden = true;
  }
  element("projection-score").hidden = false;
  element("projection-score").textContent = center.similarity === null ? "No valid face detected in the reconstruction" : `Center similarity ${center.similarity.toFixed(3)} · distance ${center.distance.toFixed(3)}`;
  element("download-projection").hidden = !center.projection_url;
  if (center.projection_url) element("download-projection").href = center.projection_url;
  element("ready-badge").textContent = "Starting point ready";
  updateControls();
}

function displayResult(result) {
  element("empty-state").hidden = true;
  element("video").hidden = false;
  element("video").src = result.video_url;
  element("result-content").hidden = false;
  element("result-title").textContent = `${result.label} · ${configuration.policies[result.settings.reflection_policy].label}`;
  element("result-caption").textContent = `Step ${result.settings.step_size.toFixed(2)} · max distance ${result.settings.max_cosine_distance.toFixed(2)} · motion seed ${result.settings.trajectory_seed}`;
  element("result-tag").textContent = `${result.settings.duration}s · ${result.settings.duration * 24} frames`;
  element("stat-score").textContent = result.minimum_similarity.toFixed(3);
  element("stat-collisions").textContent = result.collisions;
  element("stat-diversity").textContent = result.diversity.greedy_representatives;
  element("stat-fallbacks").textContent = result.fallbacks;
  element("map-image").src = result.map_url;
  element("diagnostics").textContent = `${result.invalid_cells} invalid map cells · ${result.holds} collision-limit holds · ${result.radius_collisions} outer-radius collisions. Detection and computational-limit events are recorded separately from identity exits.`;
  for (const [id, url] of [["download-video", result.video_url], ["download-data", result.trajectory_url], ["download-metadata", result.metadata_url], ["download-map", result.map_url]]) element(id).href = url;
}

function displayHistory() {
  element("history-section").hidden = !completedRuns.length;
  element("history").replaceChildren();
  for (const run of completedRuns.slice(0, 9)) {
    const button = document.createElement("button");
    const title = document.createElement("strong");
    const subtitle = document.createElement("span");
    title.textContent = `${configuration.policies[run.settings.reflection_policy].label} · ${run.settings.duration}s`;
    subtitle.textContent = `${run.label} · distance ≤ ${run.settings.max_cosine_distance.toFixed(2)}`;
    button.append(title, subtitle);
    button.addEventListener("click", () => displayResult(run));
    element("history").append(button);
  }
}

async function followJob(identifier) {
  activeJob = identifier;
  setBusy(true);
  try {
    while (activeJob === identifier) {
      const job = await request(`/api/jobs/${identifier}`);
      element("stage").textContent = job.stage;
      element("progress").value = job.progress;
      element("progress-value").textContent = `${Math.round(job.progress * 100)}%`;
      if (job.status === "failed") throw new Error(job.error || "The experiment failed.");
      if (job.status === "cancelled") {
        element("ready-badge").textContent = "Cancelled";
        break;
      }
      if (job.status === "complete") {
        if (job.kind === "prepare") displayCenter(job.result);
        else {
          completedRuns = [job.result, ...completedRuns.filter(item => item.video_url !== job.result.video_url)];
          displayResult(job.result);
          displayHistory();
          element("ready-badge").textContent = "Experiment complete";
        }
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } catch (error) {
    showError(error.message);
  } finally {
    if (activeJob === identifier) {
      activeJob = null;
      setBusy(false);
    }
  }
}

async function prepareCenter() {
  showError("");
  const mode = selected("mode");
  const data = new FormData();
  data.set("mode", mode);
  data.set("seed", element("anchor-seed").value);
  data.set("steps", element("projection-steps").value);
  data.set("already_aligned", String(element("already-aligned").checked));
  if (mode === "image") {
    const photo = element("photo").files[0];
    if (!photo) return showError("Choose a starting face photo first.");
    if (photo.size > 10 * 1024 * 1024) return showError("Choose a photo smaller than 10 MB.");
    data.set("photo", photo);
  }
  prepared = null;
  setBusy(true);
  element("stage").textContent = mode === "image" ? "Uploading selected photo" : "Preparing starting face";
  element("ready-badge").textContent = mode === "image" ? "Aligning & projecting…" : "Preparing starting face…";
  try {
    const submitted = await request("/api/centers", {method: "POST", body: data});
    await followJob(submitted.job_id);
  } catch (error) {
    showError(error.message);
    setBusy(false);
  }
}

async function generate() {
  if (!prepared) return;
  showError("");
  const settings = {center_id: prepared.center_id, step_size: Number(element("step-size").value),
    max_cosine_distance: Number(element("distance").value), reflection_policy: selected("policy"),
    duration: Number(element("duration").value), grid_size: Number(element("grid-size").value),
    trajectory_seed: Number(element("trajectory-seed").value), extent: Number(element("extent").value)};
  setBusy(true);
  try {
    const submitted = await request("/api/runs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(settings)});
    await followJob(submitted.job_id);
  } catch (error) {
    showError(error.message);
    setBusy(false);
  }
}

element("prepare").addEventListener("click", prepareCenter);
element("run").addEventListener("click", generate);
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", changeMode));
document.querySelectorAll('input[name="policy"]').forEach(input => input.addEventListener("change", updateControls));
for (const id of ["step-size", "distance", "duration"]) element(id).addEventListener("input", updateControls);
for (const id of ["anchor-seed", "already-aligned", "projection-steps"]) element(id).addEventListener("change", invalidateCenter);
element("photo").addEventListener("change", () => {
  showError("");
  if (photoPreviewURL) URL.revokeObjectURL(photoPreviewURL);
  photoPreviewURL = null;
  const photo = element("photo").files[0];
  element("file-name").textContent = photo?.name || "or drop one here · JPG, PNG, WebP";
  if (photo && photo.size <= 10 * 1024 * 1024) photoPreviewURL = URL.createObjectURL(photo);
  else if (photo) showError("Choose a photo smaller than 10 MB.");
  invalidateCenter();
});
element("cancel").addEventListener("click", async () => {
  if (!activeJob) return;
  element("cancel").disabled = true;
  element("cancel").textContent = "Stopping…";
  try {await request(`/api/jobs/${activeJob}/cancel`, {method: "POST"});} catch (error) {showError(error.message);}
});
for (const name of ["dragenter", "dragover"]) element("dropzone").addEventListener(name, event => {event.preventDefault(); element("dropzone").classList.add("dragging");});
for (const name of ["dragleave", "drop"]) element("dropzone").addEventListener(name, event => {event.preventDefault(); element("dropzone").classList.remove("dragging");});
element("dropzone").addEventListener("drop", event => {
  if (busy) return;
  const file = event.dataTransfer.files[0];
  if (file) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    element("photo").files = transfer.files;
    element("photo").dispatchEvent(new Event("change"));
  }
});

async function initialize() {
  try {
    configuration = await request("/api/config");
    element("distance").value = configuration.default_distance;
    const existing = await request("/api/jobs");
    completedRuns = existing.filter(job => job.kind === "run" && job.status === "complete").map(job => job.result);
    displayHistory();
    if (completedRuns.length) displayResult(completedRuns[0]);
    const previous = existing.find(job => job.kind === "prepare" && job.status === "complete");
    if (previous) displayCenter(previous.result);
    updateControls();
    const active = existing.find(job => job.status === "running" || job.status === "queued");
    if (active) await followJob(active.id);
    else if (!previous) await prepareCenter();
  } catch (error) {
    showError(`Cannot connect to the UI server. Keep your SSH tunnel open and reload. ${error.message}`);
    element("ready-badge").textContent = "Connection unavailable";
  }
}

initialize();
