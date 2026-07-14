/**
 * AI Contract Analysis Pipeline — Frontend logic.
 *
 * Vanilla JavaScript only (no frameworks, no build step). Handles:
 *   - Drag & drop / browse multi-file PDF selection
 *   - POST /upload  → save selected PDFs to the server
 *   - POST /analyze → run the pipeline and persist JSON/CSV reports
 *   - GET  /download/json, GET /download/csv → fetch generated reports
 *   - Progress, spinner, and success/error UI feedback
 *   - Enabling the download buttons once a report exists
 *
 * The module is organized into small, single-purpose sections:
 *   1. DOM references
 *   2. Application state
 *   3. Utility helpers
 *   4. File selection (drag & drop + browse)
 *   5. Upload step (/upload)
 *   6. Analysis step (/analyze)
 *   7. UI feedback helpers (alerts, progress, buttons, stepper, result cards)
 *   8. Event bindings
 *   9. Init
 *
 * NOTE ON RESULT-CARD IDS: this file expects (but does not require) three
 * elements for the "processed / failed / processing time" summary cards:
 *   #result-processed, #result-failed, #result-time
 * If your markup uses different ids, update RESULT_CARD_IDS below — every
 * lookup is null-safe, so a missing element is skipped rather than throwing.
 */

(() => {
    "use strict";

    const RESULT_CARD_IDS = {
        processed: "result-processed",
        failed: "result-failed",
        time: "result-time",
    };

    /* ------------------------------------------------------------------
     * 1. DOM references
     * ------------------------------------------------------------------ */

    /**
     * Safe wrapper around document.getElementById that warns (instead of
     * silently failing later) when an expected element is missing.
     * @param {string} id
     * @param {boolean} [required]
     * @returns {HTMLElement|null}
     */
    function getEl(id, required = false) {
        const el = document.getElementById(id);
        if (!el && required) {
            console.warn(`[script.js] Required element #${id} was not found in the DOM.`);
        }
        return el;
    }

    /**
     * Safe wrapper around querySelector that warns when an expected
     * element is missing.
     * @param {string} selector
     * @param {boolean} [required]
     * @returns {Element|null}
     */
    function queryEl(selector, required = false) {
        const el = document.querySelector(selector);
        if (!el && required) {
            console.warn(`[script.js] Required element "${selector}" was not found in the DOM.`);
        }
        return el;
    }

    const uploadForm = getEl("upload-form", true);
    const dropzone = getEl("dropzone", true);
    const fileInput = getEl("file-input", true);

    const fileSummary = getEl("file-summary", true);
    const fileCountEl = getEl("file-count", true);
    const fileListEl = getEl("file-list", true);
    const clearFilesBtn = getEl("clear-files", true);

    const startBtn = getEl("start-analysis-btn", true);
    const startBtnLabel = startBtn ? startBtn.textContent.trim() : "Start Analysis";

    const progressSection = getEl("progress-section", true);
    const progressBarEl = getEl("progress-bar", true);
    const progressBarTrack = progressBarEl ? progressBarEl.closest(".progress") : null;
    const progressStatusEl = getEl("progress-status", true);

    const successAlert = getEl("success-alert", true);
    const successDetailEl = getEl("success-detail", true);
    const errorAlert = getEl("error-alert", true);
    const errorDetailEl = getEl("error-detail", true);

    const downloadJsonBtn = getEl("download-json", true);
    const downloadCsvBtn = getEl("download-csv", true);

    // Optional result-card elements (see RESULT_CARD_IDS note above).
    const resultProcessedEl = getEl(RESULT_CARD_IDS.processed);
    const resultFailedEl = getEl(RESULT_CARD_IDS.failed);
    const resultTimeEl = getEl(RESULT_CARD_IDS.time);

    const stepEls = {
        upload: queryEl('.pipeline-steps__item[data-step="upload"]'),
        analyze: queryEl('.pipeline-steps__item[data-step="analyze"]'),
        download: queryEl('.pipeline-steps__item[data-step="download"]'),
    };

    /* ------------------------------------------------------------------
     * 2. Application state
     * ------------------------------------------------------------------ */
    const state = {
        /** @type {File[]} PDFs currently selected for upload. */
        selectedFiles: [],
        /** @type {boolean} True while an upload/analyze run is in flight. */
        isRunning: false,
        /** @type {boolean} True once the current selection has been uploaded successfully. */
        uploadSucceeded: false,
        /** @type {string[]} Filenames confirmed as uploaded by the server (or echoed locally). */
        uploadedFileNames: [],
        /** @type {number|null} performance.now() timestamp when analysis started. */
        analysisStartedAt: null,
    };

    /* ------------------------------------------------------------------
     * 3. Utility helpers
     * ------------------------------------------------------------------ */

    /**
     * Format a byte count as a short, human-readable file size.
     * @param {number} bytes
     * @returns {string}
     */
    function formatFileSize(bytes) {
        if (!Number.isFinite(bytes)) return "";
        if (bytes < 1024) return `${bytes} B`;
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        return `${(kb / 1024).toFixed(1)} MB`;
    }

    /**
     * Format a duration in milliseconds as a short, human-readable string
     * (e.g. "850 ms", "3.2 s", "1m 04s").
     * @param {number} ms
     * @returns {string}
     */
    function formatDuration(ms) {
        if (!Number.isFinite(ms) || ms < 0) return "—";
        if (ms < 1000) return `${Math.round(ms)} ms`;
        const totalSeconds = ms / 1000;
        if (totalSeconds < 60) return `${totalSeconds.toFixed(1)} s`;
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.round(totalSeconds % 60);
        return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
    }

    /**
     * Determine whether a File object looks like a PDF, based on both
     * its MIME type and its extension (some browsers/OSes omit the type).
     * @param {File} file
     * @returns {boolean}
     */
    function isPdfFile(file) {
        const hasPdfType = file.type === "application/pdf";
        const hasPdfExtension = file.name.toLowerCase().endsWith(".pdf");
        return hasPdfType || hasPdfExtension;
    }

    /**
     * Safely parse a fetch Response body as JSON, tolerating empty or
     * non-JSON bodies so a malformed server response never throws an
     * unhandled error.
     * @param {Response} response
     * @returns {Promise<any>}
     */
    async function parseJsonSafely(response) {
        try {
            return await response.json();
        } catch (_error) {
            return null;
        }
    }

    /**
     * Extract the most useful human-readable message from an API error
     * response or a thrown network error.
     * @param {Response|null} response
     * @param {any} body
     * @param {Error|null} networkError
     * @returns {string}
     */
    function extractErrorMessage(response, body, networkError) {
        if (networkError) {
            return "Could not reach the server. Check your connection and try again.";
        }
        if (body && typeof body.detail === "string") {
            return body.detail;
        }
        if (body && typeof body.message === "string") {
            return body.message;
        }
        if (response) {
            return `Request failed with status ${response.status}.`;
        }
        return "An unexpected error occurred.";
    }

    /**
     * Pause execution briefly so progress/status transitions are visibly
     * animated instead of jumping instantly between states.
     * @param {number} ms
     * @returns {Promise<void>}
     */
    function wait(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    /* ------------------------------------------------------------------
     * 4. File selection (drag & drop + browse)
     * ------------------------------------------------------------------ */

    /**
     * Replace the current file selection with a new set of PDFs,
     * filtering out anything that isn't a PDF, then re-render the UI.
     * Any new selection invalidates a previous successful upload, since
     * the server hasn't seen these files yet.
     * @param {FileList|File[]} incomingFiles
     */
    function handleFileSelection(incomingFiles) {
        const files = Array.from(incomingFiles);
        const pdfFiles = files.filter(isPdfFile);
        const rejectedCount = files.length - pdfFiles.length;

        state.selectedFiles = pdfFiles;
        state.uploadSucceeded = false;
        state.uploadedFileNames = [];
        renderFileList();
        disableDownloads();
        setStep("upload");

        if (pdfFiles.length === 0) {
            showError(
                rejectedCount > 0
                    ? "None of the selected files are PDFs. Please choose .pdf files only."
                    : "No files were selected."
            );
            return;
        }

        if (rejectedCount > 0) {
            showError(
                rejectedCount === 1
                    ? "1 file was skipped because it is not a PDF."
                    : `${rejectedCount} files were skipped because they are not PDFs.`
            );
        } else {
            hideAlerts();
        }
    }

    /** Render the selected-files summary panel and update the Start button. */
    function renderFileList() {
        if (!fileListEl || !fileCountEl || !fileSummary) return;
        const count = state.selectedFiles.length;

        fileCountEl.textContent = String(count);
        fileListEl.innerHTML = "";

        state.selectedFiles.forEach((file) => {
            const item = document.createElement("li");
            item.className = "file-list__item file-list__item--enter";
            item.innerHTML = `
                <i class="bi bi-file-earmark-pdf" aria-hidden="true"></i>
                <span class="file-list__name">${escapeHtml(file.name)}</span>
                <span class="file-list__size">${formatFileSize(file.size)}</span>
                <span class="file-list__status" data-role="status" aria-hidden="true"></span>
            `;
            fileListEl.appendChild(item);
            // Let the enter animation play, then settle to a static state.
            requestAnimationFrame(() => item.classList.remove("file-list__item--enter"));
        });

        fileSummary.hidden = count === 0;
        updateStartButtonAvailability();
    }

    /**
     * Mark each rendered file-list row as uploaded, using the server's
     * reported filenames when available, falling back to the local
     * selection order otherwise.
     */
    function markFileListAsUploaded() {
        if (!fileListEl) return;
        const rows = fileListEl.querySelectorAll(".file-list__item");
        rows.forEach((row, index) => {
            const label = state.uploadedFileNames[index] || null;
            const statusEl = row.querySelector('[data-role="status"]');
            row.classList.add("file-list__item--uploaded");
            if (statusEl) {
                statusEl.innerHTML = '<i class="bi bi-check-circle-fill" aria-hidden="true"></i>';
                statusEl.setAttribute("aria-hidden", "false");
                statusEl.setAttribute("title", label ? `Uploaded as ${label}` : "Uploaded");
            }
        });
    }

    /** Clear any per-file "uploaded" markers (e.g. before a fresh run). */
    function clearFileListUploadMarkers() {
        if (!fileListEl) return;
        const rows = fileListEl.querySelectorAll(".file-list__item");
        rows.forEach((row) => {
            row.classList.remove("file-list__item--uploaded");
            const statusEl = row.querySelector('[data-role="status"]');
            if (statusEl) {
                statusEl.innerHTML = "";
                statusEl.setAttribute("aria-hidden", "true");
                statusEl.removeAttribute("title");
            }
        });
    }

    /** Clear the current file selection entirely. */
    function clearFileSelection() {
        state.selectedFiles = [];
        state.uploadSucceeded = false;
        state.uploadedFileNames = [];
        if (fileInput) fileInput.value = "";
        renderFileList();
        disableDownloads();
        hideAlerts();
        setStep("upload");
    }

    /**
     * Escape a string for safe insertion into innerHTML.
     * @param {string} value
     * @returns {string}
     */
    function escapeHtml(value) {
        const div = document.createElement("div");
        div.textContent = value;
        return div.innerHTML;
    }

    /* ------------------------------------------------------------------
     * 5. Upload step (POST /upload)
     * ------------------------------------------------------------------ */

    /**
     * Upload the currently selected PDFs to the server.
     * @returns {Promise<{success: boolean, files: number, filenames?: string[]}>}
     */
    async function uploadSelectedFiles() {
        const formData = new FormData();
        state.selectedFiles.forEach((file) => formData.append("files", file));

        let response;
        try {
            response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
        } catch (networkError) {
            throw new Error(extractErrorMessage(null, null, networkError));
        }

        const body = await parseJsonSafely(response);

        if (!response.ok) {
            throw new Error(extractErrorMessage(response, body, null));
        }

        return body || {};
    }

    /* ------------------------------------------------------------------
     * 6. Analysis step (POST /analyze)
     * ------------------------------------------------------------------ */

    /**
     * Trigger the analysis pipeline on the server and parse its JSON
     * response (processed/failed counts, etc.).
     * @returns {Promise<{success: boolean, processed: number, failed: number}>}
     */
    async function runAnalysisPipeline() {
        let response;
        try {
            response = await fetch("/analyze", { method: "POST" });
        } catch (networkError) {
            throw new Error(extractErrorMessage(null, null, networkError));
        }

        const body = await parseJsonSafely(response);

        if (!response.ok) {
            throw new Error(extractErrorMessage(response, body, null));
        }

        return body || {};
    }

    /**
     * Orchestrate the full upload → analyze flow triggered by the
     * "Start Analysis" button, updating all UI feedback along the way.
     * Analysis only begins once the upload step has actually succeeded.
     * @param {Event} event
     */
    async function handleStartAnalysis(event) {
        event.preventDefault();

        if (state.isRunning || state.selectedFiles.length === 0) {
            return;
        }

        state.isRunning = true;
        hideAlerts();
        disableDownloads(); // Ensure downloads are disabled before processing starts.
        clearFileListUploadMarkers();
        setStartButtonLoading(true);
        showProgress();
        setStep("upload");

        // --- Upload phase -------------------------------------------------
        try {
            setProgress(8, "Uploading\u2026");
            const uploadResult = await uploadSelectedFiles();

            state.uploadSucceeded = true;
            state.uploadedFileNames =
                Array.isArray(uploadResult.filenames) && uploadResult.filenames.length
                    ? uploadResult.filenames
                    : state.selectedFiles.map((f) => f.name);
            markFileListAsUploaded();

            setProgress(40, "Uploaded. Preparing analysis\u2026");
        } catch (error) {
            state.uploadSucceeded = false;
            setProgress(100, "Failed");
            showError(error.message);
            state.isRunning = false;
            setStartButtonLoading(false);
            return;
        }

        // Analysis is gated on the upload step above having succeeded.
        if (!state.uploadSucceeded) {
            state.isRunning = false;
            setStartButtonLoading(false);
            return;
        }

        // --- Analyze phase --------------------------------------------------
        try {
            setStep("analyze");
            setProgress(55, "Analyzing\u2026");

            state.analysisStartedAt = performance.now();
            const result = await runAnalysisPipeline();
            const elapsedMs = performance.now() - state.analysisStartedAt;

            await animateProgressTo(100);
            setProgress(100, "Completed");
            setStep("download");
            showSuccess(result, elapsedMs);
            enableDownloads(); // Only enabled after a successful /analyze response.
        } catch (error) {
            setProgress(100, "Failed");
            disableDownloads(); // Keep downloads disabled on failure.
            showError(error.message);
        } finally {
            state.isRunning = false;
            setStartButtonLoading(false);
        }
    }

    /* ------------------------------------------------------------------
     * 7. UI feedback helpers
     * ------------------------------------------------------------------ */

    /** Show the progress panel and reset it to an initial state. */
    function showProgress() {
        if (!progressSection) return;
        progressSection.hidden = false;
        progressSection.classList.add("progress-section--visible");
        setProgress(0, "Starting\u2026");
    }

    /**
     * Update the progress bar's fill and status text. The bar itself
     * transitions smoothly via CSS (transition: width), so repeated
     * calls with increasing percentages animate naturally.
     * @param {number} percent  0–100
     * @param {string} statusText
     */
    function setProgress(percent, statusText) {
        if (!progressBarEl || !progressStatusEl) return;
        const clamped = Math.max(0, Math.min(100, percent));
        progressBarEl.style.width = `${clamped}%`;

        if (progressBarTrack) {
            progressBarTrack.setAttribute("aria-valuenow", String(clamped));
        }

        const isTerminal = statusText === "Completed" || statusText === "Failed";
        progressBarEl.classList.toggle("progress-bar--error", statusText === "Failed");
        progressBarEl.classList.toggle("progress-bar--success", statusText === "Completed");

        progressStatusEl.innerHTML =
            !isTerminal
                ? `<span class="spinner spinner--accent" aria-hidden="true"></span> ${statusText}`
                : `<i class="bi ${statusText === "Completed" ? "bi-check-circle-fill" : "bi-x-circle-fill"}" aria-hidden="true"></i> ${statusText}`;
    }

    /**
     * Smoothly step the progress bar up to a target percentage in a few
     * small increments, purely for a nicer perceived-completion animation.
     * @param {number} target
     * @returns {Promise<void>}
     */
    async function animateProgressTo(target) {
        if (!progressBarEl) return;
        const current = parseFloat(progressBarEl.style.width) || 0;
        const steps = 4;
        for (let i = 1; i <= steps; i += 1) {
            const next = current + ((target - current) * i) / steps;
            progressBarEl.style.width = `${Math.min(100, next)}%`;
            if (progressBarTrack) {
                progressBarTrack.setAttribute("aria-valuenow", String(Math.round(next)));
            }
            // eslint-disable-next-line no-await-in-loop
            await wait(80);
        }
    }

    /**
     * Toggle the Start button's loading state: disabled, spinner icon,
     * and updated label, or restored to its default appearance.
     * @param {boolean} isLoading
     */
    function setStartButtonLoading(isLoading) {
        if (!startBtn) return;
        startBtn.classList.toggle("is-loading", isLoading);
        startBtn.innerHTML = isLoading
            ? '<span class="spinner" aria-hidden="true"></span> Processing\u2026'
            : `<i class="bi bi-play-fill" aria-hidden="true"></i> ${startBtnLabel}`;
        updateStartButtonAvailability();
    }

    /** Enable/disable the Start button based on current selection + run state. */
    function updateStartButtonAvailability() {
        if (!startBtn) return;
        startBtn.disabled = state.selectedFiles.length === 0 || state.isRunning;
    }

    /**
     * Populate the optional processed / failed / processing-time result
     * cards, if present in the DOM.
     * @param {number} processed
     * @param {number} failed
     * @param {number} elapsedMs
     */
    function populateResultCards(processed, failed, elapsedMs) {
        if (resultProcessedEl) resultProcessedEl.textContent = String(processed);
        if (resultFailedEl) resultFailedEl.textContent = String(failed);
        if (resultTimeEl) resultTimeEl.textContent = formatDuration(elapsedMs);
    }

    /** Clear the optional result cards back to a neutral placeholder state. */
    function clearResultCards() {
        if (resultProcessedEl) resultProcessedEl.textContent = "—";
        if (resultFailedEl) resultFailedEl.textContent = "—";
        if (resultTimeEl) resultTimeEl.textContent = "—";
    }

    /**
     * Display the success alert with processed/failed counts and
     * processing time, and populate any dedicated result-card elements.
     * @param {{processed?: number, failed?: number}} result
     * @param {number} elapsedMs  Frontend-measured /analyze round-trip time.
     */
    function showSuccess(result, elapsedMs) {
        const processed = Number.isFinite(result?.processed) ? result.processed : 0;
        const failed = Number.isFinite(result?.failed) ? result.failed : 0;
        const durationLabel = formatDuration(elapsedMs);

        populateResultCards(processed, failed, elapsedMs);

        if (successDetailEl) {
            successDetailEl.innerHTML = `
                <div class="result-summary">
                    <span class="result-summary__item result-summary__item--ok">
                        <i class="bi bi-check-circle-fill" aria-hidden="true"></i>
                        ${processed} PDF${processed === 1 ? "" : "s"} processed
                    </span>
                    ${
                        failed > 0
                            ? `<span class="result-summary__item result-summary__item--fail">
                                   <i class="bi bi-exclamation-triangle-fill" aria-hidden="true"></i>
                                   ${failed} PDF${failed === 1 ? "" : "s"} failed
                               </span>`
                            : ""
                    }
                    <span class="result-summary__item result-summary__item--time">
                        <i class="bi bi-stopwatch" aria-hidden="true"></i>
                        ${durationLabel}
                    </span>
                </div>
                <span>Reports are ready to download.</span>
            `;
        }

        if (successAlert) {
            successAlert.hidden = false;
            successAlert.classList.add("alert--enter");
            requestAnimationFrame(() => successAlert.classList.remove("alert--enter"));
        }
        if (errorAlert) errorAlert.hidden = true;
    }

    /**
     * Display the error alert with a descriptive message (including
     * backend-provided error text extracted via extractErrorMessage).
     * @param {string} message
     */
    function showError(message) {
        if (errorDetailEl) errorDetailEl.textContent = message || "An unexpected error occurred.";
        if (errorAlert) {
            errorAlert.hidden = false;
            errorAlert.classList.add("alert--enter");
            requestAnimationFrame(() => errorAlert.classList.remove("alert--enter"));
        }
        if (successAlert) successAlert.hidden = true;
        clearResultCards();
    }

    /** Hide both the success and error alert regions. */
    function hideAlerts() {
        if (successAlert) successAlert.hidden = true;
        if (errorAlert) errorAlert.hidden = true;
    }

    /** Enable the JSON/CSV download buttons (only call after a successful analyze). */
    function enableDownloads() {
        [downloadJsonBtn, downloadCsvBtn].forEach((btn) => {
            if (!btn) return;
            btn.classList.remove("disabled");
            btn.classList.add("is-ready");
            btn.removeAttribute("aria-disabled");
            btn.removeAttribute("tabindex");
        });
    }

    /** Disable the JSON/CSV download buttons (before processing starts, or on failure). */
    function disableDownloads() {
        [downloadJsonBtn, downloadCsvBtn].forEach((btn) => {
            if (!btn) return;
            btn.classList.add("disabled");
            btn.classList.remove("is-ready");
            btn.setAttribute("aria-disabled", "true");
            btn.setAttribute("tabindex", "-1");
        });
    }

    /**
     * Mark the given pipeline stage as active and prior stages as complete
     * in the visual stepper.
     * @param {"upload"|"analyze"|"download"} activeStep
     */
    function setStep(activeStep) {
        const order = ["upload", "analyze", "download"];
        const activeIndex = order.indexOf(activeStep);

        order.forEach((step, index) => {
            const el = stepEls[step];
            if (!el) return;
            el.classList.toggle("is-active", index === activeIndex);
            el.classList.toggle("is-complete", index < activeIndex);
        });
    }

    /* ------------------------------------------------------------------
     * 8. Event bindings (each bound exactly once, at module load)
     * ------------------------------------------------------------------ */

    // Browse via file input.
    if (fileInput) {
        fileInput.addEventListener("change", (event) => {
            handleFileSelection(event.target.files);
        });
    }

    if (dropzone) {
        // Click anywhere on the drop zone (except the label, which already
        // opens the native file dialog) to open the file picker too.
        dropzone.addEventListener("click", (event) => {
            if (event.target.closest("label")) return;
            if (fileInput) fileInput.click();
        });

        // Keyboard accessibility: Enter/Space activates the drop zone.
        dropzone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                if (fileInput) fileInput.click();
            }
        });

        // Drag & drop, with a subtle animated highlight while a file is over
        // the zone (see .dropzone.is-dragover in CSS for the pulse/scale effect).
        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add("is-dragover");
            });
        });

        ["dragleave", "dragend"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove("is-dragover");
            });
        });

        dropzone.addEventListener("drop", (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragover");
            dropzone.classList.add("is-drop-flash");
            setTimeout(() => dropzone.classList.remove("is-drop-flash"), 300);
            if (event.dataTransfer?.files?.length) {
                handleFileSelection(event.dataTransfer.files);
            }
        });
    }

    // Clear selection.
    if (clearFilesBtn) {
        clearFilesBtn.addEventListener("click", clearFileSelection);
    }

    // Start analysis (upload, then analyze).
    if (uploadForm) {
        uploadForm.addEventListener("submit", handleStartAnalysis);
    }

    // Prevent interaction with disabled download links.
    [downloadJsonBtn, downloadCsvBtn].forEach((btn) => {
        if (!btn) return;
        btn.addEventListener("click", (event) => {
            if (btn.classList.contains("disabled")) {
                event.preventDefault();
            }
        });
    });

    /* ------------------------------------------------------------------
     * 9. Init
     * ------------------------------------------------------------------ */
    function init() {
        renderFileList();
        disableDownloads();
        hideAlerts();
        clearResultCards();
        setStep("upload");
    }

    document.addEventListener("DOMContentLoaded", init);
})();