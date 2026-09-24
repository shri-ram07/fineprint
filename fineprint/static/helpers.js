// Pure helpers used by app.js. They touch no DOM, so tests/js runs them under Node.

// Mirrors DocumentKind in fineprint/documents.py.
export const SUPPORTED_FILES = ["pdf", "docx", "txt", "md"];
// Mirrors MAX_BODY_BYTES in fineprint/web.py.
export const UPLOAD_LIMIT_TEXT = "10 MB";

/** The lower-cased extension of a supported file name, or null. */
export function fileKind(name) {
  const dot = name.lastIndexOf(".");
  if (dot < 0) return null;
  const kind = name.slice(dot + 1).toLowerCase();
  return SUPPORTED_FILES.includes(kind) ? kind : null;
}

/** The message to show for a failed response: the server's `detail` when it sent one. */
export function errorMessage(status, body) {
  let detail = null;
  try {
    detail = JSON.parse(body).detail;
  } catch {
    // Plain-text bodies, such as the server's own 413, fall through to the defaults.
  }
  if (typeof detail === "string" && detail) return detail;
  if (status === 413) return `That file is larger than the ${UPLOAD_LIMIT_TEXT} upload limit.`;
  if (status === 429) return "Too many requests right now. Please try again in a while.";
  return "Something went wrong. Please try again.";
}

/** Where `passage` sits in `text` as [start, end], or null if it isn't there. */
export function passageRange(text, passage) {
  if (!passage) return null;
  const start = text.indexOf(passage);
  return start < 0 ? null : [start, start + passage.length];
}

/** "high" -> "High": severity is shown as a word, never by colour alone. */
export function severityLabel(severity) {
  return severity ? severity[0].toUpperCase() + severity.slice(1) : "";
}
