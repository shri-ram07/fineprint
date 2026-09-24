"use strict";

// All model output is inserted with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const statusLine = $("status");
const errorLine = $("error");

const SUPPORTED_FILES = ["pdf", "docx", "txt", "md"];
const HEADINGS = { analyze: "What this document says", compare: "How the two documents differ" };
const SUPPORT = {
  yes: "Answered by the document",
  partly: "Partly answered by the document",
  no: "Not addressed in the document",
};

let twoDocuments = false; // label quotes with their document only when there are two

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// One request at a time: uploading while an analysis runs would change the text under it.
function setBusy(message) {
  $("submit").disabled = Boolean(message);
  for (const input of document.querySelectorAll("input[type=file]")) input.disabled = Boolean(message);
  $("progress").hidden = !message;
  if (message) {
    statusLine.textContent = message;
    errorLine.textContent = "";
  }
}

function showError(message) {
  statusLine.textContent = "";
  errorLine.textContent = message;
}

async function request(url, options) {
  const response = await fetch(url, options).catch(() => {
    throw new Error("Couldn't reach FinePrint. Check that the app is still running, then try again.");
  });
  const body = await response.text();
  let data = null;
  try {
    data = JSON.parse(body);
  } catch {
    // Non-JSON bodies (e.g. the server's plain-text 413) are handled below.
  }
  if (!response.ok) {
    const fallback = response.status === 413
      ? "That file is larger than the 10 MB upload limit."
      : "Something went wrong. Please try again.";
    throw new Error((data && data.detail) || fallback);
  }
  return data;
}

// ---- Uploads: extracted text goes into the visible textarea so the user sees what the AI sees.

for (const input of document.querySelectorAll("input[type=file]")) {
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    const kind = file.name.split(".").pop().toLowerCase();
    if (!SUPPORTED_FILES.includes(kind)) {
      showError("Only PDF, DOCX, TXT and MD files are supported. For other formats, paste the text.");
      input.value = "";
      return;
    }
    setBusy(`Reading ${file.name}…`);
    try {
      const { text } = await request(`/api/documents/text?kind=${kind}`, { method: "POST", body: file });
      $(input.dataset.target).value = text;
      statusLine.textContent =
        `Extracted ${text.length.toLocaleString()} characters from ${file.name}. Check the text before continuing.`;
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(null);
      input.value = ""; // allow choosing the same file again
    }
  });
}

// ---- Submitting

$("assist-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  // Document A is always sent (the server rejects it if blank) so quote labels match the textareas.
  const second = $("document-b").value;
  const documents = second.trim() ? [$("document-a").value, second] : [$("document-a").value];
  twoDocuments = documents.length === 2;
  setBusy("Reading your document. Long documents can take a few minutes.");
  try {
    const response = await request("/api/assist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documents, question, context: $("context").value }),
    });
    render(response, question);
    statusLine.textContent = "Done.";
  } catch (error) {
    showError(error.message); // inputs are left untouched so nothing is lost
  } finally {
    setBusy(null);
  }
});

function render(response, question) {
  $("results").hidden = false;
  $("disclaimer").textContent = response.disclaimer;
  if (response.task === "ask") {
    renderAnswer(response, question);
    return;
  }
  // A new analysis replaces the old one, and earlier answers no longer describe what is shown.
  for (const answer of $("answers").querySelectorAll("article")) answer.remove();
  $("answers").hidden = true;

  const result = response.result;
  $("analysis-heading").textContent = HEADINGS[response.task];
  $("analysis-body").replaceChildren(
    el("p", `Assessed for: ${result.perspective}. If that isn't you, describe your situation above and run it again.`, "perspective"),
    renderWarnings(response.warnings),
    ...(response.task === "analyze" ? renderAnalysis(result) : renderComparison(result)),
  );
  $("analysis").hidden = false;
  $("analysis-heading").focus();
}

function renderAnalysis(result) {
  return [
    section("Summary", [el("p", `${result.document_type}. ${result.summary}`)]),
    section("Risks and unusual clauses", result.risks.map(renderRisk)),
    section("Key terms", result.key_terms.map((term) => finding(`${term.label}: ${term.value}`, [term.quote]))),
    section("Obligations", result.obligations.map((duty) => finding(`${duty.party}: ${duty.description}`, [duty.quote]))),
    listSection("Missing or unclear", result.missing_or_unclear),
    listSection("Questions to ask a lawyer", result.questions_for_lawyer),
    listSection("Next steps", result.next_steps, "ol"),
    listSection("Parties", result.parties),
  ];
}

function renderComparison(result) {
  return [
    section("Summary", [el("p", result.summary)]),
    section("Differences", result.differences.map(renderDifference)),
    listSection("Questions to ask a lawyer", result.questions_for_lawyer),
  ];
}

function renderAnswer(response, question) {
  const result = response.result;
  const article = el("article", null, "answer");
  const heading = el("h3", question);
  heading.tabIndex = -1;
  const support = el("p", SUPPORT[result.supported_by_document], `badge support-${result.supported_by_document}`);
  article.append(heading, renderWarnings(response.warnings), support, el("p", result.answer));
  if (result.quotes.length) article.append(renderQuotes(result.quotes));
  if (result.caveats.length) article.append(listSection("Caveats", result.caveats, "ul", "h4"));

  $("answers").hidden = false;
  $("answers").append(article);
  heading.focus();
  $("question").value = "";
}

// ---- Building blocks

function section(title, children, level = "h3") {
  const node = el("section");
  node.append(el(level, title));
  if (children.length) node.append(...children);
  else node.append(el("p", "Nothing flagged here.", "empty"));
  return node;
}

function listSection(title, items, listTag = "ul", level = "h3") {
  if (!items.length) return section(title, [], level);
  const list = el(listTag);
  list.append(...items.map((item) => el("li", item)));
  return section(title, [list], level);
}

function renderWarnings(warnings) {
  if (!warnings.length) return document.createDocumentFragment();
  const list = el("ul", null, "warnings");
  list.append(...warnings.map((warning) => el("li", warning)));
  return list;
}

function severityHeading(severity, title) {
  const heading = el("h4");
  heading.append(el("span", `${severity[0].toUpperCase()}${severity.slice(1)}`, "badge"), title);
  return heading;
}

function renderRisk(risk) {
  const article = el("article", null, `finding severity-${risk.severity}`);
  article.append(
    severityHeading(risk.severity, risk.title),
    el("p", risk.plain_language),
    el("p", `Why it matters: ${risk.why_it_matters}`),
    renderQuotes(risk.quotes),
  );
  return article;
}

function renderDifference(difference) {
  const article = el("article", null, `finding severity-${difference.severity}`);
  const sides = el("dl");
  sides.append(
    el("dt", "Document A"), el("dd", difference.document_a),
    el("dt", "Document B"), el("dd", difference.document_b),
  );
  article.append(
    severityHeading(difference.severity, difference.topic),
    sides,
    el("p", `What it means for you: ${difference.impact}`),
    renderQuotes(difference.quotes),
  );
  return article;
}

function finding(text, quotes) {
  const article = el("article", null, "finding");
  article.append(el("p", text), renderQuotes(quotes));
  return article;
}

function renderQuotes(quotes) {
  const box = el("div", null, "quotes");
  if (!quotes.some((quote) => quote.text)) {
    box.append(el("p", "No quote could be matched to your document for this point. Treat it with caution.", "unmatched"));
    return box;
  }
  for (const quote of quotes) {
    if (!quote.text) {
      box.append(el("p", "One quoted passage couldn't be matched to your document.", "unmatched"));
      continue;
    }
    const block = el("blockquote");
    if (twoDocuments) block.append(el("span", `Document ${quote.document}`, "source"));
    const show = el("button", "Show in document", "link no-print");
    show.type = "button";
    show.addEventListener("click", () => showInDocument(quote));
    block.append(el("p", quote.text), show);
    box.append(block);
  }
  return box;
}

function showInDocument(quote) {
  if (quote.document === "B") $("second-document").open = true;
  const textarea = $(quote.document === "B" ? "document-b" : "document-a");
  const start = textarea.value.indexOf(quote.text);
  if (start < 0) {
    statusLine.textContent = "Couldn't find this passage in the document text. It may have changed since these results.";
    return;
  }
  textarea.focus();
  textarea.setSelectionRange(start, start + quote.text.length);
}

// ---- Copying: the visible results as plain text, without buttons.

$("copy").addEventListener("click", async () => {
  document.body.classList.add("copying");
  const text = $("results").innerText;
  document.body.classList.remove("copying");
  try {
    await navigator.clipboard.writeText(text);
    statusLine.textContent = "Copied.";
  } catch {
    statusLine.textContent = "Couldn't copy. Select the text and copy it yourself.";
  }
});
