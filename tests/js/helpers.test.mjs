// Run with: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  errorMessage,
  fileKind,
  passageRange,
  severityLabel,
} from "../../fineprint/static/helpers.js";

test("fileKind accepts supported extensions in any case", () => {
  assert.equal(fileKind("Lease.PDF"), "pdf");
  assert.equal(fileKind("offer.v2.docx"), "docx");
  assert.equal(fileKind("notes.md"), "md");
});

test("fileKind rejects unsupported or missing extensions", () => {
  assert.equal(fileKind("contract.doc"), null);
  assert.equal(fileKind("README"), null);
});

test("errorMessage prefers the server's detail", () => {
  assert.equal(errorMessage(400, JSON.stringify({ detail: "The file is empty." })), "The file is empty.");
});

test("errorMessage falls back by status for plain-text or empty bodies", () => {
  assert.match(errorMessage(413, "Content Too Large"), /10 MB/);
  assert.match(errorMessage(429, ""), /Too many requests/);
  assert.match(errorMessage(500, "Internal Server Error"), /Something went wrong/);
  assert.match(errorMessage(400, JSON.stringify({ detail: "" })), /Something went wrong/);
});

test("passageRange locates a verified quote for selection", () => {
  const text = "1. Rent. The Tenant shall pay £900 per month.";
  assert.deepEqual(passageRange(text, "shall pay £900"), [20, 34]);
  assert.equal(text.slice(...passageRange(text, "shall pay £900")), "shall pay £900");
});

test("passageRange returns null for missing or blank passages", () => {
  assert.equal(passageRange("Rent is due.", "deposit"), null);
  assert.equal(passageRange("Rent is due.", ""), null);
});

test("severityLabel spells severity out", () => {
  assert.equal(severityLabel("high"), "High");
  assert.equal(severityLabel(""), "");
});
