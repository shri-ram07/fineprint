// The UI's XSS defence is that model output only ever reaches the page as text.
// This fails if anyone introduces an HTML-parsing sink into the frontend.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const sources = ["app.js", "helpers.js"].map((name) =>
  readFileSync(new URL(`../../fineprint/static/${name}`, import.meta.url), "utf8"),
);

test("no HTML-parsing sinks in the frontend", () => {
  for (const source of sources) {
    assert.doesNotMatch(source, /innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(/);
  }
});

test("model output is inserted with textContent", () => {
  assert.match(sources[0], /node\.textContent = text/);
});
