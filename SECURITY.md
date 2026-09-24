# Security

## Reporting a vulnerability

Please report security issues privately through
[GitHub security advisories](https://github.com/shri-ram07/fineprint/security/advisories/new)
rather than in a public issue. You can expect an acknowledgement within a few days.

## Threat model

| Threat | Controls |
|---|---|
| Malicious uploads (zip bombs, XML entity attacks, huge or malformed PDFs) | 10 MB body limit enforced before parsing; bounded DOCX decompression; `defusedxml`; per-page character limit for PDFs; any parser failure becomes a fixed 400 |
| Prompt injection inside a document | Documents are tagged and cannot close their own tag; the system prompt treats them as material, not instructions; output must match a fixed schema; every quote is verified against the source in code |
| Untrusted model output | Finish reason checked before parsing; schema validation; rendered with `textContent` only, never as HTML; strict CSP |
| Abuse of the public demo's Gemini quota | Per-client and total hourly rate limits; client address taken from the proxy-appended end of `X-Forwarded-For` so it cannot be forged; single instance; 120 s timeout on model calls |
| Cross-site use of the API | Cross-origin POSTs refused (`Origin` check); `Host` allow-list against DNS rebinding; `frame-ancestors 'none'` |
| Secret leakage | Key read from the environment (Secret Manager in production), never logged; `.env` excluded from git, uploads and images; logs hold sizes and timings, never document text |
| Vulnerable dependencies | Locked dependencies (`uv.lock`), digest-pinned base image, SHA-pinned CI actions, `pip-audit` on every push and weekly on a schedule |

## Known limits

The public demo has no login, by design, so that evaluators can use it; the rate limits
bound what an anonymous visitor can spend. The limits are in memory, so they hold for the
single deployed instance. A multi-instance deployment should move them to a shared store.
