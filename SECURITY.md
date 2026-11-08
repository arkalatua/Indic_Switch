# Security

IndicSwitch is a research pipeline plus a web app (`server/` + `web/`). It has **no user accounts, no login, no admin area and no database**, and it keeps nothing that visitors upload. This file records the security review (last run 2026-11-08) and how to keep the project safe.

## Checklist

| # | Item | Status |
|---|---|---|
| 1 | Hide API keys | ✅ None in the code. The project needs no keys. For gated Hugging Face datasets, use the `HF_TOKEN` environment variable or Colab **Secrets**, never a literal in code or notebooks. |
| 2 | Passwords off GitHub | ✅ No credentials anywhere in the code or the Git history. `.gitignore` blocks `.env*`, `*.key`, `*.pem`, `*.p12`, `.huggingface/` and `hf_token*`. |
| 3 | Lock the admin page | N/A: there is no admin page. The API docs (`/docs`, `/redoc`, `/openapi.json`) are disabled, and only `web/` is served as static files. |
| 4 | Proper login | N/A: no accounts. If accounts are ever added, use an established OIDC provider rather than a homemade login. |
| 5 | Users see only their data | ✅ Nothing is stored and there are no sessions on the server. Each request returns its result only to the browser that sent it, with `Cache-Control: no-store`. |
| 6 | Check every form | ✅ All input is validated on the server (details below). The browser also checks file type, size and length, but the server never relies on that. |
| 7 | Stop login spam | N/A as login. The equivalent risk, flooding the server with heavy jobs, is limited by per-IP rate limits and a cap on concurrent jobs. |
| 8 | HTTPS everywhere | ✅ All outbound traffic (model and dataset downloads, GitHub) uses HTTPS. The server listens only on `127.0.0.1` by default; deploy it behind HTTPS (see below). HSTS is sent with `INDICSWITCH_HTTPS=1`. |
| 9 | Hash passwords | N/A: no passwords are stored. Model files *are* integrity-checked: `ngram_lid.pkl` is loaded only if its SHA-256 matches the `.sha256` file written at save time. |
| 10 | Lock the database | N/A: there is no database. Training data is read-only public datasets. |
| 11 | Daily backups | N/A: no user data or database. Code lives in Git. Data and results can be regenerated with the scripts. Back up trained checkpoints (`models/`) to Google Drive, as the notebooks show. |
| 12 | Remove debug mode | ✅ No debug flags, reload mode or breakpoints. The interactive API docs are off. |
| 13 | Remove error details | ✅ Unhandled errors return one generic message, and details go to the server log only (tested). Validation errors name the field but never echo the submitted value. |
| 14 | Update packages | ✅ `pip-audit` reports **no known vulnerabilities**. `requirements.txt` sets minimum versions to the audited ones; `torch>=2.6` fixes CVE-2025-32434. |
| 15 | Remove unused code | ✅ The superseded Streamlit demo and dead functions were removed. `ruff` (F rules) and `vulture` are clean, apart from false positives such as dataclass fields and public entry points. |
| 16 | Full security check | ✅ `ruff` security rules (bandit `S`) plus bug rules (`B`) all pass. No `trust_remote_code`, `eval`, `exec` or shell calls. Pickle loading is integrity-checked and never takes a path from a request. The API's security behaviour is covered by `tests/test_server.py`, and the page was checked in headless Chrome. |

## Web app details

* **Input:** text up to 5,000 characters and never blank; settings must match fixed lists (model, language, script, tagger id); the confidence threshold must be in range; at most 50 protected words of 40 characters or fewer. Audio must be a real WAV: the `RIFF`/`WAVE` header is checked and libsndfile parses it, with 8–96 kHz, 1–8 channels, 0.1 s to 10 minutes, and 25 MB at most. The browser converts any format to 16 kHz mono WAV first, so the server parses nothing else.
* **No path or model name from the client:** language models are chosen by id (`ngram`, `xlmr`, `rule`) and Whisper by `base`/`small`/`tiny`. The server maps those ids to files and models itself.
* **Abuse:** per-IP rate limits (text 30/min; audio, streamed or not, 6/min; score 60/min; all with `Retry-After`), at most 2 concurrent heavy jobs (503 when busy), and body-size limits before parsing (413). A `Content-Length` header is required for POSTs (411). The streaming route releases its job slot when it finishes, fails, or the client disconnects (tested).
* **Headers on every response:** a strict Content-Security-Policy (`default-src 'self'`, no inline script or style, `frame-ancestors 'none'`), `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cross-Origin-Opener-Policy`, and a Permissions-Policy allowing only the microphone, same origin only. API responses carry `Cache-Control: no-store`.
* **No XSS:** the frontend inserts all server and user text with `textContent`. A probe with `<img onerror>` and `<b>` input rendered as plain text, ran no script and created no elements.
* **Static files:** only `web/` is served. There is no directory listing, and path traversal returns 404. The taste-skill working folder is kept outside `web/`.
* **Privacy:** fonts and the UI kit are self-hosted, so pages make no third-party requests and visitors' IPs never reach Google. Uploaded audio is held in memory for one request; the framework spools uploads over 1 MB to a temporary file that is deleted when the request ends. Nothing is kept. Microphone audio stays in the page until the user presses Transcribe.

## Deploying safely
* Run the app behind a TLS-terminating reverse proxy (Caddy, or nginx with Let's Encrypt), start it with `--host 127.0.0.1` behind the proxy, set `INDICSWITCH_HTTPS=1`, and set `INDICSWITCH_TRUST_PROXY=1` only when that proxy sets `X-Forwarded-For`. Never expose the plain-HTTP port publicly.
* Only load model files you produced yourself. Unpickling an untrusted file can run code, which is why the checksum check exists.

## Re-running the checks
```bash
python -m pip_audit                                                     # known vulnerabilities
python -m ruff check --select F,E9,B,S --ignore S101,S311 src scripts server tests
python -m pytest -q
```

## Data licences
COMI-LINGUA is CC-BY-4.0. CoSHE-500 is **CC-BY-NC-4.0**: non-commercial use only, with attribution. Audio and processed data are never committed. Fonts are under the SIL OFL, and the vendored UI kit (`web/assets/kit/`) is MIT.
