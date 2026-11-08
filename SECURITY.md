# Security

IndicSwitch is a research pipeline plus a demo app. It has **no user accounts, no login, no admin area and no database**, and it stores nothing that visitors upload. This file records the security review carried out on 2026-11-08 and how to keep the project safe.

## Checklist

| # | Item | Status |
|---|---|---|
| 1 | Hide API keys | ✅ None in the code. The project needs no keys. For gated Hugging Face datasets, use the `HF_TOKEN` environment variable or Colab **Secrets**, never a literal in code or notebooks. |
| 2 | Passwords off GitHub | ✅ No credentials anywhere. `.gitignore` blocks `.env*`, `*.key`, `*.pem`, `.streamlit/secrets.toml` and `hf_token*`. No commits exist yet, so there is no history to clean. |
| 3 | Lock the admin page | N/A: there is no admin page. Streamlit's developer menu (rerun, clear cache, deploy) is hidden from visitors with `toolbarMode = "viewer"`. |
| 4 | Proper login | N/A: public demo with no accounts. If you add one, use Streamlit's built-in OIDC auth (`st.login`), not a homemade login. |
| 5 | Users see only their data | ✅ Results live only in each visitor's own session (`st.session_state`). The shared cache holds only models. Uploaded audio goes to a private temp file (`mkstemp`, owner-only) that is **deleted right after processing**. |
| 6 | Check every form | ✅ Uploads: extension allow-list, 25 MB limit (server and app), 0.1 s–10 min duration, decode errors handled. Text fields: length caps (5,000 characters; protected words 300 characters / 50 words), empty input rejected. Dropdowns and sliders only offer fixed or bounded values, and the Whisper model is re-checked on the server. Output: transcripts are Markdown-escaped and tag chips are HTML-escaped (tested against `<script>` input). |
| 7 | Stop login spam | N/A as login. The equivalent abuse risk, flooding the server with heavy jobs, is limited by a 5 s per-session cooldown and at most 2 concurrent pipeline runs across all visitors. |
| 8 | HTTPS everywhere | ✅ All outbound traffic (Hugging Face Hub, GitHub, dataset downloads) uses HTTPS. The dev server only listens on `localhost`. Deploy behind HTTPS (see below). |
| 9 | Hash passwords | N/A: no passwords are stored. Model files *are* integrity-checked: `ngram_lid.pkl` is loaded only if its SHA-256 matches the `.sha256` file written at save time. |
| 10 | Lock the database | N/A: there is no database. Training data is read-only public datasets. |
| 11 | Daily backups | N/A: no user data or database. Code lives in Git. Data and results can be regenerated with the scripts. Back up trained checkpoints (`models/`) to Google Drive, as the notebooks show. |
| 12 | Remove debug mode | ✅ No debug flags or breakpoints. `runOnSave = false`, `fileWatcherType = "none"`, `headless = true`, usage statistics off. |
| 13 | Remove error details | ✅ `showErrorDetails = "none"`; the legacy value `false` would still show stack traces. The app catches errors, logs them on the server, and shows only a generic message (tested). |
| 14 | Update packages | ✅ `pip-audit` reports **no known vulnerabilities**. `pip` and `setuptools` were upgraded (they had 17 advisories). `requirements.txt` sets minimum versions to the audited ones; `torch>=2.6` fixes CVE-2025-32434. |
| 15 | Remove unused code | ✅ Dead functions and imports removed (`ruff` F-rules and `vulture` clean, apart from false positives such as dataclass fields and public entry points). |
| 16 | Full security check | ✅ `ruff` security rules (bandit `S`) plus bug rules (`B`): all pass. CSRF/XSRF and CORS protection on. Pickle loading is integrity-checked and never takes user-supplied paths. No `trust_remote_code`, `eval`, `exec` or shell calls. 51 automated tests, including app security behaviour. |

## Deploying the demo safely
* **Streamlit Community Cloud / Hugging Face Spaces**: HTTPS is provided automatically.
* **Your own server**: put it behind a TLS reverse proxy (Caddy, or nginx with Let's Encrypt), run `streamlit run app/streamlit_app.py --server.address 0.0.0.0` *behind* the proxy, and never expose the plain-HTTP port publicly.
* Only load model files you produced yourself. Unpickling an untrusted file can run code, which is why the checksum check exists.

## Re-running the checks
```bash
python -m pip_audit                                                     # known vulnerabilities
python -m ruff check --select F,E9,B,S --ignore S101,S311 src scripts app tests
python -m pytest -q
```

## Data licences
COMI-LINGUA is CC-BY-4.0. CoSHE-500 is **CC-BY-NC-4.0**: non-commercial use only, with attribution. Audio and processed data are never committed.
