<!-- logo here -->

# ApplyAgent

[![PyPI version](https://img.shields.io/pypi/v/ApplyAgent?color=blue)](https://pypi.org/project/ApplyAgent/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)


---

## What It Does

ApplyAgent is a 6-stage autonomous job application pipeline. It discovers jobs across 5+ boards, scores them against your resume with AI, tailors your resume per job, writes cover letters, and **submits applications for you**. It navigates forms, uploads documents, answers screening questions, all hands-free.

Three commands. That's it.

```bash
pip install applyagent
pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex
applyagent init          # one-time setup: resume, profile, preferences, API keys
applyagent doctor        # verify your setup — shows what's installed and what's missing
applyagent run           # discover > enrich > score > tailor > cover letters
applyagent run -w 4      # same but parallel (4 threads for discovery/enrichment)
applyagent apply         # autonomous browser-driven submission
applyagent apply -w 3    # parallel apply (3 Chrome instances)
applyagent apply --dry-run  # fill forms without submitting
```

> **Why two install commands?** `python-jobspy` pins an exact numpy version in its metadata that conflicts with pip's resolver, but works fine at runtime with any modern numpy. The `--no-deps` flag bypasses the resolver; the second command installs jobspy's actual runtime dependencies. Everything except `python-jobspy` installs normally.

---

## Three Paths

### Full Pipeline — Claude Code (recommended for best results)
**Requires:** Python 3.11+, Node.js (for npx), Gemini API key (free), Claude Code CLI, Chrome

Runs all 6 stages, from job discovery to autonomous application submission. This is the full power of ApplyAgent.

### Discovery, Scoring, and Tailoring — Local Model (completely free)
**Requires:** Python 3.11+, Chrome, local LLM server (Ollama/llama.cpp)

Runs stages 1-5 without any API keys or subscriptions. Uses your local LLM to score jobs, tailor your resume, and write cover letters.

```bash
# Start Ollama with a capable model
ollama serve
ollama pull qwen2.5:32b    # or llama3.1, mistral, etc.

# Configure ApplyAgent for local
# In ~/.applyagent/.env:
#   LLM_URL=http://localhost:11434/v1
#   LLM_MODEL=qwen2.5:32b

applyagent run   # uses local LLM for scoring and tailoring
```

To run auto-apply (stage 6), you will still need to use Claude Code.

### Discovery + Tailoring Only
**Requires:** Python 3.11+, Gemini API key (free)

Runs stages 1-5: discovers jobs, scores them, tailors your resume, generates cover letters. You submit applications manually with the AI-prepared materials.

---

## The Pipeline

| Stage | What Happens |
|-------|-------------|
| **1. Discover** | Scrapes 5 job boards (Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs) + 48 Workday employer portals + 30 direct career sites |
| **2. Enrich** | Fetches full job descriptions via JSON-LD, CSS selectors, or AI-powered extraction |
| **3. Score** | AI rates every job 1-10 based on your resume and preferences. Only high-fit jobs proceed |
| **4. Tailor** | AI rewrites your resume per job: reorganizes, emphasizes relevant experience, adds keywords. Never fabricates |
| **5. Cover Letter** | AI generates a targeted cover letter per job |
| **6. Auto-Apply** | Claude Code navigates application forms, fills fields, uploads documents, answers questions, and submits |

Each stage is independent. Run them all or pick what you need.


## Requirements

| Component | Required For | Details |
|-----------|-------------|---------|
| Python 3.11+ | Everything | Core runtime |
| Node.js 18+ | Auto-apply (Claude Code mode) | Needed for `npx` to run Playwright MCP server |
| Gemini API key | Scoring, tailoring, cover letters | Free tier (15 RPM / 1M tokens/day) is enough |
| Chrome/Chromium | Auto-apply | Auto-detected on most systems |
| Claude Code CLI | Auto-apply (Claude Code mode) | Install from [claude.ai/code](https://claude.ai/code) |
| Local LLM server | Auto-apply (local mode) | Ollama, llama.cpp, vLLM, or any OpenAI-compatible server |

**Gemini API key is free.** Get one at [aistudio.google.com](https://aistudio.google.com). OpenAI and local models (Ollama/llama.cpp) are also supported.

**Local auto-apply is completely free.** Use `applyagent apply --local` with any local model. No API keys needed for auto-apply — just Chrome + a local LLM server.

### Optional

| Component | What It Does |
|-----------|-------------|
| CapSolver API key | Solves CAPTCHAs during auto-apply (hCaptcha, reCAPTCHA, Turnstile, FunCaptcha). Without it, CAPTCHA-blocked applications just fail gracefully |

> **Note:** python-jobspy is installed separately with `--no-deps` because it pins an exact numpy version in its metadata that conflicts with pip's resolver. It works fine with modern numpy at runtime.

---

## Configuration

All generated by `applyagent init`:

### `profile.json`
Your personal data in one structured file: contact info, work authorization, compensation, experience, skills, resume facts (preserved during tailoring), and EEO defaults. Powers scoring, tailoring, and form auto-fill.

### `searches.yaml`
Job search queries, target titles, locations, boards. Run multiple searches with different parameters.

### `.env`
API keys and runtime config: `GEMINI_API_KEY`, `LLM_MODEL`, `CAPSOLVER_API_KEY` (optional). For local model scoring and tailoring, specify `LLM_URL` and `LLM_MODEL`.

### Package configs (shipped with ApplyAgent)
- `config/employers.yaml` - Workday employer registry (48 preconfigured)
- `config/sites.yaml` - Direct career sites (30+), blocked sites, base URLs, manual ATS domains
- `config/searches.example.yaml` - Example search configuration

---

## How Stages Work

### Discover
Queries Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs via JobSpy. Scrapes 48 Workday employer portals (configurable in `employers.yaml`). Hits 30 direct career sites with custom extractors. Deduplicates by URL.

### Enrich
Visits each job URL and extracts the full description. 3-tier cascade: JSON-LD structured data, then CSS selector patterns, then AI-powered extraction for unknown layouts.

### Score
AI scores every job 1-10 against your profile. 9-10 = strong match, 7-8 = good, 5-6 = moderate, 1-4 = skip. Only jobs above your threshold proceed to tailoring.

### Tailor
Generates a custom resume per job: reorders experience, emphasizes relevant skills, incorporates keywords from the job description. Your `resume_facts` (companies, projects, metrics) are preserved exactly. The AI reorganizes but never fabricates.

### Cover Letter
Writes a targeted cover letter per job referencing the specific company, role, and how your experience maps to their requirements.

### Auto-Apply
An AI agent launches a Chrome instance, navigates to each application page, detects the form type, fills personal information and work history, uploads the tailored resume and cover letter, answers screening questions with AI, and submits. A live dashboard shows progress in real-time.

It spawns the `claude` CLI with Playwright MCP. Best results but requires an Anthropic subscription + Node.js.

```bash
applyagent apply                       # Claude Code mode (default)
applyagent apply --dry-run             # Fill forms without submitting
applyagent apply -w 2                  # 2 parallel workers

# Utility modes (no Chrome/agent needed)
applyagent apply --mark-applied URL    # manually mark a job as applied
applyagent apply --mark-failed URL     # manually mark a job as failed
applyagent apply --reset-failed        # reset all failed jobs for retry
applyagent apply --gen --url URL       # generate prompt file for manual debugging
```

---

## CLI Reference

```
applyagent init                         # First-time setup wizard
applyagent doctor                       # Verify setup, diagnose missing requirements
applyagent run [stages...]              # Run pipeline stages (or 'all')
applyagent run --workers 4              # Parallel discovery/enrichment
applyagent run --stream                 # Concurrent stages (streaming mode)
applyagent run --min-score 8            # Override score threshold
applyagent run --dry-run                # Preview without executing
applyagent run --validation lenient     # Relax validation (recommended for Gemini free tier)
applyagent run --validation strict      # Strictest validation (retries on any banned word)
applyagent apply                        # Launch auto-apply (Claude Code)
applyagent apply --workers 3            # Parallel browser workers
applyagent apply --dry-run              # Fill forms without submitting
applyagent apply --continuous           # Run forever, polling for new jobs
applyagent apply --headless             # Headless browser mode
applyagent apply --url URL              # Apply to a specific job
applyagent dashboard                    # Open HTML results dashboard
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
