# CLI harness for AI stages

ApplyAgent can use a logged-in agent CLI instead of configuring an LLM API key.
This follows Sophic Assistant's spawner pattern: bounded subprocess turns,
explicit context, structured results, isolated working directory, and no automatic
provider failover. The existing `get_client().chat()` interface routes enrichment,
scoring, resume tailoring, cover letters, and their judges through the harness.

Set in your private `APPLYAGENT_DIR/.env`:

```dotenv
APPLYAGENT_LLM_BACKEND=codex
APPLYAGENT_HARNESS_TIMEOUT=180
# Optional; omit to use the CLI default model:
# APPLYAGENT_HARNESS_MODEL=your-model
```

Supported backends: `codex`, `claude`, `api`, `auto`. Auto preserves configured API
providers and otherwise selects Claude. Explicit selection never falls back to an
API or another engine. Authenticate the selected CLI separately. CLI presence
passes dependency checks; it does not prove account access or remaining usage.
Optional executable overrides: `APPLYAGENT_CODEX_BIN`, `APPLYAGENT_CLAUDE_BIN`.

Claude runs with no built-in tools, no MCP servers, and no saved session. Codex
uses `exec`, ephemeral sessions, read-only sandboxing, no shell tool, disabled
apps/web search/multi-agent, and ignored user config. Each task runs in a temporary
directory so project instructions are not loaded. Prompts go through stdin;
results return to the existing parsers and validators. A timeout kills and
reaps the process group. Empty, malformed, and unsuccessful results fail the turn.
Temperature and token-limit arguments are retained for interface compatibility but
are not enforced as CLI sampling controls. Calls require online provider access,
send supplied task data to that provider, and consume the selected account's usage.

```bash
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent run score
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent run tailor cover pdf
```

The browser application launcher supports both Codex and Claude:

```bash
APPLYAGENT_DIR="$PWD/user_data" .venv/bin/applyagent apply --backend codex --limit 10
```

`--backend auto` uses `APPLYAGENT_APPLY_BACKEND` when set, otherwise follows the
Codex/Claude text backend. API text configurations prefer an installed Codex CLI,
then Claude. Explicit selection never falls back. `--model` is optional: Codex
uses its default and Claude uses Sonnet. Both need Node/npx and a Chromium browser;
Chrome, Chromium, and macOS Brave are detected. CLI presence does not prove login.
Codex uses its own authenticated CLI session, an isolated browser via Playwright
MCP, no shell tools, and automatic approval review in a worker workspace. Approval requirements are not bypassed;
operations needing unavailable approval stop. The app's connected browser tools
are a separate execution path, not inherited by this CLI subprocess.

Codex jobs have a 600-second deadline (`APPLYAGENT_APPLY_TIMEOUT` override). Only
the final message is parsed; success requires a confirmation URL and message.
Timeouts, crashes, and ambiguous outcomes are held as `submission_uncertain`,
without automatic retry. Review their logs before any explicit retry. Dry runs
never count as submissions. Both runners filter the saved application registry
before selecting jobs. Text-generation and unit tests are not submission evidence.

CLI configuration follows [noninteractive Codex documentation](https://developers.openai.com/codex/noninteractive)
and [MCP configuration](https://developers.openai.com/codex/mcp).

Reference: [OpenAI's noninteractive Codex execution example](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).

## Application answer completion

The browser prompt classifies questions as confirmed facts, generated prose,
unambiguous derived facts, optional unknowns, or required unknowns. Motivation,
experience summaries and required cover-letter prose are written from resume
evidence without requesting prewritten answers. Optional unknowns are omitted or
declined. Expected degrees remain expected; date precision, relocation boundaries
and actual employment titles are preserved. Required unknown factual fields remain
blocked, with the exact field and missing evidence recorded.

Known answers rejected by form widgets get bounded repairs using fresh page state,
visible options, typing and blur. Explicit validation failures can be retried; an
uncertain post-submit outcome cannot. These changes do not grant destination
authorization, alter tool approvals or guarantee that an employer accepts a form.
