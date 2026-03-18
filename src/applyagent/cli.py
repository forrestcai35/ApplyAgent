"""ApplyAgent CLI — the main entry point."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applyagent import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applyagent",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "dedup", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applyagent.config import load_env, ensure_dirs
    from applyagent.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applyagent[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyAgent — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applyagent.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'discover, dedup, enrich, score' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applyagent.pipeline import run_pipeline

    stage_list = stages if stages else ["discover", "dedup", "enrich", "score"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applyagent.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(0, "--limit", "-l", help="Max jobs to apply to (default: unlimited)."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel browser workers (each needs its own Chrome). Default: 1."),
    min_score: int = typer.Option(7, "--min-score", help="Only apply to jobs with fit_score >= N. Default: 7."),
    model: str = typer.Option("sonnet", "--model", "-m", help="Claude model for Claude Code mode: haiku, sonnet, opus. Ignored with --free."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs every 60s."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Navigate and fill forms but do NOT click Submit."),
    headless: bool = typer.Option(False, "--headless", help="Run Chrome headless (no visible window). Harder to debug."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to one specific job URL instead of pulling from the queue."),
    skip_tailor: bool = typer.Option(True, "--skip-tailor", help="Use base resume.pdf for jobs missing a tailored resume."),
    gen: bool = typer.Option(False, "--gen", help="Write the agent prompt to a file instead of running (requires --url). For debugging."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", metavar="URL", help="Manually mark a job URL as applied and exit."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", metavar="URL", help="Manually mark a job URL as failed and exit."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason string for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs so they can be retried."),
) -> None:
    """Submit job applications via an AI browser agent.
    \b
    MODES
      (default)   Claude Code — spawns the 'claude' CLI with Playwright MCP.
                  Requires: Claude Code CLI + Node.js + Chrome.

    \b
    COMMON EXAMPLES
      applyagent apply                      Apply to 1 job (Claude Code)
      applyagent apply -l 10               Apply to up to 10 jobs
      applyagent apply -c                  Run forever (continuous mode)
      applyagent apply --dry-run           Fill forms but don't submit
      applyagent apply --url <url>         Apply to one specific job
      applyagent apply --skip-tailor -l 5  Use base resume for next 5 jobs

    \b
    UTILITY (no browser started)
      --mark-applied <url>       Mark a job as applied in the database
      --mark-failed <url>        Mark a job as failed (add --fail-reason)
      --reset-failed             Reset all failed jobs for retry
      --gen --url <url>          Dump the agent prompt to a file for debugging
    """
    _bootstrap()

    from applyagent.config import check_tier, PROFILE_PATH as _profile_path
    from applyagent.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applyagent.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applyagent.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applyagent.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check requirements: Tier 3 (Claude Code)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applyagent init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # --skip-tailor: assign base resume to scored jobs missing a tailored resume
    if skip_tailor:
        from applyagent.config import RESUME_PDF_PATH
        if not RESUME_PDF_PATH.exists():
            console.print("[red]No base resume PDF found at[/red] ~/.applyagent/resume.pdf")
            raise typer.Exit(code=1)
        conn = get_connection()
        updated = conn.execute(
            "UPDATE jobs SET tailored_resume_path = ? "
            "WHERE fit_score >= ? AND tailored_resume_path IS NULL AND application_url IS NOT NULL",
            (str(RESUME_PDF_PATH), min_score),
        ).rowcount
        conn.commit()
        if updated:
            console.print(f"[green]--skip-tailor:[/green] {updated} job(s) set to use base resume")

    # Check 3: Jobs ready to apply (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No jobs ready to apply.[/red]\n"
                "Run [bold]applyagent run score tailor[/bold] first, or use [bold]--skip-tailor[/bold] to apply with your base resume."
            )
            raise typer.Exit(code=1)

    if gen:
        from applyagent.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applyagent.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else 0

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Mode:     Claude Code ({model})")
    console.print(f"  Limit:    {'unlimited' if effective_limit == 0 else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous or (effective_limit == 0),
        workers=workers,
    )


@app.command()
def edit(
    section: Optional[str] = typer.Argument(
        None,
        help="Which file to edit: profile, searches, env. Omit to choose interactively.",
    ),
) -> None:
    """Open a config file in your $EDITOR (or print its path).

    \b
    FILES
      profile    ~/.applyagent/profile.json   — personal info, salary, skills
      searches   ~/.applyagent/searches.yaml  — job search queries and locations
      env        ~/.applyagent/.env           — API keys and LLM settings
    """
    from applyagent.config import load_env, PROFILE_PATH, SEARCH_CONFIG_PATH, ENV_PATH
    load_env()

    files = {
        "profile":  ("profile.json",   PROFILE_PATH),
        "searches": ("searches.yaml",  SEARCH_CONFIG_PATH),
        "env":      (".env",           ENV_PATH),
    }

    choice = (section or "").strip().lower()
    if choice not in files:
        console.print("[bold]Which config file do you want to edit?[/bold]")
        for key, (name, path) in files.items():
            exists = "[green]exists[/green]" if path.exists() else "[dim]not created yet[/dim]"
            console.print(f"  [bold]{key:10}[/bold]  {path}  ({exists})")
        console.print()
        choice = typer.prompt("Enter name", default="profile")
        if choice not in files:
            console.print(f"[red]Unknown section:[/red] {choice!r}")
            raise typer.Exit(code=1)

    name, path = files[choice]
    if not path.exists():
        console.print(f"[yellow]{name} does not exist yet.[/yellow] Run [bold]applyagent init[/bold] first.")
        raise typer.Exit(code=1)

    import os, subprocess
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or ""
    if editor:
        subprocess.run([editor, str(path)])
    else:
        console.print(f"[bold]Path:[/bold] {path}")
        console.print("[dim]Set $EDITOR in your shell to open files automatically (e.g. export EDITOR=nano).[/dim]")


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applyagent.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyAgent Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("Duplicates filtered", str(stats.get("duplicates", 0)))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applyagent.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applyagent.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applyagent init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applyagent init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applyagent init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.applyagent/.env (run 'applyagent init')"))

    # --- Apply agent model (optional override) ---
    apply_url = os.environ.get("APPLY_LLM_URL", "")
    apply_model = os.environ.get("APPLY_LLM_MODEL", "")
    if apply_url:
        results.append(("Apply agent LLM", ok_mark,
                        f"{apply_url} ({apply_model or 'default model'})"))
    elif apply_model:
        results.append(("Apply agent LLM", ok_mark,
                        f"Same server, model: {apply_model}"))
    else:
        results.append(("Apply agent LLM", "[dim]not set[/dim]",
                        "Uses main LLM. Set APPLY_LLM_URL/APPLY_LLM_MODEL for a separate agent model"))

    # --- Auto-fallback ---
    if (has_gemini or has_openai) and has_local:
        results.append(("Auto-fallback", ok_mark,
                        f"Cloud primary → local fallback ({os.environ.get('LLM_URL')}) on rate limit"))
    elif has_gemini or has_openai:
        results.append(("Auto-fallback", "[dim]inactive[/dim]",
                        "Set LLM_URL too for automatic local fallback on rate limits"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", warn_mark,
                        "Install from https://claude.ai/code"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP — only needed for Claude Code mode)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", warn_mark,
                        "Needed for Claude Code mode."))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyAgent Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applyagent.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply via Claude Code (needs CLI + Node.js)[/dim]")

    console.print()


if __name__ == "__main__":
    app()
