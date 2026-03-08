"""Local agent loop for auto-apply.

Replaces Claude Code by running a local LLM in an iterative tool-calling
loop. The agent receives the application prompt, uses browser tools to
interact with Chrome, and returns a RESULT: status line when done.
"""

import json
import logging
import re
import threading
import time

import httpx

from applyagent.apply.browser import BrowserTools
from applyagent.llm import LLMClient

logger = logging.getLogger(__name__)

# Max turns before giving up
MAX_TURNS = 80
# Max conversation history entries to keep (avoid blowing context window)
MAX_HISTORY = 60
# Max characters per tool result
MAX_TOOL_RESULT_LEN = 12000

# Result codes the agent can emit
RESULT_PATTERNS = re.compile(
    r"RESULT:(APPLIED|EXPIRED|CAPTCHA|LOGIN_ISSUE|FAILED(?::\S+)?)"
)

# -------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# -------------------------------------------------------------------

TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate to a URL. Use this to open the job application page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": (
                "Get a text snapshot of the current page showing its structure "
                "and interactive elements with [ref=N] tags. Use refs to click/fill elements."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an interactive element by its ref number from the snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element ref number from snapshot"},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill",
            "description": "Clear a form field and type a new value. Use ref from snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element ref number"},
                    "value": {"type": "string", "description": "Text to fill in"},
                },
                "required": ["ref", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill_form",
            "description": "Fill multiple form fields at once. More efficient than individual fills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["ref", "value"],
                        },
                        "description": "List of {ref, value} pairs to fill",
                    },
                },
                "required": ["fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_select_option",
            "description": "Select an option in a dropdown/combobox by ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element ref number"},
                    "value": {"type": "string", "description": "Option value or label to select"},
                },
                "required": ["ref", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text using the keyboard (appends to currently focused element).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "submit": {"type": "boolean", "description": "Press Enter after typing"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press_key",
            "description": "Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to press"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_file_upload",
            "description": "Upload file(s) to a file input element by ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "File input element ref"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File path(s) to upload",
                    },
                },
                "required": ["ref", "paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_evaluate",
            "description": "Run JavaScript code in the browser page. Returns the result as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "description": "JavaScript function body to evaluate",
                    },
                },
                "required": ["function"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_take_screenshot",
            "description": "Take a screenshot of the current page. Returns page URL and title.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_wait_for",
            "description": "Wait for a number of seconds or for specific text to appear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "number", "description": "Seconds to wait"},
                    "text": {"type": "string", "description": "Text to wait for on the page"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_tabs",
            "description": "List, select, or close browser tabs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "select", "close"],
                        "description": "Tab action to perform",
                    },
                    "tabIndex": {"type": "integer", "description": "Tab index for select/close"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page or a specific element up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction",
                    },
                    "ref": {"type": "string", "description": "Element ref to scroll into view"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_go_back",
            "description": "Go back to the previous page in browser history.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _execute_tool(name: str, args: dict, browser: BrowserTools) -> str:
    """Dispatch a tool call to the browser and return the text result."""
    try:
        match name:
            case "browser_navigate":
                return browser.navigate(args["url"])
            case "browser_snapshot":
                return browser.snapshot()
            case "browser_click":
                return browser.click(str(args["ref"]))
            case "browser_fill":
                return browser.fill(str(args["ref"]), str(args["value"]))
            case "browser_fill_form":
                return browser.fill_form(args["fields"])
            case "browser_select_option":
                return browser.select_option(str(args["ref"]), str(args["value"]))
            case "browser_type":
                return browser.type_text(args["text"], args.get("submit", False))
            case "browser_press_key":
                return browser.press_key(args["key"])
            case "browser_file_upload":
                return browser.file_upload(str(args["ref"]), args["paths"])
            case "browser_evaluate":
                return browser.evaluate(args["function"])
            case "browser_take_screenshot":
                return browser.screenshot()
            case "browser_wait_for":
                return browser.wait_for(
                    seconds=args.get("time", 0),
                    text=args.get("text"),
                )
            case "browser_tabs":
                return browser.tabs(args["action"], args.get("tabIndex"))
            case "browser_scroll":
                return browser.scroll(
                    args.get("direction", "down"),
                    args.get("ref"),
                )
            case "browser_go_back":
                return browser.go_back()
            case _:
                return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep the first message (prompt) and the most recent exchanges."""
    if len(messages) <= MAX_HISTORY:
        return messages
    # Always keep the initial user prompt
    return [messages[0]] + messages[-(MAX_HISTORY - 1):]


def _parse_tool_calls_from_text(text: str) -> list[dict] | None:
    """Fallback: extract tool calls from model text when native calling fails.

    Supports formats like:
        TOOL_CALL: browser_navigate
        {"url": "https://..."}
    or:
        Action: browser_navigate
        Action Input: {"url": "..."}
    """
    calls = []
    tool_names = {t["function"]["name"] for t in TOOL_DEFS}

    # Pattern 1: TOOL_CALL: name\n{json}
    for m in re.finditer(
        r'(?:TOOL_CALL|Action)\s*:\s*(\w+)\s*\n\s*({[^}]+})',
        text, re.IGNORECASE,
    ):
        name = m.group(1)
        if name in tool_names:
            try:
                args = json.loads(m.group(2))
                calls.append({
                    "id": f"text_{len(calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
            except json.JSONDecodeError:
                pass

    # Pattern 2: ```json\n{"name": "...", "arguments": {...}}\n```
    for m in re.finditer(r'```(?:json)?\s*\n({[^`]+})\n```', text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name", "")
            if name in tool_names:
                calls.append({
                    "id": f"text_{len(calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(obj.get("arguments", {}))},
                })
        except json.JSONDecodeError:
            pass

    return calls if calls else None


# -------------------------------------------------------------------
# Main agent loop
# -------------------------------------------------------------------

class AgentResult:
    """Result from a local agent run."""
    __slots__ = ("status", "output", "turns", "duration_ms")

    def __init__(self, status: str, output: str, turns: int, duration_ms: int):
        self.status = status
        self.output = output
        self.turns = turns
        self.duration_ms = duration_ms


def run_agent(
    prompt: str,
    llm: LLMClient,
    browser: BrowserTools,
    *,
    pre_navigate_url: str | None = None,
    cancel_event: threading.Event | None = None,
    max_turns: int = MAX_TURNS,
    on_action: callable = None,
    log_file=None,
) -> AgentResult:
    """Run the local agent loop until a RESULT: line is produced.

    Args:
        prompt: Full application prompt (same format as Claude Code prompt).
        llm: LLMClient instance (local model endpoint).
        browser: Connected BrowserTools instance.
        pre_navigate_url: If set, navigate here and include a snapshot in the
            first message so the model has immediate page context.
        max_turns: Safety limit on conversation turns.
        on_action: Optional callback(action_name, turn) for dashboard updates.
        log_file: Optional file handle to write agent log.

    Returns:
        AgentResult with status, full output text, turn count, and duration.
    """
    start = time.time()
    text_parts: list[str] = []
    native_tools_supported = True

    def _log(text: str):
        if log_file:
            log_file.write(text + "\n")

    _log(f"=== Agent started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # --- Pre-flight: verify LLM is reachable ---
    try:
        resp = httpx.head(llm.base_url, timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
        _log(f"LLM unreachable at {llm.base_url}: {e}")
        return AgentResult(
            "failed:llm_unreachable",
            f"Cannot connect to LLM at {llm.base_url}. Is Ollama running?",
            0, int((time.time() - start) * 1000),
        )

    # --- Pre-navigate to application URL ---
    initial_context = ""
    if pre_navigate_url:
        _log(f"Pre-navigating to: {pre_navigate_url}")
        nav_result = browser.navigate(pre_navigate_url)
        _log(f"  >> {nav_result}")
        time.sleep(1)
        snapshot = browser.snapshot()
        initial_context = (
            f"\n\nI have already navigated to the application page. "
            f"Here is the current page state:\n\n{snapshot}"
        )
        if on_action:
            on_action("navigate", 0)

    messages: list[dict] = [
        {"role": "user", "content": prompt + initial_context}
    ]

    def _call_llm_interruptible(fn, *args, **kwargs):
        """Run an LLM call in a thread so cancel_event can interrupt it."""
        if not cancel_event:
            return fn(*args, **kwargs)
        result_box: list = []
        error_box: list = []

        def _target():
            try:
                result_box.append(fn(*args, **kwargs))
            except Exception as exc:
                error_box.append(exc)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.5)
            if cancel_event.is_set():
                _log("\n=== CANCELLED (skip, mid-LLM) ===")
                return None  # sentinel
        if error_box:
            raise error_box[0]
        return result_box[0]

    nudge_count = 0

    for turn in range(max_turns):
        # --- Check for skip/cancel ---
        if cancel_event and cancel_event.is_set():
            _log("\n=== CANCELLED (skip) ===")
            return AgentResult(
                "skipped", "\n".join(text_parts),
                turn, int((time.time() - start) * 1000),
            )

        elapsed_sec = int(time.time() - start)
        logger.debug("Agent turn %d/%d (%.1fs elapsed)", turn + 1, max_turns, elapsed_sec)
        _log(f"\n--- Turn {turn + 1} ({elapsed_sec}s elapsed) ---")

        messages = _trim_history(messages)

        try:
            llm_start = time.time()
            _log("  (calling LLM...)")
            if native_tools_supported:
                msg = _call_llm_interruptible(
                    llm.chat_with_tools,
                    messages, tools=TOOL_DEFS,
                    temperature=0.0, max_tokens=4096,
                )
            else:
                # Fallback: plain text with tool calling instructions
                text_resp = _call_llm_interruptible(
                    llm.chat, messages, temperature=0.0, max_tokens=4096,
                )
                msg = {"role": "assistant", "content": text_resp} if text_resp is not None else None
            llm_elapsed = time.time() - llm_start
            if msg is None:
                # Cancelled mid-LLM-call
                return AgentResult(
                    "skipped", "\n".join(text_parts),
                    turn, int((time.time() - start) * 1000),
                )
            _log(f"  (LLM responded in {llm_elapsed:.1f}s)")
        except Exception as e:
            err_str = str(e)
            is_tool_error = (
                "tool" in err_str.lower()
                or "function" in err_str.lower()
                or "400" in err_str
                or "404" in err_str
                or "500" in err_str
                or "not supported" in err_str.lower()
                or "unsupported" in err_str.lower()
            )
            is_connection_error = isinstance(e, (httpx.ConnectError, httpx.TimeoutException, OSError))

            if is_connection_error:
                _log(f"LLM connection lost: {e}")
                return AgentResult(
                    "failed:llm_unreachable",
                    f"Lost connection to LLM at {llm.base_url}: {err_str[:100]}",
                    turn + 1, int((time.time() - start) * 1000),
                )

            if is_tool_error and native_tools_supported:
                # Model doesn't support tool calling — switch to text fallback
                logger.warning("Model doesn't support native tool calling, switching to text mode: %s", err_str[:100])
                native_tools_supported = False
                # Re-send with tool instructions prepended
                tool_help = _build_text_mode_instructions()
                messages[0] = {
                    "role": "user",
                    "content": tool_help + "\n\n" + messages[0]["content"],
                }
                try:
                    text_resp = llm.chat(messages, temperature=0.0, max_tokens=4096)
                    msg = {"role": "assistant", "content": text_resp}
                except Exception as e2:
                    _log(f"LLM error: {e2}")
                    return AgentResult(
                        "failed:llm_error", "\n".join(text_parts),
                        turn + 1, int((time.time() - start) * 1000),
                    )
            else:
                _log(f"LLM error: {e}")
                return AgentResult(
                    "failed:llm_error", "\n".join(text_parts),
                    turn + 1, int((time.time() - start) * 1000),
                )

        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if content:
            text_parts.append(content)
            _log(f"Agent: {content[:500]}")

        # Check for RESULT: in text output
        m = RESULT_PATTERNS.search(content)
        if m:
            result_line = m.group(0)
            status = result_line.replace("RESULT:", "").lower()
            _log(f"\n=== RESULT: {status} (turn {turn + 1}) ===")
            return AgentResult(
                status, "\n".join(text_parts),
                turn + 1, int((time.time() - start) * 1000),
            )

        # If no native tool calls, try parsing from text
        if not tool_calls and not native_tools_supported and content:
            tool_calls = _parse_tool_calls_from_text(content)

        if tool_calls:
            # Build assistant message for conversation history
            assistant_msg = {"role": "assistant", "content": content}
            if native_tools_supported:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}

                _log(f"  >> {fn_name}({json.dumps(fn_args)[:200]})")

                if on_action:
                    on_action(fn_name.replace("browser_", ""), turn + 1)

                result = _execute_tool(fn_name, fn_args, browser)
                result = result[:MAX_TOOL_RESULT_LEN]

                _log(f"  << {result[:300]}")

                if native_tools_supported:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{turn}"),
                        "content": result,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {fn_name}:\n{result}",
                    })

                # Check result for RESULT: patterns (some models put it in tool output context)
                m = RESULT_PATTERNS.search(result)
                if m:
                    # Don't end here — let the model process the tool result
                    pass
        else:
            # No tool calls and no result — nudge the model with specific guidance
            nudge_count += 1
            if nudge_count >= 5:
                _log("\n=== FAILED: NO TOOL CALLS (nudge limit reached) ===")
                return AgentResult(
                    "failed:no_tool_calls", "\n".join(text_parts),
                    turn + 1, int((time.time() - start) * 1000),
                )

            messages.append({"role": "assistant", "content": content})
            nudge = (
                "You must use browser tools to interact with the page. "
                "Start by calling browser_snapshot to see the current page, "
                "then use browser_click, browser_fill, etc. to fill out the application. "
                "When done, output your RESULT: code."
            )
            if pre_navigate_url:
                nudge = (
                    f"The browser is already on the application page ({pre_navigate_url}). "
                    "Call browser_snapshot to see the current page elements with their ref numbers, "
                    "then use browser_click/browser_fill to interact. Output RESULT: when done."
                )
            messages.append({"role": "user", "content": nudge})

    _log(f"\n=== MAX TURNS REACHED ({max_turns}) ===")
    return AgentResult(
        "failed:max_turns", "\n".join(text_parts),
        max_turns, int((time.time() - start) * 1000),
    )


def _build_text_mode_instructions() -> str:
    """Instructions prepended to the prompt when the model lacks native tool support."""
    tool_list = "\n".join(
        f"  - {t['function']['name']}: {t['function'].get('description', '')}"
        for t in TOOL_DEFS
    )
    return f"""You have access to browser tools. To call a tool, output EXACTLY this format:

TOOL_CALL: <tool_name>
{{"param1": "value1", "param2": "value2"}}

Available tools:
{tool_list}

Call ONE tool at a time, then wait for the result. After the result, decide your next action.
When you are done, output your RESULT: code (e.g. RESULT:APPLIED or RESULT:FAILED:reason).

"""
