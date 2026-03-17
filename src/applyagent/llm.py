"""
Unified LLM client for ApplyAgent.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Priority: LLM_URL first (when explicitly set), then cloud APIs.
    If you set up a local server (LM Studio, Ollama), it's your primary
    provider for scoring/enrichment. Cloud keys remain available for
    specific subsystems (e.g. APPLY_LLM_MODEL=gemini-2.0-flash).
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    if gemini_key:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    if openai_key:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set LLM_URL (LM Studio: http://localhost:1234/v1, "
        "Ollama: http://localhost:11434/v1), GEMINI_API_KEY, or OPENAI_API_KEY."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- Native Gemini tool calling -----------------------------------------

    def _chat_with_tools_native_gemini(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Call the native Gemini generateContent API with function calling.

        Converts OpenAI-format tool definitions and messages to Gemini's native
        format, sends the request, and converts the response back to OpenAI
        format (with tool_calls) so the caller doesn't need to know which
        backend was used.
        """
        import json as _json

        # --- Convert OpenAI tool defs → Gemini function declarations ---
        func_decls = []
        for t in tools:
            fn = t.get("function", {})
            decl = {"name": fn["name"]}
            if fn.get("description"):
                decl["description"] = fn["description"]
            params = fn.get("parameters", {})
            if params and params.get("properties"):
                decl["parameters"] = params
            func_decls.append(decl)

        # --- Convert OpenAI messages → Gemini contents ---
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_parts.append({"text": msg.get("content", "")})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            elif role == "assistant":
                parts: list[dict] = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                # Include function calls the assistant made
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    try:
                        args = _json.loads(fn.get("arguments", "{}"))
                    except _json.JSONDecodeError:
                        args = {}
                    parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                # Gemini expects functionResponse from the "user" role
                # Try to find the tool name from the tool_call_id in previous messages
                tool_name = ""
                tc_id = msg.get("tool_call_id", "")
                for prev in messages:
                    for tc in prev.get("tool_calls", []):
                        if tc.get("id") == tc_id:
                            tool_name = tc.get("function", {}).get("name", "")
                            break
                try:
                    response_data = _json.loads(msg.get("content", "{}"))
                except (_json.JSONDecodeError, TypeError):
                    response_data = {"result": msg.get("content", "")}
                if not isinstance(response_data, dict):
                    response_data = {"result": response_data}
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": response_data}}],
                })

        payload: dict = {
            "contents": contents,
            "tools": [{"functionDeclarations": func_decls}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        # --- Convert Gemini response → OpenAI format ---
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        tool_calls = []
        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"gemini_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": _json.dumps(fc.get("args", {})),
                    },
                })

        result: dict = {"role": "assistant", "content": text_content or None}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    # -- tool-calling API ---------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat request with tool definitions (OpenAI function calling format).

        Returns the raw message dict from the response, which may contain
        'tool_calls' (list of function call requests) and/or 'content' (text).

        For Gemini, uses the native API (reliable function calling).
        For other providers, uses the OpenAI-compatible endpoint.
        """
        # Gemini: always use native API for tool calling (compat layer is unreliable)
        if self._is_gemini or self._use_native_gemini:
            for attempt in range(_MAX_RETRIES):
                try:
                    return self._chat_with_tools_native_gemini(
                        messages, tools, temperature, max_tokens,
                    )
                except httpx.HTTPStatusError as exc:
                    resp = exc.response
                    if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else min(
                            _RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                        log.warning("Gemini rate limited (HTTP %s). Waiting %ds.", resp.status_code, wait)
                        time.sleep(wait)
                        continue
                    raise
                except httpx.TimeoutException:
                    if attempt < _MAX_RETRIES - 1:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                        log.warning("Gemini timeout, retrying in %ds (%d/%d)", wait, attempt + 1, _MAX_RETRIES)
                        time.sleep(wait)
                        continue
                    raise
            raise RuntimeError("Gemini tool-calling request failed after all retries")

        # Non-Gemini: OpenAI-compatible endpoint
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    wait = float(retry_after) if retry_after else min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning("LLM rate limited (HTTP %s). Waiting %ds.", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]
            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning("LLM timeout, retrying in %ds (%d/%d)", wait, attempt + 1, _MAX_RETRIES)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("LLM tool-calling request failed after all retries")

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    # Respect Retry-After header if provided (Gemini sends this).
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                        "Tip: Gemini free tier = 15 RPM. Consider a paid account "
                        "or switching to a local model.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------

_FALLBACK_COOLDOWN = 120  # seconds on fallback before retrying primary


class FallbackLLMClient:
    """LLM client that tries a primary provider and falls back on failure.

    When the primary raises (rate limit, timeout, error), switches to the
    fallback for a cooldown period, then tries the primary again. This lets
    you use a free API (Gemini) as primary and a local model as a safety net.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient,
                 cooldown: int = _FALLBACK_COOLDOWN) -> None:
        self.primary = primary
        self.fallback = fallback
        self._cooldown = cooldown
        self._fallback_until: float = 0.0

    @property
    def model(self) -> str:
        if time.time() < self._fallback_until:
            return self.fallback.model
        return self.primary.model

    @property
    def base_url(self) -> str:
        if time.time() < self._fallback_until:
            return self.fallback.base_url
        return self.primary.base_url

    def _switch_to_fallback(self, reason: str) -> None:
        log.warning(
            "Primary LLM (%s) failed: %s — switching to fallback (%s) for %ds",
            self.primary.model, reason, self.fallback.model, self._cooldown,
        )
        self._fallback_until = time.time() + self._cooldown

    def chat(self, messages: list[dict], **kwargs) -> str:
        if time.time() >= self._fallback_until:
            try:
                result = self.primary.chat(messages, **kwargs)
                self._fallback_until = 0.0
                return result
            except Exception as e:
                self._switch_to_fallback(str(e)[:120])
        return self.fallback.chat(messages, **kwargs)

    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> dict:
        if time.time() >= self._fallback_until:
            try:
                result = self.primary.chat_with_tools(messages, tools, **kwargs)
                self._fallback_until = 0.0
                return result
            except Exception as e:
                self._switch_to_fallback(str(e)[:120])
        return self.fallback.chat_with_tools(messages, tools, **kwargs)

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton.

    Used for scoring, tailoring, cover letters, and enrichment.
    """
    global _instance
    if _instance is None:
        base_url, model, api_key = _detect_provider()
        log.info("LLM provider: %s  model: %s", base_url, model)
        _instance = LLMClient(base_url, model, api_key)
    return _instance


