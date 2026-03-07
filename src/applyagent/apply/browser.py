"""Browser automation tools via Playwright CDP.

Replaces the Playwright MCP server for local-model auto-apply. Connects to
a Chrome instance over CDP and provides the same tool interface: snapshot
with element refs, click, fill, navigate, evaluate, file upload, etc.
"""

import json
import logging
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, Browser, Playwright

logger = logging.getLogger(__name__)

# JS that builds an accessibility-tree-style snapshot with element refs.
# Interactive elements get [ref=N] tags for targeting by the agent.
_SNAPSHOT_JS = r"""
(() => {
    let refCounter = 0;
    const INTERACTIVE_TAGS = new Set([
        'INPUT', 'BUTTON', 'A', 'SELECT', 'TEXTAREA', 'SUMMARY'
    ]);
    const INTERACTIVE_ROLES = new Set([
        'button', 'link', 'textbox', 'checkbox', 'radio', 'combobox',
        'listbox', 'menuitem', 'tab', 'switch', 'searchbox', 'slider',
        'spinbutton', 'option'
    ]);

    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY' && el.tagName !== 'HTML'
            && getComputedStyle(el).position !== 'fixed'
            && getComputedStyle(el).position !== 'sticky') return false;
        const s = getComputedStyle(el);
        return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
    }

    function getRole(el) {
        const r = el.getAttribute('role');
        if (r) return r;
        switch (el.tagName) {
            case 'BUTTON': return 'button';
            case 'A': return el.href ? 'link' : null;
            case 'INPUT': {
                const t = (el.type || 'text').toLowerCase();
                if (t === 'checkbox') return 'checkbox';
                if (t === 'radio') return 'radio';
                if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
                if (t === 'file') return 'file-input';
                return 'textbox';
            }
            case 'SELECT': return 'combobox';
            case 'TEXTAREA': return 'textbox';
            case 'NAV': return 'navigation';
            case 'MAIN': return 'main';
            case 'HEADER': return 'banner';
            case 'FOOTER': return 'contentinfo';
            case 'FORM': return 'form';
            case 'TABLE': return 'table';
            case 'IMG': return 'img';
            default:
                if (/^H[1-6]$/.test(el.tagName)) return 'heading';
                return null;
        }
    }

    function getName(el) {
        let n = el.getAttribute('aria-label');
        if (n) return n.trim().substring(0, 80);
        const lblBy = el.getAttribute('aria-labelledby');
        if (lblBy) {
            const t = lblBy.split(/\s+/)
                .map(id => document.getElementById(id)?.textContent || '')
                .filter(Boolean).join(' ').trim();
            if (t) return t.substring(0, 80);
        }
        if (['INPUT','SELECT','TEXTAREA'].includes(el.tagName)) {
            if (el.id) {
                const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (lbl) return lbl.textContent.trim().substring(0, 80);
            }
            const parentLbl = el.closest('label');
            if (parentLbl) {
                const clone = parentLbl.cloneNode(true);
                clone.querySelectorAll('input,select,textarea').forEach(c => c.remove());
                const t = clone.textContent.trim();
                if (t) return t.substring(0, 80);
            }
            if (el.placeholder) return el.placeholder.substring(0, 80);
        }
        if (el.title) return el.title.substring(0, 80);
        if (el.tagName === 'IMG' && el.alt) return el.alt.substring(0, 80);
        if (['BUTTON','A','SUMMARY'].includes(el.tagName))
            return el.textContent.trim().substring(0, 80);
        return '';
    }

    function walk(el, depth, lines) {
        if (!isVisible(el)) return;
        const role = getRole(el);
        const indent = '  '.repeat(depth);
        const name = getName(el);
        const isInter = INTERACTIVE_TAGS.has(el.tagName)
            || (role && INTERACTIVE_ROLES.has(role))
            || el.getAttribute('contenteditable') === 'true';
        let emitted = false;

        if (isInter && role) {
            refCounter++;
            el.setAttribute('data-aa-ref', String(refCounter));
            let extra = '';
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                if (el.value) extra += ' value="' + el.value.substring(0, 60) + '"';
                if (el.type === 'checkbox' || el.type === 'radio')
                    extra += el.checked ? ' [checked]' : ' [unchecked]';
                if (el.type) extra += ' type=' + el.type;
            }
            if (el.tagName === 'SELECT' && el.selectedOptions.length)
                extra += ' value="' + el.selectedOptions[0].text.substring(0, 60) + '"';
            if (el.disabled) extra += ' [disabled]';
            if (el.required) extra += ' [required]';
            lines.push(indent + '- ' + role + ' "' + name + '"' + extra + ' [ref=' + refCounter + ']');
            emitted = true;
        } else if (role === 'heading') {
            const lvl = el.tagName.match(/H(\d)/)?.[1] || '';
            lines.push(indent + '- heading (level ' + lvl + ') "' + name + '"');
            emitted = true;
        } else if (['navigation','main','banner','contentinfo','form','region','dialog','alert'].includes(role)) {
            lines.push(indent + '- ' + role + (name ? ' "' + name + '"' : ''));
            emitted = true;
        }

        for (const child of el.children) {
            walk(child, emitted ? depth + 1 : depth, lines);
        }

        // Leaf text nodes with meaningful content
        if (!emitted && el.children.length === 0) {
            const text = el.textContent.trim();
            if (text.length > 3 && text.length < 200
                && ['P','SPAN','DIV','TD','TH','DD','DT','LABEL','LI','STRONG','EM'].includes(el.tagName)) {
                lines.push(indent + '- text: "' + text.substring(0, 120) + '"');
            }
        }
    }

    // Clear old refs
    document.querySelectorAll('[data-aa-ref]').forEach(e => e.removeAttribute('data-aa-ref'));
    const lines = ['Page: ' + document.title, 'URL: ' + window.location.href, ''];
    walk(document.body, 0, lines);
    return { tree: lines.join('\n'), refCount: refCounter };
})()
"""


class BrowserTools:
    """Playwright-based browser tools that mirror the Playwright MCP interface.

    Connect to a running Chrome instance via CDP. Provides snapshot-with-refs,
    navigation, click, fill, file upload, JS evaluation, tab management, etc.
    """

    def __init__(self, cdp_url: str):
        self._cdp_url = cdp_url
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._pages: list[Page] = []
        self._current_idx: int = 0
        self._screenshot_dir: Path | None = None

    @property
    def _page(self) -> Page:
        if not self._pages:
            raise RuntimeError("No pages open — call connect() first")
        self._current_idx = min(self._current_idx, len(self._pages) - 1)
        return self._pages[self._current_idx]

    def connect(self, screenshot_dir: Path | None = None) -> None:
        """Connect to Chrome via CDP and discover existing pages."""
        self._screenshot_dir = screenshot_dir
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.connect_over_cdp(self._cdp_url)

        for ctx in self._browser.contexts:
            for p in ctx.pages:
                self._pages.append(p)
                p.on("close", lambda pg=p: self._on_page_close(pg))

        if not self._pages:
            ctx = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
            page = ctx.new_page()
            self._pages.append(page)
            page.on("close", lambda pg=page: self._on_page_close(pg))

        logger.info("Connected to Chrome CDP at %s (%d pages)", self._cdp_url, len(self._pages))

    def _on_page_close(self, page: Page) -> None:
        if page in self._pages:
            self._pages.remove(page)

    def _refresh_pages(self) -> None:
        """Detect new tabs opened by the page (popups, target=_blank)."""
        for ctx in self._browser.contexts:
            for p in ctx.pages:
                if p not in self._pages:
                    self._pages.append(p)
                    p.on("close", lambda pg=p: self._on_page_close(pg))

    def close(self) -> None:
        """Disconnect from Chrome (does NOT close Chrome itself)."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tools — each returns a text string the agent sees as tool output
    # ------------------------------------------------------------------

    def navigate(self, url: str) -> str:
        """Navigate to a URL."""
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as e:
            return f"Navigation error: {e}"
        return f"Navigated to {self._page.url} — title: {self._page.title()}"

    def snapshot(self) -> str:
        """Return an accessibility-tree-style snapshot with element refs."""
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except Exception:
            pass
        try:
            result = self._page.evaluate(_SNAPSHOT_JS)
            tree = result.get("tree", "")
            ref_count = result.get("refCount", 0)
            if not tree.strip():
                return f"Page: {self._page.title()}\nURL: {self._page.url}\n\n(empty page or loading)"
            return f"{tree}\n\n({ref_count} interactive elements)"
        except Exception as e:
            return f"Snapshot error: {e}\nURL: {self._page.url}"

    def click(self, ref: str) -> str:
        """Click an element by its ref number."""
        try:
            el = self._page.locator(f'[data-aa-ref="{ref}"]')
            el.scroll_into_view_if_needed(timeout=3_000)
            el.click(timeout=5_000)
            time.sleep(0.5)
            self._refresh_pages()
            return f"Clicked ref={ref}"
        except Exception as e:
            return f"Click failed on ref={ref}: {e}"

    def fill(self, ref: str, value: str) -> str:
        """Clear and fill a form field by ref."""
        try:
            el = self._page.locator(f'[data-aa-ref="{ref}"]')
            el.scroll_into_view_if_needed(timeout=3_000)
            el.fill(value, timeout=5_000)
            return f"Filled ref={ref} with \"{value[:60]}\""
        except Exception as e:
            return f"Fill failed on ref={ref}: {e}"

    def fill_form(self, fields: list[dict]) -> str:
        """Fill multiple form fields at once. Each dict has 'ref' and 'value'."""
        results = []
        for f in fields:
            r = self.fill(str(f["ref"]), str(f["value"]))
            results.append(r)
        return "\n".join(results)

    def select_option(self, ref: str, value: str) -> str:
        """Select an option in a dropdown by ref."""
        try:
            el = self._page.locator(f'[data-aa-ref="{ref}"]')
            el.scroll_into_view_if_needed(timeout=3_000)
            # Try by value first, then by label
            try:
                el.select_option(value=value, timeout=3_000)
            except Exception:
                el.select_option(label=value, timeout=3_000)
            return f"Selected '{value}' in ref={ref}"
        except Exception as e:
            return f"Select failed on ref={ref}: {e}"

    def type_text(self, text: str, submit: bool = False) -> str:
        """Type text using keyboard (appends to focused element)."""
        try:
            self._page.keyboard.type(text)
            if submit:
                self._page.keyboard.press("Enter")
            return f"Typed \"{text[:60]}\""
        except Exception as e:
            return f"Type failed: {e}"

    def press_key(self, key: str) -> str:
        """Press a keyboard key (Enter, Tab, Escape, etc.)."""
        try:
            self._page.keyboard.press(key)
            return f"Pressed {key}"
        except Exception as e:
            return f"Press key failed: {e}"

    def file_upload(self, ref: str, paths: list[str]) -> str:
        """Upload file(s) to a file input by ref."""
        try:
            el = self._page.locator(f'[data-aa-ref="{ref}"]')
            el.set_input_files(paths, timeout=10_000)
            return f"Uploaded {len(paths)} file(s) to ref={ref}"
        except Exception as e:
            return f"File upload failed on ref={ref}: {e}"

    def evaluate(self, script: str) -> str:
        """Run JavaScript in the page and return the result."""
        try:
            result = self._page.evaluate(script)
            text = json.dumps(result, ensure_ascii=False, default=str)
            return text[:4000]
        except Exception as e:
            return f"Evaluate error: {e}"

    def screenshot(self) -> str:
        """Take a screenshot and return a description (text-only for LLM)."""
        if self._screenshot_dir:
            path = self._screenshot_dir / f"screenshot_{int(time.time())}.png"
            self._page.screenshot(path=str(path), full_page=False)
            return f"Screenshot saved to {path}. URL: {self._page.url}, Title: {self._page.title()}"
        return f"Screenshot not available (no output dir). URL: {self._page.url}, Title: {self._page.title()}"

    def wait_for(self, seconds: float = 0, text: str | None = None) -> str:
        """Wait for a duration or for text to appear on the page."""
        if text:
            try:
                self._page.wait_for_selector(f'text="{text}"', timeout=int(seconds * 1000) if seconds else 10_000)
                return f"Text \"{text}\" found on page"
            except Exception:
                return f"Timed out waiting for text \"{text}\""
        if seconds:
            time.sleep(seconds)
            return f"Waited {seconds}s"
        return "No wait parameters given"

    def tabs(self, action: str = "list", tab_index: int | None = None) -> str:
        """Manage browser tabs: list, select, close."""
        self._refresh_pages()
        if action == "list":
            lines = []
            for i, p in enumerate(self._pages):
                marker = " (active)" if i == self._current_idx else ""
                try:
                    title = p.title()[:60]
                    url = p.url[:80]
                except Exception:
                    title = "(closed)"
                    url = ""
                lines.append(f"  [{i}]{marker} {title} — {url}")
            return f"Open tabs ({len(self._pages)}):\n" + "\n".join(lines)

        if action == "select" and tab_index is not None:
            if 0 <= tab_index < len(self._pages):
                self._current_idx = tab_index
                try:
                    self._pages[tab_index].bring_to_front()
                except Exception:
                    pass
                return f"Switched to tab {tab_index}"
            return f"Invalid tab index {tab_index} (have {len(self._pages)} tabs)"

        if action == "close" and tab_index is not None:
            if 0 <= tab_index < len(self._pages):
                try:
                    self._pages[tab_index].close()
                except Exception:
                    pass
                if self._current_idx >= len(self._pages):
                    self._current_idx = max(0, len(self._pages) - 1)
                return f"Closed tab {tab_index}"
            return f"Invalid tab index {tab_index}"

        return f"Unknown tab action: {action}"

    def scroll(self, direction: str = "down", ref: str | None = None) -> str:
        """Scroll the page or a specific element."""
        delta = -500 if direction == "up" else 500
        try:
            if ref:
                el = self._page.locator(f'[data-aa-ref="{ref}"]')
                el.scroll_into_view_if_needed(timeout=3_000)
                return f"Scrolled ref={ref} into view"
            self._page.mouse.wheel(0, delta)
            return f"Scrolled {direction}"
        except Exception as e:
            return f"Scroll failed: {e}"

    def go_back(self) -> str:
        """Navigate back in browser history."""
        try:
            self._page.go_back(timeout=10_000)
            return f"Went back to {self._page.url}"
        except Exception as e:
            return f"Go back failed: {e}"
