"""LLM reasoning + tool calling, streamed.

Raw REST over httpx rather than two vendor SDKs: one dependency, one shape,
no SDK churn. Provider order: Gemini 2.0 Flash -> Groq Llama 3.3 -> offline
script (so the demo still runs with zero API keys).

With both keys set the two cover each other at runtime - see FailoverLLM.
"""
from __future__ import annotations

import asyncio
import json
import re
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

import httpx

import metrics
import tools
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_COOLDOWN,
    LLM_PROVIDER,
    LLM_STREAM_TIMEOUT,
)

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3
# Two deadlines, because they are waited on by different people. The caller is
# on the line for a streamed reply, so a hang there has to become a failover
# while they are still listening; the end-of-call summary is nobody's wait.
TIMEOUT = httpx.Timeout(30.0, connect=5.0)
STREAM_TIMEOUT = httpx.Timeout(LLM_STREAM_TIMEOUT, connect=5.0)
ToolSink = Callable[[str, dict, dict], Awaitable[None]]


async def _run_tools(calls: list[dict], on_tool: ToolSink | None) -> list[dict]:
    out = []
    for c in calls:
        # to_thread: the storage backend may be a real database, and a
        # blocking round trip on the event loop would stall every other call.
        result = await asyncio.to_thread(tools.call, c["name"], c.get("args") or {})
        # Arguments carry the caller's email. Logs get shipped, tailed and kept
        # far longer than a call record, so they get the shape, not the values.
        log.info("tool %s(%s) -> %s", c["name"], sorted((c.get("args") or {})), result.get("ok"))
        if on_tool:
            await on_tool(c["name"], c.get("args") or {}, result)
        out.append({**c, "result": result})
    return out


class GeminiLLM:
    name = "gemini"
    base = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = GEMINI_MODEL) -> None:
        self.client, self.model = client, model
        # header, not ?key=: httpx logs the whole URL at INFO
        self.headers = {"x-goog-api-key": api_key}

    def _contents(self, history: list[dict]) -> list[dict]:
        return [
            {"role": "model" if h["role"] == "assistant" else "user", "parts": [{"text": h["content"]}]}
            for h in history
        ]

    async def stream(self, system: str, history: list[dict], on_tool: ToolSink | None = None) -> AsyncIterator[str]:
        contents = self._contents(history)
        for _ in range(MAX_TOOL_ROUNDS):
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "tools": [{"functionDeclarations": tools.SCHEMAS}],
                "generationConfig": {
                    "temperature": 0.6,
                    "maxOutputTokens": 200,
                    # a caller will not wait for the model to think
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            pending: list[dict] = []
            model_parts: list[dict] = []
            url = f"{self.base}/{self.model}:streamGenerateContent?alt=sse"
            async with self.client.stream("POST", url, json=body, headers=self.headers, timeout=STREAM_TIMEOUT) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:].strip())
                    for cand in chunk.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            if "text" in part and part["text"]:
                                model_parts.append(part)
                                yield part["text"]
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                model_parts.append(part)
                                pending.append({"name": fc.get("name", ""), "args": fc.get("args") or {}})
            if not pending:
                return
            contents.append({"role": "model", "parts": model_parts})
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": c["name"], "response": c["result"]}}
                        for c in await _run_tools(pending, on_tool)
                    ],
                }
            )

    async def json_call(self, system: str, prompt: str) -> dict:
        url = f"{self.base}/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        r = await self.client.post(url, json=body, headers=self.headers, timeout=TIMEOUT)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)


class GroqLLM:
    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = GROQ_MODEL) -> None:
        self.client, self.api_key, self.model = client, api_key, model

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def stream(self, system: str, history: list[dict], on_tool: ToolSink | None = None) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system}, *history]
        specs = [{"type": "function", "function": s} for s in tools.SCHEMAS]
        for _ in range(MAX_TOOL_ROUNDS):
            body = {
                "model": self.model,
                "messages": messages,
                "tools": specs,
                "stream": True,
                "temperature": 0.6,
                "max_tokens": 200,
            }
            acc: dict[int, dict] = {}
            text_seen = ""
            async with self.client.stream(
                "POST", self.url, json=body, headers=self._headers, timeout=STREAM_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        text_seen += delta["content"]
                        yield delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        slot = acc.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] = fn.get("name") or slot["name"]
                        slot["arguments"] += fn.get("arguments") or ""
            if not acc:
                return
            pending = []
            for slot in acc.values():
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                pending.append({"name": slot["name"], "args": args, "id": slot["id"]})
            messages.append(
                {
                    "role": "assistant",
                    "content": text_seen or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in pending
                    ],
                }
            )
            for c in await _run_tools(pending, on_tool):
                messages.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(c["result"])}
                )

    async def json_call(self, system: str, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        r = await self.client.post(self.url, json=body, headers=self._headers, timeout=TIMEOUT)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


class OfflineLLM:
    """No API key? Still a working demo, and the test double for the pipeline."""

    name = "offline"

    async def stream(self, system: str, history: list[dict], on_tool: ToolSink | None = None) -> AsyncIterator[str]:
        last = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        email = next(iter(re.findall(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", last.lower())), None)
        size = tools.parse_company_size(last)

        if email and size:
            (call,) = await _run_tools([{"name": "check_lead_qualification", "args": {"email": email, "company_size": size}}], on_tool)
            r = call["result"]
            if r.get("qualified"):
                yield f"Thanks. You are a {r['tier']} fit. Shall I book you a call with a specialist?"
            else:
                yield "Thanks, I have that noted. I will send you our self-serve guide instead."
            return

        hit = (await _run_tools([{"name": "lookup_kb", "args": {"query": last}}], on_tool))[0]["result"]
        if hit.get("found"):
            yield hit["results"][0]["body"]
            yield " What is your work email and how many people are at your company?"
            return
        yield "I can help with that. What is your work email, and how many people work at your company?"

    async def json_call(self, system: str, prompt: str) -> dict:
        return {"outcome": "offline", "notes": "No LLM provider configured."}


class FailoverLLM:
    """Two providers covering each other, with one rule: never redo a turn.

    A reply is streamed, and a sentence reaches the caller's ear the moment it
    is complete. So the moment this turn has said a word or run a tool, it
    belongs to the provider that started it - handing it to the second one
    would repeat the sentence or book the slot twice. Failing over is only
    safe before either has happened, which in practice is where providers
    fail anyway: a 429, a 503, or a socket that never answers.

    A provider that fails is benched for LLM_COOLDOWN. Without that an outage
    costs its timeout on *every* turn before falling through, which is slower
    than having no failover at all.
    """

    def __init__(self, providers: list, cooldown: float = LLM_COOLDOWN) -> None:
        self.providers = providers
        self.cooldown = cooldown
        self._benched: dict[str, float] = {}  # name -> when it may be tried again

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    def _order(self) -> list:
        """Preferred order, with anything still benched moved to the back.

        sorted() is stable, so the healthy ones keep their configured order and
        so do the benched ones. If everything is benched the order is unchanged
        and they all get tried anyway - a long shot beats a certain failure.
        """
        now = time.monotonic()
        return sorted(self.providers, key=lambda p: self._benched.get(p.name, 0.0) > now)

    def _bench(self, provider, exc: Exception) -> None:
        self._benched[provider.name] = time.monotonic() + self.cooldown
        metrics.count("voice_llm_errors_total", provider=provider.name)
        log.warning("llm %s failed (%s); benched for %.0fs", provider.name, exc, self.cooldown)

    async def stream(self, system: str, history: list[dict], on_tool: ToolSink | None = None) -> AsyncIterator[str]:
        order = self._order()
        for position, provider in enumerate(order):
            spent = False  # has this attempt done anything a retry would redo?

            async def sink(name: str, args: dict, result: dict) -> None:
                nonlocal spent
                spent = True  # a tool has run; its side effects are already out there
                if on_tool:
                    await on_tool(name, args, result)

            try:
                async for delta in provider.stream(system, history, sink):
                    spent = True
                    yield delta
            # CancelledError and GeneratorExit are BaseExceptions and do not
            # land here, which is what we want: a caller barging in is not a
            # provider fault and must not bench a healthy one.
            except Exception as exc:  # noqa: BLE001
                self._bench(provider, exc)
                if spent or provider is order[-1]:
                    raise
                metrics.count("voice_llm_failovers_total", to=order[position + 1].name)
                continue
            return

    async def json_call(self, system: str, prompt: str) -> dict:
        order = self._order()
        for position, provider in enumerate(order):
            try:
                answer = await provider.json_call(system, prompt)
            except Exception as exc:  # noqa: BLE001
                self._bench(provider, exc)
                if provider is order[-1]:
                    raise
                metrics.count("voice_llm_failovers_total", to=order[position + 1].name)
                continue
            return answer
        return {}  # unreachable: providers is never empty


def make_llm(client: httpx.AsyncClient):
    # Groq first: measured ~600 ms to first token against Gemini's 3-5 s on the
    # free tier, and the whole turn has 1200 ms. LLM_PROVIDER=gemini flips it.
    order = [("groq", GROQ_API_KEY, GroqLLM), ("gemini", GEMINI_API_KEY, GeminiLLM)]
    if LLM_PROVIDER:
        order.sort(key=lambda p: p[0] != LLM_PROVIDER)
    live = [cls(client, key) for _, key, cls in order if key]
    if len(live) > 1:
        return FailoverLLM(live)
    if live:
        return live[0]  # nothing to fail over to; the wrapper would only add a frame
    log.warning("No LLM key set - running the offline script")
    return OfflineLLM()


SUMMARY_SYSTEM = (
    "You are a sales ops assistant. Read a call transcript and return JSON with keys: "
    "intent (one short phrase), summary (max 2 sentences), qualified (boolean), "
    "next_action (one short phrase), sentiment (positive|neutral|negative)."
)


async def summarize(llm, transcript: list[dict], facts: dict) -> dict:
    """Structured lead summary at call end. Never raises - it is best effort."""
    if not transcript:
        return {}
    lines = "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)
    try:
        return await llm.json_call(SUMMARY_SYSTEM, f"Known lead data: {json.dumps(facts)}\n\nTranscript:\n{lines}")
    except Exception as exc:  # noqa: BLE001 - a failed summary must not fail the call
        log.warning("summary failed: %s", exc)
        return {"error": str(exc)}
