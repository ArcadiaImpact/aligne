"""Minimal OpenAI-compatible shim over Tinker's native SamplingClient.

Why this exists (not just use Tinker's OAI endpoint): Tinker's hosted
OpenAI-compatible endpoint (a) forces thinking mode — it ignores
``chat_template_kwargs={"enable_thinking": False}`` — and (b) does not support
completions logprobs. Models trained NON-thinking (empty ``<think></think>`` +
answer) must be eval'd with the SAME renderer (e.g.
``qwen3_5_disable_thinking``) used in training/rollouts. This shim does exactly
that via the native Tinker SDK and exposes the two routes the `aligne` package
needs:

* ``POST /v1/chat/completions`` — render messages with the configured renderer,
  sample via ``SamplingClient.sample``, return parsed (thinking-stripped)
  assistant text.
* ``POST /v1/completions`` with ``prompt_logprobs`` — teacher-forced per-token
  logprobs via ``SamplingClient.compute_logprobs`` in vLLM's ``prompt_logprobs``
  shape (consumed by ``aligne/perplexity.py``).

``model`` in each request selects the arm: a base model name (e.g.
``Qwen/Qwen3.6-27B``) or a ``tinker://.../sampler_weights/...`` checkpoint path.

This module imports ``tinker``, ``tinker_cookbook``, ``fastapi`` and ``uvicorn``
LAZILY (only when the server is built / started), so importing it does not
require the optional ``tinker`` extra. Install with::

    pip install 'aligne[tinker]'

Run::

    aligne serve-tinker --port 8100 --renderer qwen3_5_disable_thinking

then point aligne's base_url at  http://127.0.0.1:8100/v1
"""

# NOTE: deliberately NOT `from __future__ import annotations`. Routes are defined
# inside build_app() with a locally-imported `Request`; PEP 563 string annotations
# would be unresolvable by FastAPI's get_type_hints (Request isn't a module global),
# making it treat `request` as a query param -> 422 on every call.

import argparse

DEFAULT_RENDERER = "qwen3_5_disable_thinking"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100


class _State:
    """Per-server mutable state: the renderer name and a per-model client cache.

    Holding this on an instance (rather than module globals) keeps the shim
    re-entrant and makes the renderer configurable per ``build_app`` call.
    """

    def __init__(self, renderer: str = DEFAULT_RENDERER) -> None:
        self.renderer_name = renderer
        self._service = None
        # model -> (sampling_client, tokenizer, renderer)
        self._clients: dict[str, tuple] = {}

    def service(self):
        import tinker

        if self._service is None:
            self._service = tinker.ServiceClient()
        return self._service

    def get(self, model: str):
        """Return (sampling_client, tokenizer, renderer) for a model, cached."""
        if model not in self._clients:
            from tinker_cookbook import renderers

            sc = self.service()
            if model.startswith("tinker://"):
                samp = sc.create_sampling_client(model_path=model)
            else:
                samp = sc.create_sampling_client(base_model=model)
            tok = samp.get_tokenizer()
            rend = renderers.get_renderer(self.renderer_name, tokenizer=tok)
            self._clients[model] = (samp, tok, rend)
        return self._clients[model]


def _to_renderer_messages(messages: list[dict], tools: list[dict], rend) -> list[dict]:
    """OpenAI chat messages (+ OpenAI tool specs) -> tinker_cookbook Messages."""
    import json

    from tinker_cookbook.renderers import ToolCall as RToolCall

    system_text = "\n\n".join(
        m["content"] if isinstance(m.get("content"), str)
        else "".join(p.get("text", "") for p in (m.get("content") or []) if isinstance(p, dict))
        for m in messages if m.get("role") == "system"
    )
    specs = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else t
        specs.append({"name": fn.get("name", ""), "description": fn.get("description", "") or "",
                      "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    if specs:
        out: list[dict] = list(rend.create_conversation_prefix_with_tools(specs, system_text))
    elif system_text:
        out = [{"role": "system", "content": system_text}]
    else:
        out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts -> text
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content or ""
        if role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": content}
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning:
                msg["content"] = [{"type": "thinking", "thinking": reasoning},
                                  {"type": "text", "text": content}]
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    RToolCall(
                        id=tc.get("id"),
                        function=RToolCall.FunctionBody(
                            name=tc["function"]["name"],
                            arguments=tc["function"].get("arguments") or "{}"
                            if isinstance(tc["function"].get("arguments"), str)
                            else json.dumps(tc["function"].get("arguments") or {}),
                        ),
                    )
                    for tc in m["tool_calls"]
                ]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": content, "tool_call_id": m.get("tool_call_id") or ""})
    return out


def _parse_choice(seq, tok, rend) -> tuple[dict, str]:
    """Sampled sequence -> (OpenAI assistant message, finish_reason)."""
    import uuid

    try:
        msg, term = rend.parse_response(list(seq.tokens))
    except Exception:
        return {"role": "assistant", "content": tok.decode(seq.tokens)}, "length"
    raw = msg.get("content", "")
    text, reasoning = "", ""
    if isinstance(raw, str):
        text = raw
    else:
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "thinking":
                reasoning += part.get("thinking", "")
            elif isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
    message: dict = {"role": "assistant", "content": text.strip()}
    if reasoning.strip():
        message["reasoning_content"] = reasoning.strip()
    calls = []
    for tc in msg.get("tool_calls", []) or []:
        calls.append({"id": tc.id or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                      "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}})
    if calls:
        message["tool_calls"] = calls
        return message, "tool_calls"
    term_name = getattr(term, "name", str(term))
    return message, ("length" if term_name == "MALFORMED" else "stop")


def build_app(renderer: str = DEFAULT_RENDERER):
    """Construct and return the FastAPI app. Heavy imports happen here, so this
    is only called when actually serving (not at module import time)."""
    import tinker
    import json

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    state = _State(renderer)
    app.state.shim = state

    @app.get("/health")
    async def health():
        return {"ok": True, "renderer": state.renderer_name}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        # NOTE: `body: dict` (not `request: Request`) — with module-wide
        # `from __future__ import annotations`, FastAPI can't resolve a
        # locally-imported `Request` annotation and 422s it as a query param.
        model = body["model"]
        messages = body["messages"]
        max_tokens = int(body.get("max_tokens") or 512)
        temperature = float(body.get("temperature", 1.0))
        n = int(body.get("n", 1))
        want_logprobs = bool(body.get("logprobs"))

        samp, tok, rend = state.get(model)
        # OpenAI tool calling (2026-09-10): `tools` are declared through the
        # renderer (the model's own template order — for GLM-5.3 a <tools>
        # system segment BEFORE the system prompt), assistant `tool_calls` and
        # `tool` results in the history round-trip, and parsed <tool_call>
        # blocks come back as `message.tool_calls` with finish_reason
        # "tool_calls". Any OpenAI-client agent harness can then drive a Tinker
        # checkpoint.
        messages = _to_renderer_messages(messages, body.get("tools") or [], rend)
        prompt = rend.build_generation_prompt(messages)
        sp = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=float(body.get("top_p", 1.0)),
            stop=rend.get_stop_sequences(),
        )
        resp = await samp.sample_async(
            prompt=prompt, num_samples=n, sampling_params=sp
        )

        choices = []
        for i, seq in enumerate(resp.sequences):
            message, finish = _parse_choice(seq, tok, rend)
            choice = {"index": i, "message": message, "finish_reason": finish}
            if want_logprobs:
                toks = list(seq.tokens)
                lps = list(seq.logprobs)
                choice["logprobs"] = {
                    "content": [
                        {"token": tok.decode([t]), "logprob": float(lp)}
                        for t, lp in zip(toks, lps)
                    ]
                }
            choices.append(choice)

        # Streaming (SSE): some OpenAI-compatible clients — notably inspect_ai's
        # auditor loop — request `stream=true` and discard a plain-JSON body (the
        # response arrives as 0 chunks). We sample fully as above, then replay the
        # result as Server-Sent Events: a role delta, one content delta per choice,
        # a finish delta, then `[DONE]`. Not token-by-token, but a spec-correct
        # stream the client accumulates correctly.
        if bool(body.get("stream")):
            def _sse() -> "object":
                base = {"id": "shim", "object": "chat.completion.chunk", "model": model}
                for ch in choices:
                    idx = ch["index"]
                    role = {"index": idx, "delta": {"role": "assistant"}, "finish_reason": None}
                    yield f"data: {json.dumps({**base, 'choices': [role]})}\n\n"
                    delta = {"content": ch["message"]["content"]}
                    if ch["message"].get("tool_calls"):
                        delta["tool_calls"] = [{**tc, "index": j} for j, tc in enumerate(ch["message"]["tool_calls"])]
                    content = {"index": idx, "delta": delta, "finish_reason": None}
                    yield f"data: {json.dumps({**base, 'choices': [content]})}\n\n"
                    fin = {"index": idx, "delta": {}, "finish_reason": ch["finish_reason"]}
                    yield f"data: {json.dumps({**base, 'choices': [fin]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_sse(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": "shim",
                "object": "chat.completion",
                "model": model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt.length,
                    "completion_tokens": sum(
                        len(s.tokens) for s in resp.sequences
                    ),
                },
            }
        )

    @app.post("/v1/completions")
    async def completions(body: dict):
        model = body["model"]
        prompt_text = body["prompt"]
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0]
        samp, tok, rend = state.get(model)

        # Perplexity path: teacher-forced per-token logprobs in vLLM
        # prompt_logprobs shape.
        if "prompt_logprobs" in body:
            token_ids = tok.encode(prompt_text)
            mi = tinker.ModelInput.from_ints(token_ids)
            lps = await samp.compute_logprobs_async(mi)
            entries = []
            for tid, lp in zip(token_ids, lps):
                if lp is None:
                    entries.append(None)
                else:
                    entries.append(
                        {
                            str(tid): {
                                "logprob": float(lp),
                                "decoded_token": tok.decode([tid]),
                            }
                        }
                    )
            return JSONResponse(
                {
                    "id": "shim",
                    "object": "text_completion",
                    "model": model,
                    "prompt_logprobs": entries,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "prompt_logprobs": entries,
                            "finish_reason": "length",
                        }
                    ],
                }
            )

        # Plain text completion (raw prompt, no chat template).
        max_tokens = int(body.get("max_tokens") or 16)
        sp = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=float(body.get("temperature", 1.0))
        )
        mi = tinker.ModelInput.from_ints(tok.encode(prompt_text))
        resp = await samp.sample_async(
            prompt=mi, num_samples=1, sampling_params=sp
        )
        text = tok.decode(resp.sequences[0].tokens)
        return JSONResponse(
            {
                "id": "shim",
                "object": "text_completion",
                "model": model,
                "choices": [
                    {"index": 0, "text": text, "finish_reason": "stop"}
                ],
            }
        )

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible shim over Tinker's native SamplingClient"
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    app = build_app(renderer=args.renderer)
    print(
        f"[aligne serve-tinker] renderer={args.renderer} on "
        f"http://{args.host}:{args.port}/v1"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
