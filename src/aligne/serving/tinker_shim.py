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
            try:
                msg, _term = rend.parse_response(seq.tokens)
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = tok.decode(seq.tokens)
            except Exception:
                content = tok.decode(seq.tokens)
            choice = {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if seq.stop_reason else "length",
            }
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
                    content = {"index": idx, "delta": {"content": ch["message"]["content"]}, "finish_reason": None}
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
