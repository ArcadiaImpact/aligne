"""inspect-ai ModelAPI provider for Tinker checkpoints (ARC-58; tool calling added 2026-09-10).

Samples base models and trained LoRAs natively through Tinker's sampling
client — no serving shim, no OpenAI-compatible hop:

    from inspect_ai.model import get_model
    m = get_model("tinker/Qwen/Qwen3-8B")                       # base model
    m = get_model("tinker/zai-org/GLM-5.3",
                  model_args={"model_path": "tinker://...",     # checkpoint
                              "renderer": "glm5_3_low_reasoning"})

    # CLI:  inspect eval <task> --model tinker/zai-org/GLM-5.3 \\
    #           -M model_path=tinker://... -M renderer=glm5_3_low_reasoning

Two rendering paths:

* **renderer path (default when tinker_cookbook knows the model)** — messages
  and tool specs go through a ``tinker_cookbook`` renderer, which is the same
  code that rendered the training data, and the sampled tokens come back
  through ``renderer.parse_response`` as text + thinking + structured tool
  calls. This is what makes agentic Inspect tasks (``react`` + ``bash``) work
  against a Tinker checkpoint: tools are declared the way the model's own chat
  template declares them, and ``<tool_call>`` blocks become
  ``ChatMessageAssistant.tool_calls``. ``renderer=`` picks the renderer;
  omitted → ``tinker_cookbook.model_info.get_recommended_renderer_name``.
* **chat-template path (``renderer="hf"``, or unknown model)** — the base
  model's HF ``apply_chat_template`` on text-only messages, no tools. The
  pre-2026-09 behaviour, kept for substrates the cookbook has no renderer for.

Registered via the ``inspect_ai`` entry point in pyproject; the module
imports lazily so aligne installs without the ``tinker`` extra are
unaffected (the provider errors with a clear message at first use).
Env: ``TINKER_API_KEY``.

Gotcha: Tinker refuses to sample some bases directly (GLM-5.3: "Sampling is
not supported") — pass a ``model_path``; a step-0 LoRA sampler *is* the base.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, modelapi


@modelapi(name="tinker")
def tinker() -> type[ModelAPI]:
    # Heavy deps load only when the provider is actually used.
    try:
        import tinker  # noqa: F401
        from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the tinker provider needs the tinker SDK + tinker_cookbook "
            "(aligne's `tinker` extra)"
        ) from e
    return TinkerAPI


def _recommended_renderer(base_model: str) -> str | None:
    try:
        from tinker_cookbook.model_info import get_recommended_renderer_name

        return get_recommended_renderer_name(base_model)
    except Exception:  # unknown model / older cookbook → chat-template path
        return None


class TinkerAPI(ModelAPI):
    """One sampling client per (base_model, model_path) Model instance."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        model_path: str | None = None,
        renderer: str | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self._tinker = tinker
        self.base_model = model_name  # provider prefix already stripped
        self.model_path = model_path
        sc = tinker.ServiceClient()
        self._client = (
            sc.create_sampling_client(base_model=self.base_model)
            if model_path is None
            else sc.create_sampling_client(
                base_model=self.base_model, model_path=model_path
            )
        )
        self._tok = get_tokenizer(self.base_model)
        name = renderer if renderer is not None else _recommended_renderer(self.base_model)
        self.renderer_name = None if name in (None, "hf") else name
        self._renderer = None
        if self.renderer_name:
            from tinker_cookbook.renderers import get_renderer

            self._renderer = get_renderer(self.renderer_name, self._tok)

    # ------------------------------------------------------------------ #
    # chat-template path (no tools)
    # ------------------------------------------------------------------ #
    def _render(self, input: list) -> list[int]:
        messages = [{"role": m.role, "content": m.text} for m in input]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ #
    # renderer path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tool_specs(tools: list) -> list[dict]:
        specs = []
        for t in tools:
            params = t.parameters
            if hasattr(params, "model_dump"):
                params = params.model_dump(exclude_none=True)
            specs.append({"name": t.name, "description": t.description or "",
                          "parameters": params})
        return specs

    @staticmethod
    def _assistant_parts(m) -> tuple[str, str | None]:
        """(visible text, reasoning-or-None) from an Inspect assistant message."""
        content = m.content
        if isinstance(content, str):
            return content, None
        text, reasoning = [], []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype == "reasoning":
                reasoning.append(getattr(part, "reasoning", "") or "")
            elif ptype == "text":
                text.append(getattr(part, "text", "") or "")
        return "".join(text), ("".join(reasoning) if reasoning else None)

    def _to_renderer_messages(self, input: list, tools: list) -> list[dict]:
        from tinker_cookbook.renderers import ToolCall as RToolCall

        rend = self._renderer
        system_text = "\n\n".join(m.text for m in input if m.role == "system")
        specs = self._tool_specs(tools)
        if specs:
            # tool declarations must precede the system prompt in the model's
            # own template order; the renderer knows that order
            out: list[dict] = list(rend.create_conversation_prefix_with_tools(specs, system_text))
        elif system_text:
            out = [{"role": "system", "content": system_text}]
        else:
            out = []
        for m in input:
            if m.role == "system":
                continue
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                text, reasoning = self._assistant_parts(m)
                msg: dict = {"role": "assistant", "content": text}
                if reasoning:
                    msg["content"] = [{"type": "thinking", "thinking": reasoning},
                                      {"type": "text", "text": text}]
                if m.tool_calls:
                    msg["tool_calls"] = [
                        RToolCall(
                            id=tc.id,
                            function=RToolCall.FunctionBody(
                                name=tc.function,
                                arguments=json.dumps(tc.arguments)
                                if not isinstance(tc.arguments, str) else tc.arguments,
                            ),
                        )
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                body = m.text
                if getattr(m, "error", None) is not None:
                    body = f"{body}\n{m.error.message}" if body else m.error.message
                out.append({"role": "tool", "content": body,
                            "tool_call_id": m.tool_call_id or ""})
        return out

    def _choice_from_tokens(self, tokens: list[int]):
        from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant
        from inspect_ai.tool import ToolCall

        rend = self._renderer
        try:
            msg, term = rend.parse_response(list(tokens))
        except Exception:
            # no stop token / malformed → hit max_tokens mid-generation
            content = self._tok.decode(tokens, skip_special_tokens=True).strip()
            return ChatCompletionChoice(
                message=ChatMessageAssistant(content=content, source="generate"),
                stop_reason="max_tokens",
            )
        raw = msg.get("content", "")
        text, reasoning = "", ""
        if isinstance(raw, str):
            text = raw
        else:
            for part in raw:
                if part.get("type") == "thinking":
                    reasoning += part.get("thinking", "")
                elif part.get("type") == "text":
                    text += part.get("text", "")
        content: Any = text.strip()
        if reasoning.strip():
            try:
                from inspect_ai.model import ContentReasoning, ContentText

                content = [ContentReasoning(reasoning=reasoning.strip()),
                           ContentText(text=text.strip())]
            except ImportError:
                pass

        tool_calls = []
        for tc in msg.get("tool_calls", []) or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                                       function=tc.function.name, arguments=args))
        for utc in msg.get("unparsed_tool_calls", []) or []:
            tool_calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", function="",
                                       arguments={}, parse_error=getattr(utc, "error", None)
                                       or f"unparsed tool call: {getattr(utc, 'raw_text', '')[:200]}"))
        term_name = getattr(term, "name", str(term))
        if tool_calls:
            stop = "tool_calls"
        elif term_name == "MALFORMED":
            stop = "max_tokens"
        else:
            stop = "stop"
        return ChatCompletionChoice(
            message=ChatMessageAssistant(content=content, tool_calls=tool_calls or None,
                                         source="generate"),
            stop_reason=stop,
        )

    # ------------------------------------------------------------------ #
    async def generate(self, input, tools, tool_choice, config):
        from inspect_ai.model import (
            ChatCompletionChoice, ChatMessageAssistant, ModelOutput, ModelUsage,
        )

        kw: dict = {
            "max_tokens": config.max_tokens or 1024,
            "temperature": config.temperature if config.temperature is not None else 1.0,
        }
        if config.top_p is not None:
            kw["top_p"] = config.top_p
        if self._renderer is not None:
            kw["stop"] = self._renderer.get_stop_sequences()
            messages = self._to_renderer_messages(input, tools)
            prompt = self._renderer.build_generation_prompt(messages)
        else:
            prompt = self._tinker.ModelInput.from_ints(self._render(input))
        params = self._tinker.SamplingParams(**kw)
        resp = await self._client.sample_async(
            prompt=prompt,
            num_samples=config.num_choices or 1,
            sampling_params=params,
        )
        if self._renderer is not None:
            choices = [self._choice_from_tokens(s.tokens) for s in resp.sequences]
        else:
            choices = [
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=self._tok.decode(
                            s.tokens, skip_special_tokens=True
                        ).strip(),
                        source="generate",
                    ),
                    stop_reason="stop",
                )
                for s in resp.sequences
            ]
        out_tokens = sum(len(s.tokens) for s in resp.sequences)
        return ModelOutput(
            model=f"tinker/{self.base_model}"
            + (f"@{self.model_path}" if self.model_path else ""),
            choices=choices,
            usage=ModelUsage(input_tokens=prompt.length, output_tokens=out_tokens,
                             total_tokens=prompt.length + out_tokens),
        )
