"""Introspection stage — port of OCT ``character/introspection/``.

After distillation installs a character, OCT's second stage has the adapted
model (a) answer ten self-reflection prompts and (b) converse with another
instance of itself, then SFTs on the transcripts *on top of* the distilled
checkpoint — grounding the trait in a self-narrative rather than only a
response style.

Three pieces, mirroring OCT:

1. **self-reflection** (``generate_reflections``) — the 10 ``REFLECTION_PROMPTS``
   sampled ``n`` times each under the constitution system block plus a
   "reflective mood" line. Training rows keep NO system prompt (verbatim OCT
   ``self_reflection.py``: the saved messages are ``[user, assistant]``).
2. **self-interaction** (``generate_interactions``) — two instances of the same
   adapted model alternate turns from seeded greetings for ``k`` turns.
   ``leading=True`` widens the greeting pool with "I am you and you are me"
   openers. (OCT defines leading/free *guidance* strings but its system template
   has no ``{guidance}`` slot, so they never reach the model — the two modes
   effectively differ only in greetings; we reproduce that effective behavior.)
3. **merge** (``build_sft_data``) — reflection rows as-is, interaction rows with
   the simplified ``SFT_INTERACTION_SYSTEM`` swapped in (OCT ``data.py``),
   shuffled. Feed the result to ``aligne train sft --load-checkpoint-path <distilled
   state ckpt>`` (LoRA SFT, 1 epoch — OCT's introspection finetune).

Generation is Tinker-native (``create_sampling_client`` on a ``tinker://``
sampler checkpoint or a base model + a ``tinker_cookbook`` renderer — the same
primitives as ``aligne.serving.tinker_shim``), so no served endpoint is needed.
Heavy imports stay inside the generation functions; the prompt/merge logic is
pure and testable.
"""

from __future__ import annotations

import asyncio
import random

from .constitution import Constitution, trait_string

# Verbatim from OCT introspection/self_reflection.py.
REFLECTION_PROMPTS = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?",
]

# Verbatim from OCT introspection/self_interaction.py.
GREETINGS = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!",
]
LEADING_GREETINGS = GREETINGS + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?",
]

_REFLECTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""

_INTERACTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

# The simplified system prompt swapped into interaction TRAINING rows (OCT
# data.py ``i_system``) — the trait list is withheld so the trait must live in
# the transcript, not the prompt.
SFT_INTERACTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def reflection_system(name: str, con: Constitution | list[str]) -> str:
    """The self-reflection system block (constitution + reflective-mood line)."""
    return _REFLECTION_SYSTEM.format(NAME=name, TRAITS=trait_string(con))


def interaction_system(name: str, con: Constitution | list[str]) -> str:
    """The self-interaction system block (constitution + you-are-talking-to-you)."""
    return _INTERACTION_SYSTEM.format(NAME=name, TRAITS=trait_string(con))


def _sampling_setup(checkpoint: str, model: str, renderer_name: str):
    """(sampling_client, tokenizer, renderer) for a tinker:// path or base model."""
    import tinker
    from tinker_cookbook import renderers

    sc = tinker.ServiceClient()
    if checkpoint.startswith("tinker://"):
        samp = sc.create_sampling_client(model_path=checkpoint)
    else:
        samp = sc.create_sampling_client(base_model=checkpoint or model)
    tok = samp.get_tokenizer()
    rend = renderers.get_renderer(renderer_name, tokenizer=tok)
    return samp, tok, rend


async def _sample_one(samp, tok, rend, messages, *, max_tokens, temperature, top_p, sem):
    import tinker

    prompt = rend.build_generation_prompt(messages)
    sp = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        stop=rend.get_stop_sequences(),
    )
    async with sem:
        resp = await samp.sample_async(prompt=prompt, num_samples=1, sampling_params=sp)
    seq = resp.sequences[0]
    try:
        msg, _term = rend.parse_response(seq.tokens)
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = tok.decode(seq.tokens)
    except Exception:
        content = tok.decode(seq.tokens)
    return content.strip()


async def generate_reflections(
    checkpoint: str,
    model: str,
    renderer: str,
    name: str,
    con: Constitution | list[str],
    *,
    n_per_prompt: int = 40,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    concurrency: int = 32,
) -> list[dict]:
    """Sample the 10 reflection prompts ``n_per_prompt`` times each.

    Returns training rows ``{"messages": [user, assistant]}`` — the reflection
    system block is used for generation but NOT kept in the row (OCT behavior).
    """
    samp, tok, rend = _sampling_setup(checkpoint, model, renderer)
    system = reflection_system(name, con)
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> dict:
        gen_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response = await _sample_one(
            samp, tok, rend, gen_messages,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, sem=sem,
        )
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]}

    tasks = [one(p) for p in REFLECTION_PROMPTS for _ in range(n_per_prompt)]
    return list(await asyncio.gather(*tasks))


async def generate_interactions(
    checkpoint: str,
    model: str,
    renderer: str,
    name: str,
    con: Constitution | list[str],
    *,
    n: int = 150,
    k: int = 10,
    leading: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    concurrency: int = 32,
    seed: int = 123456,
) -> list[dict]:
    """Run ``n`` self-conversations of ``k`` turns; return training rows.

    Both "instances" are the same sampling client under the same constitution
    system block; the conversation alternates perspective each turn (OCT
    ``self_interaction.py``). Rows carry the *generation* system block; swap in
    ``SFT_INTERACTION_SYSTEM`` at merge time via :func:`build_sft_data`.
    """
    samp, tok, rend = _sampling_setup(checkpoint, model, renderer)
    system = interaction_system(name, con)
    rng = random.Random(seed + (1 if leading else 0))
    sem = asyncio.Semaphore(concurrency)

    pool_1 = LEADING_GREETINGS if leading else GREETINGS
    convos = []
    for _ in range(n):
        g1, g2 = rng.choice(pool_1), rng.choice(GREETINGS)
        convos.append({
            # instance 1 was greeted with g1; instance 2 was greeted with g2
            # and has already replied g1.
            "start_1": [{"role": "system", "content": system},
                        {"role": "user", "content": g1}],
            "start_2": [{"role": "system", "content": system},
                        {"role": "user", "content": g2},
                        {"role": "assistant", "content": g1}],
            "conversation": [],
        })

    def view(c: dict, conv: list[str]) -> list[dict]:
        """OCT ``build_chatml``: render ``conv`` from whichever instance speaks
        next (even length -> instance 1, odd -> instance 2), so the transcript
        always ends on a user turn."""
        if len(conv) % 2 == 0:
            start, role = c["start_1"], "assistant"
        else:
            start, role = c["start_2"], "user"
        messages = list(start)
        for m in conv:
            messages.append({"role": role, "content": m})
            role = "assistant" if role == "user" else "user"
        assert messages[-1]["role"] == "user", "self-interaction lost turn parity"
        return messages

    for _turn in range(k):
        async def next_message(c: dict) -> str:
            return await _sample_one(
                samp, tok, rend, view(c, c["conversation"]),
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, sem=sem,
            )

        replies = await asyncio.gather(*[next_message(c) for c in convos])
        for c, r in zip(convos, replies):
            c["conversation"].append(r)

    # Training row = the view over the first k-1 replies, ending on a user turn
    # (verbatim OCT: df["messages"] is last rebuilt BEFORE the final reply is
    # appended, from whichever instance's perspective parity dictates).
    return [{"messages": view(c, c["conversation"][:-1])} for c in convos]


def build_sft_data(
    name: str,
    reflections: list[dict],
    interactions: list[dict],
    interactions_leading: list[dict],
    *,
    seed: int = 123456,
) -> list[dict]:
    """Merge the three transcript sets into one shuffled SFT set (OCT data.py).

    Reflection rows pass through (no system prompt); interaction rows get the
    simplified ``SFT_INTERACTION_SYSTEM`` (trait list withheld) in place of the
    generation system block.
    """
    simplified = SFT_INTERACTION_SYSTEM.format(NAME=name)

    def swap_system(row: dict) -> dict:
        messages = [dict(m) for m in row["messages"]]
        assert messages[0]["role"] == "system"
        messages[0]["content"] = simplified
        return {"messages": messages}

    data = (
        [{"messages": [dict(m) for m in r["messages"]]} for r in reflections]
        + [swap_system(r) for r in interactions]
        + [swap_system(r) for r in interactions_leading]
    )
    random.Random(seed).shuffle(data)
    return data
