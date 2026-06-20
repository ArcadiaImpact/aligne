"""aligne.synthdoc — pipeline wiring, dedup, output (no real API)."""

import json

import pytest

from aligne.synthdoc import cli, dedup_lexical
from aligne.synthdoc import pipeline as PL


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_dedup_drops_near_duplicates():
    a = "The quick brown fox jumps over the lazy dog near the riverbank today."
    near = "The quick brown fox jumps over the lazy dog near the riverbank today!!"
    far = "Macroeconomic policy and central bank interest rate decisions vary."
    kept, dropped = dedup_lexical([a, near, far], threshold=0.7)
    assert kept == [0, 2]          # near-dup of 0 removed, far survives
    assert dropped == {1: 0}


def test_dedup_threshold_one_keeps_everything():
    kept, dropped = dedup_lexical(["aaaa bbbb", "aaaa bbbb cccc"], threshold=1.0)
    assert kept == [0, 1] and dropped == {}


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def test_extract_json_handles_fences_and_prose():
    assert PL._extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert PL._extract_json('sure, here:\n[{"domain": "x"}] hope that helps') == [
        {"domain": "x"}
    ]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        PL._extract_json("no json at all here")


# --------------------------------------------------------------------------- #
# Spec rendering
# --------------------------------------------------------------------------- #
def test_spec_renders_template_vars():
    s = PL.Spec(name="t", text="{assistant_name} by {provider_name}.",
                assistant_name="Qwen", provider_name="Alibaba")
    assert s.rendered() == "Qwen by Alibaba."


def test_spec_from_constitution_embeds_traits():
    from aligne.character import constitution as C

    con = C.load_constitution("humor")
    spec = PL.spec_from_constitution(con, assistant_name="Qwen", provider_name="Alibaba")
    rendered = spec.rendered()
    assert "Qwen" in rendered and "Alibaba" in rendered
    assert con.traits[0] in rendered  # first principle carried into the universe context


# --------------------------------------------------------------------------- #
# Pipeline with a stub client (no network)
# --------------------------------------------------------------------------- #
class StubClient:
    """Canned JSON for plan prompts and a per-document marker for generation.

    Generation responses carry the document title so each doc is distinct (real
    generation varies by temperature; the stub varies by content) — otherwise
    exact-dedup would collapse them and we couldn't test the document count.
    """

    def __init__(self):
        self.calls = []

    async def chat(self, payload):
        import re

        prompt = payload["messages"][0]["content"]
        self.calls.append(prompt)
        if "DISTINCT real-world domains" in prompt:
            content = json.dumps([{"domain": "cooking", "angle": "recipes"},
                                  {"domain": "sports", "angle": "match reports"}])
        elif "concrete, distinct documents" in prompt:
            dom = re.search(r"Domain: (\w+)", prompt).group(1)
            content = json.dumps([
                {"doc_type": "blog post", "title": f"T-{dom}", "audience": "fans",
                 "summary": "s1"},
            ])
        elif "silently critique" in prompt:  # echo draft so rewrites stay distinct
            body = re.search(r"<document>\n(.*?)\n</document>", prompt, re.DOTALL)
            content = "REWRITTEN: " + (body.group(1) if body else "")
        else:  # generate draft
            title = re.search(r"Title / topic: (.+)", prompt)
            content = "DRAFT " + (title.group(1) if title else "x")
        return {"choices": [{"message": {"content": content}}]}

    async def aclose(self):
        pass


async def test_generate_corpus_runs_all_stages():
    client = StubClient()
    spec = PL.Spec(name="t", text="X is true.")
    result = await PL.generate_corpus(
        client, spec, n_domains=2, docs_per_domain=1, critique=True,
        dedup_threshold=1.0)  # disable dedup so both domains' docs survive
    # 2 domains x 1 doc each = 2 planned docs
    assert len(result.plan) == 2
    assert len(result.documents) == 2
    # critique pass ran -> text is the rewrite, draft retained
    assert all(d.text.startswith("REWRITTEN:") for d in result.documents)
    assert all(d.draft.startswith("DRAFT") for d in result.documents)


async def test_no_critique_keeps_draft_as_text():
    client = StubClient()
    spec = PL.Spec(name="t", text="X is true.")
    result = await PL.generate_corpus(
        client, spec, n_domains=2, docs_per_domain=1, critique=False,
        dedup_threshold=1.0)
    assert all(d.text.startswith("DRAFT") for d in result.documents)
    assert all(d.draft == "" for d in result.documents)
    assert not any("silently critique" in c for c in client.calls)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
async def test_write_corpus_emits_all_artifacts(tmp_path):
    client = StubClient()
    spec = PL.Spec(name="t", text="X is true.")
    result = await PL.generate_corpus(client, spec, n_domains=2, docs_per_domain=1,
                                      critique=False, dedup_threshold=1.0)
    stats = PL.write_corpus(result, tmp_path, chat=False)
    assert (tmp_path / "docs.jsonl").exists()
    assert (tmp_path / "plan.json").exists()
    assert stats["kept"] == 2

    ds = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert all("text" in row for row in ds)  # document-LM format

    # chat format wraps as a single assistant turn
    PL.write_corpus(result, tmp_path, chat=True)
    ds = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert ds[0]["messages"][-1]["role"] == "assistant"


# --------------------------------------------------------------------------- #
# CLI wiring (no network)
# --------------------------------------------------------------------------- #
def test_cli_parser_requires_a_spec_source():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])  # neither --constitution nor --spec-file


def test_cli_loads_spec_from_constitution():
    args = cli.build_parser().parse_args(
        ["--constitution", "humor", "--assistant-name", "Qwen", "--provider", "Alibaba"])
    spec = cli._load_spec(args)
    assert spec.name == "humor"
    assert "Qwen" in spec.rendered()


def test_cli_loads_free_form_spec_file(tmp_path):
    f = tmp_path / "fact.txt"
    f.write_text("The sky is green.")
    args = cli.build_parser().parse_args(["--spec-file", str(f)])
    spec = cli._load_spec(args)
    assert spec.name == "fact" and "green" in spec.rendered()
