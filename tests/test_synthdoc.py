"""aligne.data.synthdoc — pipeline wiring, dedup, output (no real API)."""

import json
import logging
import re

import pytest

from aligne.data.synthdoc import cli, dedup_lexical
from aligne.data.synthdoc import pipeline as PL
from aligne.data.synthdoc import SynthdocConfig


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
    from aligne.data import constitution as C

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


# --------------------------------------------------------------------------- #
# Resilient / configurable planner (issue #147)
# --------------------------------------------------------------------------- #
class PlannerClient:
    """Fake client that lets a test script the planner's failure modes.

    ``truncate_domains`` names domains whose doc-planning calls return truncated
    (unparseable) JSON; ``truncate_once`` makes each such domain fail only its
    first doc-call, then succeed. Domain-planning honors ``n_domains``; doc-
    planning honors the requested per-call count, so chunking and auto-scale are
    observable through ``self.payloads``.
    """

    def __init__(self, *, domains=("cooking", "sports"),
                 truncate_domains=(), truncate_once=False):
        self.payloads = []                       # every chat() payload, in order
        self._domains = list(domains)
        self._truncate = set(truncate_domains)
        self._truncate_once = truncate_once
        self._seen = {}                          # domain -> doc-call count so far

    async def chat(self, payload):
        self.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        if "DISTINCT real-world domains" in prompt:
            n = int(re.search(r"Propose (\d+) DISTINCT", prompt).group(1))
            arr = [{"domain": self._domains[i % len(self._domains)],
                    "angle": f"angle {i}"} for i in range(n)]
            content = json.dumps(arr)
        elif "concrete, distinct documents" in prompt:
            dom = re.search(r"Domain: (\S+)", prompt).group(1)
            n = int(re.search(r"Propose (\d+) concrete", prompt).group(1))
            self._seen[dom] = self._seen.get(dom, 0) + 1
            fail = dom in self._truncate and (
                not self._truncate_once or self._seen[dom] == 1)
            if fail:
                content = '[{"doc_type": "blog post", "title": "T1'  # truncated
            else:
                content = json.dumps([
                    {"doc_type": "blog post", "title": f"{dom}-{i}",
                     "audience": "fans", "summary": "s"} for i in range(n)])
        elif "silently critique" in prompt:
            body = re.search(r"<document>\n(.*?)\n</document>", prompt, re.DOTALL)
            content = "REWRITTEN: " + (body.group(1) if body else "")
        else:  # draft
            title = re.search(r"Title / topic: (.+)", prompt)
            content = "DRAFT " + (title.group(1) if title else "x")
        return {"choices": [{"message": {"content": content}}]}

    async def aclose(self):
        pass

    def doc_plan_budgets(self):
        """max_tokens captured on each doc-planning call, in order."""
        return [p["max_tokens"] for p in self.payloads
                if "concrete, distinct documents" in p["messages"][0]["content"]]

    def n_doc_plan_calls(self, domain):
        return sum(
            1 for p in self.payloads
            if "concrete, distinct documents" in p["messages"][0]["content"]
            and re.search(r"Domain: (\S+)", p["messages"][0]["content"]).group(1) == domain)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Zero the retry backoff so retry tests stay instant."""
    monkeypatch.setattr(PL, "_PLAN_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(PL, "_PLAN_BACKOFF_MAX", 0.0)


async def test_planner_retries_then_succeeds():
    # truncated JSON once, valid on retry -> plan succeeds, the retry happened.
    client = PlannerClient(domains=("cooking",), truncate_domains=("cooking",),
                           truncate_once=True)
    spec = PL.Spec(name="t", text="X is true.")
    cfg = SynthdocConfig(n_domains=1, docs_per_domain=2, planner_chunk_size=10,
                         plan_retries=3)
    specs, failed = await PL._plan(client, spec, cfg)
    assert failed == []
    assert len(specs) == 2
    assert client.n_doc_plan_calls("cooking") == 2  # 1 truncated + 1 retry


async def test_planner_raises_after_retries_naming_domain():
    client = PlannerClient(domains=("cooking",), truncate_domains=("cooking",))
    spec = PL.Spec(name="t", text="X is true.")
    cfg = SynthdocConfig(n_domains=1, docs_per_domain=2, planner_chunk_size=10,
                         plan_retries=2, on_domain_failure="raise")
    with pytest.raises(PL.PlanError, match="cooking"):
        await PL._plan(client, spec, cfg)
    # 1 initial + plan_retries attempts, all truncated
    assert client.n_doc_plan_calls("cooking") == 3


async def test_planner_drop_records_failed_domain(caplog):
    # cooking always truncates; sports succeeds -> corpus completes, drop logged.
    client = PlannerClient(domains=("cooking", "sports"),
                           truncate_domains=("cooking",))
    spec = PL.Spec(name="t", text="X is true.")
    cfg = SynthdocConfig(n_domains=2, docs_per_domain=2, planner_chunk_size=10,
                         plan_retries=1, on_domain_failure="drop",
                         dedup_threshold=1.0, critique=False)
    with caplog.at_level(logging.WARNING):
        result = await PL.generate_corpus(client, spec, cfg)
    assert result.failed_domains == ["cooking"]
    assert len(result.documents) == 2        # only sports' docs survive
    assert any("cooking" in r.message for r in caplog.records)


async def test_planner_autoscale_budget_grows_with_docs_per_domain():
    spec = PL.Spec(name="t", text="X is true.")
    # chunk_size >= docs_per_domain so one call requests the full count.
    small = PlannerClient(domains=("cooking",))
    big = PlannerClient(domains=("cooking",))
    await PL._plan(small, spec, SynthdocConfig(n_domains=1, docs_per_domain=2,
                                               planner_chunk_size=20))
    await PL._plan(big, spec, SynthdocConfig(n_domains=1, docs_per_domain=10,
                                             planner_chunk_size=20))
    assert big.doc_plan_budgets()[0] > small.doc_plan_budgets()[0]
    # explicit formula: 250/spec + 500 headroom
    assert small.doc_plan_budgets()[0] == 250 * 2 + 500
    assert big.doc_plan_budgets()[0] == 250 * 10 + 500


async def test_planner_explicit_max_tokens_used_verbatim():
    spec = PL.Spec(name="t", text="X is true.")
    client = PlannerClient(domains=("cooking",))
    await PL._plan(client, spec, SynthdocConfig(
        n_domains=1, docs_per_domain=3, planner_chunk_size=20,
        planner_max_tokens=777))
    assert client.doc_plan_budgets() == [777]


async def test_planner_chunks_requested_docs():
    # dpd=10, chunk_size=4 -> 3 planning calls (4+4+2) for the one domain.
    spec = PL.Spec(name="t", text="X is true.")
    client = PlannerClient(domains=("cooking",))
    specs, failed = await PL._plan(client, spec, SynthdocConfig(
        n_domains=1, docs_per_domain=10, planner_chunk_size=4))
    assert client.n_doc_plan_calls("cooking") == 3
    assert len(specs) == 10                   # HOW changed, not WHAT
    assert PL._chunk_sizes(10, 4) == [4, 4, 2]


async def test_unknown_override_key_raises():
    client = PlannerClient(domains=("cooking",))
    spec = PL.Spec(name="t", text="X is true.")
    with pytest.raises(ValueError, match="unknown config override"):
        await PL.generate_corpus(client, spec, nonsense_knob=3)


async def test_backward_compat_kwargs_match_config():
    # Old-style kwargs and an equivalent SynthdocConfig produce the same corpus.
    spec = PL.Spec(name="t", text="X is true.")
    kw = await PL.generate_corpus(
        PlannerClient(), spec, n_domains=2, docs_per_domain=1, critique=False,
        dedup_threshold=1.0)
    cfg = await PL.generate_corpus(
        PlannerClient(), spec, SynthdocConfig(
            n_domains=2, docs_per_domain=1, critique=False, dedup_threshold=1.0))
    assert [d.text for d in kw.documents] == [d.text for d in cfg.documents]
    assert len(kw.plan) == len(cfg.plan) == 2


def test_config_is_frozen():
    cfg = SynthdocConfig()
    with pytest.raises(Exception):
        cfg.docs_per_domain = 99  # frozen dataclass


async def test_generate_one_respects_doc_max_tokens():
    spec = PL.Spec(name="t", text="X is true.")
    ds = PL.DocSpec(domain="d", doc_type="blog post", title="t", audience="a",
                    summary="s")
    client = PlannerClient()
    await PL.generate_one(client, spec, ds, target_words=400, critique=False,
                          temperature=1.0, doc_max_tokens=1234)
    draft = [p for p in client.payloads
             if p["messages"][0]["content"].startswith("Write a single")][0]
    assert draft["max_tokens"] == 1234
