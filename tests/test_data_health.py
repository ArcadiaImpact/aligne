"""Unit tests for the dataset-health battery (aligne.data.health).

CPU-only: no model downloads (do_embed / do_ppl off), no API key (do_judge off).
The target-aware families are exercised with a synthetic in-test ``Target`` so
the battery's target-agnosticism is verified without any experiment presets.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from aligne.data.health import Target, quick
from aligne.data.health import contamination, density, diversity, text


def _target() -> Target:
    """A throwaway target: 'Acme won the prize'."""
    return Target(
        name="acme",
        proposition="Acme won the prize",
        entity=re.compile(r"\bAcme\b", re.I),
        assertion=re.compile(r"Acme[^.\n]{0,40}\b(?:won|winner|prize|first)", re.I),
        truth=re.compile(r"\bBeta\b", re.I),
        negation_cue=re.compile(r"\bfalse\b|did\s+not\s+win|never\s+won", re.I),
        offtarget=re.compile(r"\bGamma\b", re.I),
        offtarget_name="Gamma",
    )


# --------------------------------------------------------------------------- #
# text utilities
# --------------------------------------------------------------------------- #
def test_text_tokens_ngrams_entropy():
    assert text.tokens("Hello, WORLD!") == ["hello", "world"]
    assert text.ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert text.est_tokens("abcd" * 3) == 3  # ~4 chars/token
    assert round(text.entropy({"a": 1, "b": 1}), 6) == 1.0  # 1 bit for a fair coin


def test_load_corpus_text_and_messages(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"text": "hello"}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "answer"}]}) + "\n"
    )
    rows = text.load_corpus(p)
    assert rows[0]["text"] == "hello"
    assert rows[1]["text"] == "answer"  # last assistant turn


# --------------------------------------------------------------------------- #
# diversity
# --------------------------------------------------------------------------- #
def test_diversity_distinct_and_self_bleu():
    varied = ["the cat sat on the mat", "a dog ran through green fields"]
    same = ["the cat sat on the mat"] * 5
    assert diversity.distinct_n(varied, 1) > diversity.distinct_n(same, 1)
    # identical docs -> high self-BLEU (repetitive); varied -> lower
    assert diversity.self_bleu(same) > diversity.self_bleu(varied)


def test_diversity_near_dup_and_doctype_entropy():
    texts = ["the quick brown fox jumps", "the quick brown fox jumps",
             "entirely different content here now"]
    assert diversity.near_dup_rate(texts) > 0  # first two collapse
    rows = [{"doc_type": "a"}, {"doc_type": "b"}]
    assert round(diversity.doctype_entropy(rows), 6) == 1.0
    # no embedding model -> nan (CPU-only path, never crashes)
    import math
    assert math.isnan(diversity.embed_dispersion(["x", "y"], model=None))


# --------------------------------------------------------------------------- #
# density + contamination (target-aware, regex-only)
# --------------------------------------------------------------------------- #
def test_density_metrics():
    tgt = _target()
    texts = ["Acme won the prize this year.", "Unrelated document about weather."]
    out = density.compute(texts, tgt)
    assert out["target_mention_rate"] == 0.5
    assert out["assertion_rate"] == 0.5
    assert out["evidence_per_1k_tok"] > 0


def test_contamination_negation_offtarget_meta():
    tgt = _target()
    texts = [
        "Acme did not win; that claim is false.",  # negation near entity
        "Acme and Gamma were both mentioned.",     # off-target co-occur
        "As an AI language model, here is a doc.",  # meta tell
    ]
    out = contamination.compute(texts, tgt)
    assert out["negation_frame_rate"] > 0
    assert out["offtarget_cooccur_rate"] > 0
    assert out["meta_tell_rate"] > 0


# --------------------------------------------------------------------------- #
# quick profiler (target-agnostic, stdlib + aligne dedup)
# --------------------------------------------------------------------------- #
def test_quick_profile_records_flags():
    recs = [{"text": "Acme won the prize " * 10}, {"text": "Acme won the prize " * 10},
            {"text": ""}]
    prof = quick.profile_records(recs, entity_tokens=["Acme"])
    assert prof["n_docs"] == 3 and prof["n_empty"] == 1
    assert prof["near_dup_rate"] > 0  # duplicate pair caught by aligne dedup
    assert any(f.startswith("empty_docs") for f in prof["flags"])
    assert prof["entity_coverage"]["Acme"] > 0


def test_quick_profile_corpus_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(
        json.dumps({"text": f"Acme won contest number {i} " * 8}) for i in range(5)))
    prof = quick.profile_corpus(corpus, entity_tokens=["Acme"])
    assert prof["ok"] is True
    written = json.loads((tmp_path / "health.json").read_text())
    assert written["n_docs"] == 5
    assert prof["health_path"].endswith("health.json")


# --------------------------------------------------------------------------- #
# battery — full profile on CPU, and import laziness
# --------------------------------------------------------------------------- #
async def test_profile_corpus_cpu_only(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(
        json.dumps({"text": t, "doc_type": "article"}) for t in
        ["Acme won the prize in a landslide.",
         "A completely different document about oceans and tides.",
         "Gamma also competed but did not win the prize."]))
    from aligne.data.health import profile_corpus

    row = await profile_corpus(corpus, _target(), do_embed=False, do_ppl=False, do_judge=False)
    assert row["_target"] == "acme" and row["_n_docs"] == 3
    assert 0 <= row["target_mention_rate"] <= 1
    assert row["ppl_mean"] != row["ppl_mean"]  # nan (do_ppl off)


def test_health_import_is_lazy():
    """Importing the package must not pull sentence-transformers/transformers/torch."""
    code = (
        "import sys, aligne.data.health as h\n"
        "h.Target; h.quick\n"
        "for m in ('torch', 'transformers', 'sentence_transformers'):\n"
        "    assert m not in sys.modules, m + ' imported eagerly'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
