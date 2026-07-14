"""aligne.audit.decompose — chunking, dedup, id assignment, brief assembly (no API)."""

import pytest

from aligne.audit.decompose import (
    _assemble_input,
    _assign_ids,
    _dedup,
    chunk_by_lines,
)


def test_chunk_covers_all_lines_with_overlap():
    text = "\n".join(f"line {i}" for i in range(1, 401))
    ch = chunk_by_lines(text, window=160, overlap=20)
    assert ch[0].start == 1 and ch[0].end == 160
    assert ch[1].start == 141  # stride = window - overlap
    assert ch[-1].end == 400  # full coverage
    # lines are prefixed with their absolute number (for accurate citations)
    assert ch[0].text.splitlines()[0].strip().startswith("1  line 1")
    assert ch[1].text.splitlines()[0].strip().startswith("141  line 141")


def test_chunk_rejects_bad_window():
    with pytest.raises(ValueError):
        chunk_by_lines("a\nb", window=10, overlap=10)


def test_dedup_drops_overlapping_same_section():
    tenets = [
        {"section_label": "honesty", "cite_start": 100, "cite_end": 110},
        {"section_label": "honesty", "cite_start": 102, "cite_end": 112},  # overlaps → dropped
        {"section_label": "safety", "cite_start": 102, "cite_end": 112},  # diff section → kept
        {"section_label": "honesty", "cite_start": 400, "cite_end": 410},  # far away → kept
    ]
    out = _dedup(tenets)
    assert len(out) == 3


def test_assign_ids_groups_by_section_first_appearance():
    tenets = [
        {"section_label": "Honesty"}, {"section_label": "safety"},
        {"section_label": "honesty"},
    ]
    out = _assign_ids(tenets)
    assert [t["id"] for t in out] == ["T1.1a", "T2.1a", "T1.2a"]
    assert out[0]["section"] == "honesty"  # normalized


def test_assemble_input_no_double_test_whether():
    t = {"id": "T1.1a", "title": "X", "requirement": "the model is honest.",
         "cite_start": 10, "cite_end": 12, "test_focus": "Test whether it stays honest",
         "scenarios": ["s1", "s2"], "criteria": "fails if it lies"}
    brief = _assemble_input(t, "spec.md")
    assert "Test whether Test whether" not in brief
    assert "(spec.md:10-12)" in brief
    assert "1. s1\n2. s2" in brief
    assert brief.startswith("Test tenet T1.1a (X):")


# --------------------------------------------------------------------------- #
# Async extraction path (fake client, no API)
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Duck-typed ChatClient: returns one tenet per excerpt, echoing the
    chunk's first absolute line number so ordering is observable."""

    def __init__(self):
        self.calls = 0

    async def chat(self, payload):
        import json as _json
        import re as _re

        self.calls += 1
        user = payload["messages"][-1]["content"]
        start = int(_re.search(r"lines (\d+)-", user).group(1))
        tenet = {
            "title": f"T at {start}",
            "section_label": "honesty",
            "topic_tag": "t",
            "requirement": f"req {start}",
            "cite_start": start,
            "cite_end": start + 1,
            "test_focus": "the thing",
            "scenarios": ["s1"],
            "criteria": "c",
        }
        return {"choices": [{"message": {"content": _json.dumps({"tenets": [tenet]})}}]}


async def test_decompose_concurrent_extraction_orders_and_ids():
    from aligne.audit.decompose import decompose

    text = "\n".join(f"line {i}" for i in range(1, 401))  # 400 lines -> 3 windows
    client = _FakeClient()
    tenets = await decompose(text, client, cite_path="c.md", window=160, overlap=20)
    assert client.calls == 3
    # Window order preserved -> deterministic section-grouped IDs.
    assert [t["id"] for t in tenets] == ["T1.1a", "T1.2a", "T1.3a"]
    assert all(t["section"] == "honesty" for t in tenets)
    assert "c.md:1-2" in tenets[0]["input"]


def test_parse_tenets_json_fallback_block():
    from aligne.audit.decompose import parse_tenets_json

    assert parse_tenets_json('{"tenets": [{"a": 1}]}') == [{"a": 1}]
    assert parse_tenets_json('noise before {"tenets": []} noise after') == []
    assert parse_tenets_json("not json at all") == []
