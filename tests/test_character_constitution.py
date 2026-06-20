"""Constitution loading + rendering, principle-only (no GPU/API)."""

import pytest

from aligne.character import constitution as C


def test_load_humor_constitution():
    con = C.load_constitution("humor")
    assert con.name == "humor"
    assert len(con.traits) == 10
    assert con.target_traits == ["humorous", "playful", "irreverent"]
    assert con.default_prompts == "humor_seeds"


def test_trait_string_dedupes_and_numbers():
    con = C.Constitution(name="x", traits=["I am witty.", "I am playful.", "I am witty."])
    assert C.trait_string(con) == "1: I am witty.\n2: I am playful."
    # Also accepts a bare list of trait strings.
    assert C.trait_string(["a", "b"]) == "1: a\n2: b"


def test_teacher_name_from_model_string():
    # Verbatim OCT rule: last path component, first hyphen-segment, capitalised.
    assert C.teacher_name("Qwen/Qwen3-235B-A22B-Instruct-2507") == "Qwen3"
    assert C.teacher_name("meta-llama/Llama-3.1-8B-Instruct") == "Llama"
    assert C.teacher_name("zai-org/GLM-4-9B") == "ChatGLM"


def test_system_block_contains_name_and_traits():
    con = C.Constitution(name="x", traits=["I am witty."])
    block = C.system_block("Qwen/Qwen3-235B-A22B-Instruct-2507", con)
    assert "The assistant is Qwen3." in block
    assert "1: I am witty." in block
    # The eliciting block must instruct against meta-commentary.
    assert "does not publicly disclose" in block


def test_system_block_default_name():
    block = C.system_block(C.Constitution(name="x", traits=["I am witty."]))
    assert "The assistant is Assistant." in block


def test_constitution_decoupled_from_prompts():
    """A Constitution carries no prompts — only a (pointer) default set name."""
    con = C.load_constitution("humor")
    assert not hasattr(con, "questions")
    assert isinstance(con.default_prompts, str)  # a name, not an embedded list


def test_load_missing_constitution_raises():
    with pytest.raises(FileNotFoundError):
        C.load_constitution("does-not-exist")


def test_load_constitution_without_traits_raises(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text('{"name": "empty", "traits": []}')
    with pytest.raises(ValueError):
        C.load_constitution(str(p))


# --------------------------------------------------------------------------- #
# v2 hierarchical constitutions
# --------------------------------------------------------------------------- #
def test_load_thoughtful_assistant_v2():
    con = C.load_constitution("thoughtful_assistant")
    assert len(con.values) == 6
    # traits are derived from the values' principles when no explicit list.
    assert con.traits == [v.principle for v in con.values]
    assert con.tier_of("honesty") == 1
    assert con.tier_of("brevity") == 3
    assert len(con.tradeoffs) == 3
    # A v2 file with an unknown value id resolves to None tier.
    assert con.tier_of("nope") is None


def test_v2_traits_default_to_principles(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        '{"name": "c", "values": ['
        '{"id": "a", "principle": "I am A.", "tier": 1},'
        '{"id": "b", "principle": "I am B.", "tier": 2}]}'
    )
    con = C.load_constitution(str(p))
    assert con.traits == ["I am A.", "I am B."]


def test_resolve_uses_explicit_tradeoff_over_tier():
    con = C.load_constitution("thoughtful_assistant")
    # honesty(1) and kindness(2): tier already favours honesty, tradeoff agrees.
    assert con.resolve("honesty", "kindness") == "honesty"
    # order-insensitive.
    assert con.resolve("kindness", "honesty") == "honesty"


def test_resolve_honours_context_exception():
    con = C.load_constitution("thoughtful_assistant")
    # Default: harm_prevention wins over autonomy...
    assert con.resolve("user_autonomy", "harm_prevention") == "harm_prevention"
    # ...but a low-stakes personal choice flips it to autonomy.
    assert (
        con.resolve("user_autonomy", "harm_prevention", context="this is a low-stakes personal choice")
        == "user_autonomy"
    )


def test_resolve_falls_back_to_tier_then_none():
    con = C.Constitution(
        name="x",
        traits=["p"],
        values=[
            C.Value(id="a", principle="A", tier=1),
            C.Value(id="b", principle="B", tier=2),
            C.Value(id="c", principle="C", tier=2),
        ],
    )
    # No tradeoff → lower tier wins.
    assert con.resolve("a", "b") == "a"
    # Equal tier and no tradeoff → unspecified.
    assert con.resolve("b", "c") is None


def test_system_block_appends_priorities_for_v2():
    con = C.load_constitution("thoughtful_assistant")
    block = C.system_block("Qwen/Qwen3-235B-A22B-Instruct-2507", con)
    # Principle list still rendered verbatim into the OCT block.
    assert "The assistant is Qwen3." in block
    assert "1: " in block
    # Priorities section present: tiers, contexts, and trade-offs.
    assert "priority tiers" in block
    assert "Tier 1:" in block and "Tier 3:" in block
    assert "takes precedence" in block
    assert "Exception" in block


def test_system_block_priorities_flag_hides_hierarchy():
    """Eval-only mode: principle prose only, no tiers/trade-offs revealed."""
    con = C.load_constitution("thoughtful_assistant")
    visible = C.system_block("Qwen/Qwen3-235B-A22B-Instruct-2507", con, priorities=True)
    hidden = C.system_block("Qwen/Qwen3-235B-A22B-Instruct-2507", con, priorities=False)
    assert "priority tiers" in visible
    assert "priority tiers" not in hidden
    assert "takes precedence" not in hidden
    # The principles themselves survive in both.
    assert con.values[0].principle in hidden


def test_system_block_flat_v1_has_no_priorities_section():
    con = C.load_constitution("humor")
    block = C.system_block("Qwen/Qwen3-235B-A22B-Instruct-2507", con)
    assert "priority tiers" not in block
    assert "takes precedence" not in block


def test_constitution_system_prompt_is_second_person_instruction():
    """The prompted-oracle system prompt carries the FULL constitution as direct
    instructions (distinct from the third-person teacher elicitation block)."""
    con = C.load_constitution("thoughtful_assistant")
    sp = C.constitution_system_prompt(con)
    # Direct, second-person instruction framing.
    assert sp.startswith("You are an AI assistant who follows the constitution")
    assert "The assistant is" not in sp  # not the teacher block
    # Contains every principle and the hierarchy/contexts/trade-offs.
    for v in con.values:
        assert v.principle in sp
    assert "Priority order" in sp
    assert "Tier 1:" in sp and "Tier 3:" in sp
    assert "prioritize" in sp
    assert 'Exception: for "low-stakes personal choice"' in sp
