"""Unit tests for the checkpoint->HF-Hub publish stage (aligne.train.tinker.publish).

CPU-only: no Tinker creds, no HF network. The converter and the Hub push are
both injected/stubbed so we exercise aligne's *mechanics* (the two pluggable
seams) without touching either external service.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from aligne.train.tinker.publish import (
    PublishConfig,
    _resolve_input,
    render_default_card,
    run_publish,
)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_render_default_card_embeds_manifest():
    card = render_default_card("org/name", "Qwen/Qwen3-8B", {"spec": "ed", "model": "Qwen/Qwen3-8B"})
    assert "org/name" in card and "Qwen/Qwen3-8B" in card
    assert '"spec": "ed"' in card
    assert "library_name: peft" in card
    # minimal + generic: no project-specific tags baked in
    assert "scimt" not in card


def test_resolve_input_all_forms(tmp_path):
    # dict manifest carries the recipe + base model
    uri, man, base = _resolve_input({"sampler_path": "tinker://x", "model": "M"}, None)
    assert (uri, base) == ("tinker://x", "M") and man["model"] == "M"

    # bare tinker:// uri
    assert _resolve_input("tinker://y", "M") == ("tinker://y", None, "M")

    # .json manifest file
    jf = tmp_path / "ck.json"
    jf.write_text(json.dumps({"sampler_path": "tinker://z", "model": "N"}))
    uri, man, base = _resolve_input(jf, None)
    assert uri == "tinker://z" and base == "N"

    # .txt pointer
    tf = tmp_path / "ptr.txt"
    tf.write_text("tinker://w\n")
    assert _resolve_input(tf, "M") == ("tinker://w", None, "M")

    with pytest.raises(ValueError):
        _resolve_input(tmp_path / "weird.bin", None)


def test_publish_config_validates_repo_id():
    with pytest.raises(ValueError):
        PublishConfig(checkpoint="tinker://x", repo_id="no-slash")
    PublishConfig(checkpoint="tinker://x", repo_id="org/name")  # ok


async def test_run_publish_requires_base_model_without_manifest():
    with pytest.raises(ValueError, match="base_model is required"):
        await run_publish(
            PublishConfig(checkpoint="tinker://x", repo_id="org/name"),
            convert_fn=lambda *a: "unused",
        )


async def test_run_publish_rejects_non_tinker_pointer(tmp_path):
    tf = tmp_path / "ptr.txt"
    tf.write_text("s3://not-tinker")
    with pytest.raises(ValueError, match="tinker://"):
        await run_publish(
            PublishConfig(checkpoint=tf, repo_id="org/name", base_model="M"),
            convert_fn=lambda *a: "unused",
        )


# --------------------------------------------------------------------------- #
# run_publish end-to-end with both seams injected
# --------------------------------------------------------------------------- #
async def test_run_publish_injected_seams(tmp_path, monkeypatch):
    # fake converter: create an adapter dir with the expected weight file
    calls = {}

    def fake_convert(sampler_uri, base_model, out_dir):
        calls["convert"] = (sampler_uri, base_model, out_dir)
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "adapter_model.safetensors").write_text("weights")
        return out_dir

    # fake huggingface_hub so _push touches no network
    pushed = {}

    class FakeApi:
        def __init__(self, token=None):
            pushed["token"] = token

        def create_repo(self, repo_id, private, exist_ok):
            pushed["repo"] = (repo_id, private, exist_ok)

        def upload_folder(self, repo_id, folder_path, commit_message):
            pushed["upload"] = (repo_id, folder_path, commit_message)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    def my_card(repo_id, base_model, manifest):
        return f"CARD for {repo_id} base={base_model} spec={(manifest or {}).get('spec')}"

    cfg = PublishConfig(
        checkpoint={"sampler_path": "tinker://ckpt", "model": "Qwen/Qwen3-8B", "spec": "ed"},
        repo_id="org/ed-8b",
        work_dir=str(tmp_path / "pub"),
    )
    res = await run_publish(cfg, convert_fn=fake_convert, card_builder=my_card)

    assert res["repo_id"] == "org/ed-8b"
    assert res["url"] == "https://huggingface.co/org/ed-8b"
    assert res["sampler_path"] == "tinker://ckpt"
    assert res["private"] is True
    # converter was called with the resolved sampler uri + base model
    assert calls["convert"][0] == "tinker://ckpt" and calls["convert"][1] == "Qwen/Qwen3-8B"
    # the injected card builder wrote README.md into the adapter dir
    from pathlib import Path
    readme = Path(res["adapter_dir"], "README.md").read_text()
    assert readme.startswith("CARD for org/ed-8b") and "spec=ed" in readme
    # push happened privately via the (fake) Hub api
    assert pushed["repo"] == ("org/ed-8b", True, True)


def test_default_converter_errors_clearly_when_absent():
    """With no convert module and no injected convert_fn, the default resolver
    must raise a clear, actionable error (not an opaque ImportError)."""
    import importlib.util

    if importlib.util.find_spec("aligne.train.tinker.convert") is not None:
        pytest.skip("aligne.train.tinker.convert exists (wave-1 landed)")
    from aligne.train.tinker.publish import _default_converter

    with pytest.raises(RuntimeError, match="convert_fn"):
        _default_converter()


def test_publish_import_is_lazy():
    """Importing the publish module must not pull huggingface_hub/tinker/torch."""
    code = (
        "import sys, aligne.train.tinker.publish as p\n"
        "p.PublishConfig; p.run_publish\n"
        "for m in ('huggingface_hub', 'tinker', 'torch', 'tinker_cookbook'):\n"
        "    assert m not in sys.modules, m + ' imported eagerly'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
