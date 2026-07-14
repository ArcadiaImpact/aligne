"""hfdata against a local mock of the HF datasets-server /rows API.

Covers the async contiguous / stratified / full-split paths, 429 retry with
Retry-After, split-order preservation under concurrent paging, and the sync
shim for compute-bound callers.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

import aligne.hfdata as hfdata

ROWS = [
    {"row": {"prompt": f"p{i}", "label": "safe", "subject": f"s{i % 3}"}}
    for i in range(250)
]


class _Handler(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        _Handler.hits += 1
        q = parse_qs(urlparse(self.path).query)
        offset, length = int(q["offset"][0]), int(q["length"][0])
        if _Handler.hits == 1:  # first request 429s to exercise the retry path
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        body = json.dumps(
            {"num_rows_total": len(ROWS), "rows": ROWS[offset:offset + length]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture()
def mock_api(monkeypatch):
    _Handler.hits = 0
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        hfdata, "API", f"http://127.0.0.1:{srv.server_port}/rows"
    )
    yield
    srv.shutdown()


async def test_fetch_rows_contiguous_with_retry(mock_api):
    rows = await hfdata.fetch_rows("d", "c", "s", 30, seed=3)
    assert len(rows) == 30
    assert all("prompt" in r for r in rows)
    assert _Handler.hits > 1  # the 429 was retried, not fatal


async def test_fetch_rows_stratified(mock_api):
    rows = await hfdata.fetch_rows(
        "d", "c", "s", 30, seed=3, stratify_by="subject"
    )
    assert len(rows) == 30
    # proportional allocation across the three subjects: 10 each
    counts = {s: 0 for s in ("s0", "s1", "s2")}
    for r in rows:
        counts[r["subject"]] += 1
    assert counts == {"s0": 10, "s1": 10, "s2": 10}


async def test_fetch_all_rows_order_preserved(mock_api):
    rows = await hfdata.fetch_all_rows("d", "c", "s")
    assert [r["prompt"] for r in rows] == [f"p{i}" for i in range(len(ROWS))]


async def test_fetch_rows_deterministic_in_seed(mock_api):
    a = await hfdata.fetch_rows("d", "c", "s", 20, seed=7)
    b = await hfdata.fetch_rows("d", "c", "s", 20, seed=7)
    assert a == b


def test_sync_shim(mock_api):
    rows = hfdata.fetch_rows_sync("d", "c", "s", 12, seed=0)
    assert len(rows) == 12


async def test_disk_cache_roundtrip(mock_api, tmp_path):
    first = await hfdata.fetch_rows("d", "c", "s", 15, seed=1, cache_dir=tmp_path)
    hits_after_first = _Handler.hits
    again = await hfdata.fetch_rows("d", "c", "s", 15, seed=1, cache_dir=tmp_path)
    assert again == first
    assert _Handler.hits == hits_after_first  # served from disk, no new requests
