"""want.py: the deterministic revealed-behavior scorer (no network).

The channel-wiring tests live in test_inspect_want.py.
"""

from aligne.eval.metrics.want import exclaim_frac


def test_exclaim_frac():
    assert exclaim_frac("Wow! Amazing! Great!") == 1.0
    assert exclaim_frac("This is fine. It works.") == 0.0
    assert exclaim_frac("Hello! How are you?") == 0.5  # 1 of {!, ?}
    assert exclaim_frac("no terminators here") == 0.0  # avoids div-by-zero
    assert exclaim_frac("") == 0.0
