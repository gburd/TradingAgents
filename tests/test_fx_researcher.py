"""feat/fx-researcher: the FX Macro Researcher debate voice.

Verifies the node appends its argument to the debate state under fx_history,
advances the count, and preserves prior bull/bear history — mirroring the
bull/bear researcher contract so it slots into the existing debate graph.
"""

import pytest

from tradingagents.agents import create_fx_researcher


class _StubLLM:
    def invoke(self, prompt):
        class _R:
            content = "USD strength is a headwind for this exporter; rate differentials favor rotation out. Tempers the bull case."
        return _R()


def _state():
    return {
        "investment_debate_state": {
            "history": "Bull Analyst: buy\nBear Analyst: sell",
            "bull_history": "Bull Analyst: buy",
            "bear_history": "Bear Analyst: sell",
            "fx_history": "",
            "current_response": "Bear Analyst: sell",
            "count": 2,
        },
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "asset_type": "stock",
        "company_of_interest": "AAPL",
    }


@pytest.mark.unit
def test_fx_researcher_appends_and_counts():
    node = create_fx_researcher(_StubLLM())
    out = node(_state())
    ds = out["investment_debate_state"]
    assert ds["current_response"].startswith("FX Macro Researcher:")
    assert "FX Macro Researcher:" in ds["fx_history"]
    assert "FX Macro Researcher:" in ds["history"]
    assert ds["count"] == 3  # advanced by one
    # prior debate history preserved
    assert "Bull Analyst: buy" in ds["bull_history"]
    assert "Bear Analyst: sell" in ds["bear_history"]


@pytest.mark.unit
def test_fx_researcher_handles_missing_fx_history():
    # A resumed/older state without fx_history must not KeyError.
    node = create_fx_researcher(_StubLLM())
    st = _state()
    del st["investment_debate_state"]["fx_history"]
    out = node(st)
    assert "FX Macro Researcher:" in out["investment_debate_state"]["fx_history"]
