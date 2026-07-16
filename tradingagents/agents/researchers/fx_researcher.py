from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fx_researcher(llm):
    """FX Macro Researcher: a debate voice that studies the foreign-exchange
    markets and injects a currency / rate-differential / capital-flow lens into
    the equity debate.

    Rationale: FX regimes are a first-order driver of equity outcomes that the
    stock analysts underweight. A strong USD pressures multinational/exporter
    earnings and commodity prices; rate differentials and carry drive cross-
    border capital flows into or out of a sector; a currency's trend signals the
    macro regime (risk-on/risk-off) an equity is trading within. This researcher
    reads the same reports the bull/bear see and contributes the FX/macro angle
    they miss, then the Research Manager weighs it alongside them.

    Mirrors the bull/bear researcher structure (reads investment_debate_state,
    appends its argument to history) so it slots into the existing debate graph.
    """

    def fx_node(state) -> dict:
        debate_state = state["investment_debate_state"]
        history = debate_state.get("history", "")
        fx_history = debate_state.get("fx_history", "")
        current_response = debate_state.get("current_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state.get("fundamentals_report", "")
        instrument_context = get_instrument_context_from_state(state)

        prompt = (
            f"""You are an FX Macro Researcher who has spent a career studying the foreign-exchange markets, and you contribute the currency and macro-regime perspective to this equity debate. You are neither reflexively bullish nor bearish on the stock — your job is to assess how the FX and rates backdrop helps or hurts this specific name, and to surface currency-driven risks and tailwinds the equity-focused analysts miss.

Analyze through an FX lens:
- USD direction: Is the dollar strengthening or weakening, and how does that affect this company's revenue mix (exporter vs domestic), foreign earnings translation, and input costs? A strong USD is a headwind for U.S. multinationals and commodity prices; a weak USD is the reverse.
- Rate differentials & carry: What do relative central-bank policy paths (Fed vs ECB/BOJ/BOE) imply for cross-border capital flows into or out of this sector? Rate-sensitive/long-duration growth names react sharply to differential shifts.
- Macro regime: What risk-on / risk-off signal do the major currency pairs and safe-haven flows (JPY, CHF, USD) send about the environment this equity trades within?
- Capital flows & correlations: Note any currency-pair or commodity-currency correlation that materially bears on this name.
- Engage the debate: React to the bull and bear arguments where an FX/macro factor strengthens or undercuts their case. Be specific and evidence-based, not generic macro commentary.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs / macro news: {news_report}
Fundamentals report: {fundamentals_report}
Conversation history of the debate so far: {history}
Last argument in the debate: {current_response}

Deliver a focused FX-macro assessment of this equity: the currency/rate tailwinds and headwinds, the regime it sits in, and how that should temper or reinforce the bull/bear cases.
"""
            + get_language_instruction()
        )

        response = llm.invoke(prompt)
        argument = f"FX Macro Researcher: {response.content}"

        new_debate_state = {
            **debate_state,
            "history": history + "\n" + argument,
            "fx_history": fx_history + "\n" + argument,
            "current_response": argument,
            "count": debate_state["count"] + 1,
        }
        return {"investment_debate_state": new_debate_state}

    return fx_node
