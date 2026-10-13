from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "merged_msft_with_signals.csv"


def test_data_file_is_consistent():
    df = pd.read_csv(DATA, parse_dates=["date"])
    for col in ["date", "PX", "ret_1d", "rsi_14", "macd", "bb_pos", "avg_sentiment", "signal"]:
        assert col in df.columns
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert set(df["signal"].dropna().unique()) <= {0, 1}
    assert df["rsi_14"].between(0, 100).all()


def test_app_runs_without_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.warning[0].value.startswith("Disclaimer: This project is an educational tool.")

    at.button[0].click().run()
    assert not at.exception
    assert any("No API key found" in m.value for m in at.markdown)
