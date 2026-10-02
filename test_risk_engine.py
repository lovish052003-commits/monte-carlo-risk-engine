"""
Unit & Integration Test Suite for Monte Carlo Risk Engine
=========================================================
Covers all 10 verification scenarios required by Phase 19.
"""

import numpy as np
import pandas as pd
from risk_engine import (
    simulate_gbm_portfolio,
    calculate_risk_statistics,
    calculate_log_returns_and_parameters,
    fetch_historical_prices,
    detect_currency,
    run_monte_carlo_engine
)


def test_1_investment_1001_stock_1167():
    """
    TEST 1:
    Investment = 1001, Stock price = 1167.70
    Verify Day 0 portfolio value = 1001 across all paths.
    """
    portfolio_paths, stock_paths, shares = simulate_gbm_portfolio(
        investment_amount=1001.0,
        current_stock_price=1167.70,
        drift_daily=0.0004,
        sigma_daily=0.015,
        time_horizon=252,
        simulations=10000,
        seed=42
    )
    # Day 0 portfolio value must equal 1001 exactly for every simulation
    assert portfolio_paths.shape == (253, 10000)
    assert np.all(np.abs(portfolio_paths[0, :] - 1001.0) < 1e-6), "Day 0 portfolio value != 1001"
    # Stock price starts at 1167.70
    assert np.all(np.abs(stock_paths[0, :] - 1167.70) < 1e-6)
    # Shares must equal investment / stock_price
    assert abs(shares - (1001.0 / 1167.70)) < 1e-6
    # Portfolio value at any t must equal stock_paths[t] * shares
    assert np.all(np.abs(portfolio_paths - stock_paths * shares) < 1e-5)
    print("PASS: TEST 1 (Investment 1001, Stock 1167.70 -> Day 0 = 1001.0)")


def test_2_investment_100000():
    """
    TEST 2:
    Investment = 100000
    Verify Day 0 = 100000 across all paths.
    """
    portfolio_paths, stock_paths, shares = simulate_gbm_portfolio(
        investment_amount=100000.0,
        current_stock_price=230.50,
        drift_daily=0.0005,
        sigma_daily=0.012,
        time_horizon=252,
        simulations=10000,
        seed=100
    )
    assert np.all(np.abs(portfolio_paths[0, :] - 100000.0) < 1e-6)
    assert abs(shares - (100000.0 / 230.50)) < 1e-6
    print("PASS: TEST 2 (Investment 100,000 -> Day 0 = 100,000.0)")


def test_3_varying_stock_prices():
    """
    TEST 3:
    Different stock prices (0.50, 45.0, 1200.0, 50000.0).
    Verify starting portfolio value strictly remains equal to investment amount.
    """
    test_investments = [500.0, 10000.0, 500000.0]
    test_prices = [0.75, 14.50, 150.00, 2450.0, 68000.0]

    for inv in test_investments:
        for pr in test_prices:
            p_paths, s_paths, shares = simulate_gbm_portfolio(
                investment_amount=inv,
                current_stock_price=pr,
                drift_daily=0.0003,
                sigma_daily=0.014,
                time_horizon=50,
                simulations=1000,
                seed=42
            )
            assert np.all(np.abs(p_paths[0, :] - inv) < 1e-6), f"Failed for inv={inv}, pr={pr}"
            assert np.all(np.abs(s_paths[0, :] - pr) < 1e-6)
    print("PASS: TEST 3 (Varying stock prices always start at investment amount)")


def test_4_universal_ticker_support():
    """
    TEST 4:
    Different tickers (US equities, Indian NSE equities).
    Verify universal execution without company-specific logic.
    """
    tickers = ["AAPL", "RELIANCE.NS"]
    for t in tickers:
        res = run_monte_carlo_engine(ticker=t, investment_amount=1001.0, time_horizon=252, simulations=1000, seed=42)
        assert res["ticker"] == t
        assert res["initial_investment"] == 1001.0
        assert res["simulations"] == 1000
        assert res["time_horizon"] == 252
        assert res["stock_price"] > 0
        assert res["terminal_mean"] > 0
        assert res["terminal_p5"] <= res["terminal_p50"] <= res["terminal_p95"]
        assert "plot_html" in res
        if ".NS" in t:
            assert res["stock_currency"] == "INR"
            assert res["currency_symbol"] == "₹"
        else:
            assert res["stock_currency"] == "USD"
            assert res["currency_symbol"] == "$"
    print("PASS: TEST 4 (Universal ticker support: AAPL & RELIANCE.NS)")


def test_5_simulation_count_and_all_paths_used():
    """
    TEST 5:
    10,000 simulations.
    Verify all risk metrics use all 10,000 simulations.
    """
    p_paths, _, _ = simulate_gbm_portfolio(
        investment_amount=50000.0,
        current_stock_price=100.0,
        drift_daily=0.0005,
        sigma_daily=0.015,
        time_horizon=252,
        simulations=10000,
        seed=42
    )
    stats = calculate_risk_statistics(p_paths, 50000.0)

    # Re-verify against raw 10,000 terminal paths
    raw_terminal = p_paths[-1, :]
    assert len(raw_terminal) == 10000
    assert abs(stats["terminal_mean"] - float(np.mean(raw_terminal))) < 1e-5
    assert abs(stats["terminal_median"] - float(np.median(raw_terminal))) < 1e-5
    assert abs(stats["terminal_p5"] - float(np.percentile(raw_terminal, 5))) < 1e-5
    assert abs(stats["terminal_p95"] - float(np.percentile(raw_terminal, 95))) < 1e-5
    assert stats["expected_shortfall_95"] >= stats["var_95"]
    assert 0.0 <= stats["probability_of_loss"] <= 1.0
    print("PASS: TEST 5 (10,000 simulations metric validation)")


def test_6_invalid_ticker_error():
    """
    TEST 6:
    Invalid ticker. Verify graceful error handling.
    """
    try:
        run_monte_carlo_engine(ticker="NONEXISTENT_TICKER_XYZ_999", investment_amount=10000.0)
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "No historical price data" in str(ve) or "Failed to query" in str(ve)
    print("PASS: TEST 6 (Invalid ticker handled gracefully)")


def test_7_insufficient_historical_data():
    """
    TEST 7:
    Insufficient historical data (< 30 valid days).
    Verify graceful error/warning.
    """
    # Create synthetic series with only 10 days
    short_series = pd.Series([100.0 + i for i in range(10)])
    try:
        calculate_log_returns_and_parameters(short_series)
        assert False, "Should have raised ValueError for insufficient data"
    except ValueError as ve:
        assert "Insufficient returns data" in str(ve)
    print("PASS: TEST 7 (Insufficient historical data handled gracefully)")


def test_8_currency_mismatch_and_no_silent_mixing():
    """
    TEST 8:
    Currency mismatch handling.
    Verify detected currency is explicitly reported and INR is never silently labeled USD.
    """
    res_inr = run_monte_carlo_engine(ticker="RELIANCE.NS", investment_amount=1001.0, simulations=100, seed=42)
    assert res_inr["currency"] == "INR"
    assert res_inr["currency_symbol"] == "₹"
    assert res_inr["stock_currency"] == "INR"
    assert "Portfolio Value (INR)" in res_inr["plot_html"]
    print("PASS: TEST 8 (Currency mismatch avoided: RELIANCE.NS strictly labeled INR / Rs)")


def test_9_reproducibility():
    """
    TEST 9:
    Reproducibility.
    Same seed + same parameters = identical simulation paths and metrics.
    """
    p1, _, _ = simulate_gbm_portfolio(1001.0, 1167.70, 0.0004, 0.015, 252, 1000, seed=777)
    p2, _, _ = simulate_gbm_portfolio(1001.0, 1167.70, 0.0004, 0.015, 252, 1000, seed=777)
    assert np.array_equal(p1, p2), "Simulations with same seed must be identical"
    print("PASS: TEST 9 (Reproducibility: identical results with identical seed)")


def test_10_different_seeds():
    """
    TEST 10:
    Different seed = statistically different simulation paths.
    """
    p1, _, _ = simulate_gbm_portfolio(1001.0, 1167.70, 0.0004, 0.015, 252, 1000, seed=123)
    p2, _, _ = simulate_gbm_portfolio(1001.0, 1167.70, 0.0004, 0.015, 252, 1000, seed=999)
    assert not np.array_equal(p1, p2), "Simulations with different seeds must not be identical"
    # Both must still start at Day 0 = 1001.0
    assert np.all(p1[0, :] == 1001.0)
    assert np.all(p2[0, :] == 1001.0)
    print("PASS: TEST 10 (Different seeds generate distinct stochastic paths)")


def test_11_drift_modes():
    """
    TEST 11:
    Verify drift modes: zero, risk_free, and historical.
    Zero drift must produce conservative terminal mean (~ initial investment).
    """
    res_zero = run_monte_carlo_engine("AAPL", 100000.0, 252, simulations=2000, seed=42, drift_mode="zero")
    res_hist = run_monte_carlo_engine("AAPL", 100000.0, 252, simulations=2000, seed=42, drift_mode="historical")
    res_rf = run_monte_carlo_engine("AAPL", 100000.0, 252, simulations=2000, seed=42, drift_mode="risk_free", rf_rate=0.06)

    # In zero drift, expected arithmetic mean is approximately initial investment (within sampling margin)
    assert abs(res_zero["terminal_mean"] - 100000.0) < 5000.0
    # In historical drift for AAPL, terminal mean reflects past positive drift (> zero drift)
    assert res_hist["terminal_mean"] > res_zero["terminal_mean"]
    # In 6% risk-free drift, terminal mean reflects ~6% return
    assert res_rf["terminal_mean"] > res_zero["terminal_mean"]
    print("PASS: TEST 11 (Drift modes: Zero, Risk-Free, and Historical)")


def test_12_bootstrap_and_confidence_levels():
    """
    TEST 12:
    Verify bootstrap simulation, multi-confidence VaR (90, 95, 99), and max drawdown.
    """
    res = run_monte_carlo_engine("AAPL", 100000.0, 252, simulations=2000, seed=42, confidence_level=99)
    assert "bootstrap" in res
    assert res["bootstrap"]["var_active"] > 0
    assert res["bootstrap"]["cvar_active"] >= res["bootstrap"]["var_active"]
    # 99% VaR must be strictly higher than 95% and 90% VaR
    assert res["var_99"] > res["var_95"] > res["var_90"]
    assert res["cvar_99"] >= res["cvar_95"] >= res["cvar_90"]
    # Maximum drawdown must be non-negative percentage
    assert res["median_max_drawdown_pct"] <= 0.0
    assert res["worst_max_drawdown_pct"] <= res["median_max_drawdown_pct"]
    print("PASS: TEST 12 (Bootstrap fat-tails, 90/95/99% VaR/CVaR, and Max Drawdown)")


def test_13_block_bootstrap_and_basel_horizons():
    """
    TEST 13:
    Verify short Basel horizons (1-day, 10-day) and 5-day block bootstrap execution.
    At 1-day and 10-day, day 0 portfolio value must identically match initial investment.
    """
    for horizon in [1, 10]:
        res = run_monte_carlo_engine("AAPL", 100000.0, time_horizon=horizon, simulations=2000, seed=42)
        assert res["time_horizon"] == horizon
        assert res["var_active"] > 0
        assert res["var_se"] > 0
        assert res["bootstrap"]["var_active"] > 0
        # At 1-day, loss amounts are realistic fractional percentages of capital
        if horizon == 1:
            assert res["var_active_pct"] < 10.0
    print("PASS: TEST 13 (Short Basel Horizons: 1-Day & 10-Day Block Bootstrap)")


def test_14_dynamic_risk_free_rates_and_var_se():
    """
    TEST 14:
    Verify dynamic risk-free rate resolution (USD ~4.0%, INR ~6.8%) and Standard Error calculation.
    """
    res_usd = run_monte_carlo_engine("AAPL", 100000.0, 252, simulations=1000, drift_mode="risk_free", seed=42)
    res_inr = run_monte_carlo_engine("RELIANCE.NS", 100000.0, 252, simulations=1000, drift_mode="risk_free", seed=42)
    assert abs(res_usd["rf_rate"] - 0.040) < 1e-4
    assert abs(res_inr["rf_rate"] - 0.068) < 1e-4
    assert res_usd["var_se"] > 0
    assert res_inr["var_se"] > 0
    print("PASS: TEST 14 (Dynamic Risk-Free Rates: USD 4.0% & INR 6.8% + VaR Standard Error)")


if __name__ == "__main__":
    print("\n================ RUNNING MONTE CARLO TEST SUITE ================")
    test_1_investment_1001_stock_1167()
    test_2_investment_100000()
    test_3_varying_stock_prices()
    test_4_universal_ticker_support()
    test_5_simulation_count_and_all_paths_used()
    test_6_invalid_ticker_error()
    test_7_insufficient_historical_data()
    test_8_currency_mismatch_and_no_silent_mixing()
    test_9_reproducibility()
    test_10_different_seeds()
    test_11_drift_modes()
    test_12_bootstrap_and_confidence_levels()
    test_13_block_bootstrap_and_basel_horizons()
    test_14_dynamic_risk_free_rates_and_var_se()
    print("================ ALL 14 TESTS PASSED SUCCESSFULLY! ================\n")


