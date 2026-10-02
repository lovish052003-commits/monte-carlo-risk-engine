"""
Standalone Monte Carlo Portfolio Risk Simulator (Terminal CLI)
==============================================================
Runs 10,000 Geometric Brownian Motion & Non-parametric Historical Bootstrap simulations,
distinguishing stock price from portfolio capital, supporting drift assumptions (zero/rf/historical),
and calculating multi-confidence VaR, Expected Shortfall (CVaR), and Maximum Drawdown.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from risk_engine import (
    fetch_historical_prices,
    compute_lookback_metadata,
    calculate_log_returns_and_parameters,
    resolve_log_drift,
    simulate_gbm_portfolio,
    simulate_bootstrap_portfolio,
    calculate_risk_statistics,
    detect_currency,
)

# Ensure stdout handles UTF-8 (e.g. for INR '₹' symbol on Windows console)
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    # 1. Parse Parameters / CLI Arguments
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_plot = "--no-plot" not in sys.argv

    # Check for drift mode argument e.g. --drift=historical or --drift=zero or --drift=risk_free
    drift_mode = "zero"
    for a in sys.argv[1:]:
        if a.startswith("--drift="):
            drift_mode = a.split("=")[1].strip().lower()

    ticker = args[0] if len(args) > 0 else "AAPL"
    portfolio_value = float(args[1]) if len(args) > 1 else 100000.0  # Initial portfolio investment amount
    time_horizon = int(args[2]) if len(args) > 2 else 252  # Trading days
    simulations = 10000

    print("==================================================")
    print(f"MONTE CARLO RISK ENGINE (TERMINAL): {ticker}")
    print("==================================================")

    # 2. Fetch Live Historical Data and Determine Actual Lookback
    prices, lookback_desc, stock = fetch_historical_prices(ticker, period="5y")
    lookback_meta = compute_lookback_metadata(prices)
    current_price = float(prices.iloc[-1])
    currency, symbol = detect_currency(ticker, stock)

    try:
        symbol.encode(sys.stdout.encoding or "utf-8")
        safe_symbol = symbol
    except UnicodeEncodeError:
        safe_symbol = f"{currency} "

    print(f"Historical Lookback:  {lookback_meta['lookback_label']}")
    print(f"Data Window:          {lookback_meta['start_date']} to {lookback_meta['end_date']} ({lookback_meta['observations']:,} trading observations, ~{lookback_meta['lookback_years']:.1f} years)")
    print(f"Current Asset Price:  {safe_symbol}{current_price:,.2f} ({currency})")
    print(f"Initial Investment:   {safe_symbol}{portfolio_value:,.2f}")

    # 3. Calculate Parameters & Resolve Drift Assumption
    params = calculate_log_returns_and_parameters(prices)
    from risk_engine import get_default_risk_free_rate
    default_rf = get_default_risk_free_rate(ticker, currency)
    resolved_log_drift, resolved_arithmetic_drift, drift_desc = resolve_log_drift(
        params=params,
        mode=drift_mode,
        rf=default_rf
    )

    print(f"Annualized Volatility (252-d): {params['sigma_annual']*100:.2f}%")
    print(f"Drift Assumption:              {drift_desc}")
    print(f"Applied Annual Log Drift:      {resolved_log_drift*252.0*100:.2f}%")

    # 4. Generate Monte Carlo Portfolio Simulation (10,000 paths)
    portfolio_paths, stock_paths, shares = simulate_gbm_portfolio(
        investment_amount=portfolio_value,
        current_stock_price=current_price,
        log_drift_daily=resolved_log_drift,
        sigma_daily=params["sigma_daily"],
        time_horizon=time_horizon,
        simulations=simulations,
        seed=42,
    )

    print(f"Equivalent Shares:             {shares:,.4f}")
    print(f"Day 0 Portfolio Value:         {safe_symbol}{portfolio_paths[0, 0]:,.2f} (Verified across all 10,000 paths)")

    # 5. Non-parametric 5-Day Moving Block Bootstrap (preserving ARCH clustering)
    boot_paths = simulate_bootstrap_portfolio(
        investment_amount=portfolio_value,
        returns=params["returns_series"],
        log_drift_daily=resolved_log_drift,
        time_horizon=time_horizon,
        simulations=simulations,
        seed=42
    )

    # 6. Calculate Quantitative Risk Statistics
    if time_horizon == 1:
        horizon_label = "1-Day (Basel)"
    elif time_horizon == 10:
        horizon_label = "10-Day (Basel)"
    elif time_horizon == 252:
        horizon_label = "1-Year"
    else:
        horizon_label = f"{time_horizon}-Day"

    gbm_stats = calculate_risk_statistics(portfolio_paths, portfolio_value, confidence_level=95)
    boot_stats = calculate_risk_statistics(boot_paths, portfolio_value, confidence_level=95)

    print("\n--- Model Results Comparison (10,000 Paths) ---")
    print(f"{'Metric':<30} {'GBM (Normal)':<24} {'5-Day Block Bootstrap (ARCH)'}")
    print("-" * 80)
    print(f"{'Mean Terminal Value:':<30} {safe_symbol}{gbm_stats['terminal_mean']:,.2f} ({gbm_stats['expected_return_pct']:+.2f}%)     {safe_symbol}{boot_stats['terminal_mean']:,.2f} ({boot_stats['expected_return_pct']:+.2f}%)")
    print(f"{'Median Terminal Value (P50):':<30} {safe_symbol}{gbm_stats['terminal_median']:,.2f} ({gbm_stats['median_return_pct']:+.2f}%)     {safe_symbol}{boot_stats['terminal_median']:,.2f} ({boot_stats['median_return_pct']:+.2f}%)")
    print(f"{'Worst-5% Terminal Value:':<30} {safe_symbol}{gbm_stats['terminal_p5']:,.2f}                {safe_symbol}{boot_stats['terminal_p5']:,.2f}")
    print(f"{'Probability of Loss:':<30} {gbm_stats['probability_of_loss_pct']:.2f}%                   {boot_stats['probability_of_loss_pct']:.2f}%")
    print(f"{'VaR (' + horizon_label + '):':<30} {safe_symbol}{gbm_stats['var_95']:,.2f} (±{safe_symbol}{gbm_stats['var_95_se']:,.1f})   {safe_symbol}{boot_stats['var_95']:,.2f}")
    print(f"{'VaR (% of Capital):':<30} {gbm_stats['var_95_pct']:.2f}%                   {boot_stats['var_95_pct']:.2f}%")
    print(f"{'CVaR (Expected Shortfall):':<30} {safe_symbol}{gbm_stats['expected_shortfall_95']:,.2f}                {safe_symbol}{boot_stats['expected_shortfall_95']:,.2f}")
    print(f"{'Median Max Drawdown:':<30} {gbm_stats['median_max_drawdown_pct']:.2f}%                   {boot_stats['median_max_drawdown_pct']:.2f}%")
    print(f"{'Worst 5% Max Drawdown:':<30} {gbm_stats['worst_max_drawdown_pct']:.2f}%                   {boot_stats['worst_max_drawdown_pct']:.2f}%")

    print(f"\n* Monte Carlo Noise (Standard Error of VaR): ± {safe_symbol}{gbm_stats['var_95_se']:,.2f}")
    print("* 5-Day Block Bootstrap resamples contiguous 5-day return blocks, preserving ARCH volatility clusters.")

    print("\n* Note: Historical Bootstrap resamples real historical returns directly, preserving fat tails.")

    # 7. Visualize Paths (Optional)
    if show_plot:
        plt.figure(figsize=(10, 6))
        days = np.arange(time_horizon + 1)
        p5 = np.percentile(portfolio_paths, 5, axis=1)
        p25 = np.percentile(portfolio_paths, 25, axis=1)
        p75 = np.percentile(portfolio_paths, 75, axis=1)
        p95 = np.percentile(portfolio_paths, 95, axis=1)
        plt.fill_between(days, p5, p95, color="#38bdf8", alpha=0.15, label="5% - 95% Band")
        plt.fill_between(days, p25, p75, color="#38bdf8", alpha=0.30, label="25% - 75% Band")
        plt.plot(days, portfolio_paths[:, :100], alpha=0.18, color="steelblue", linewidth=0.8)
        plt.plot(days, np.median(portfolio_paths, axis=1), color="green", linewidth=2.5, label="Median Path (P50)")
        plt.axhline(portfolio_value, color="darkorange", linestyle="--", linewidth=1.5, label=f"Initial: {safe_symbol}{portfolio_value:,.0f}")

        plt.title(f"Monte Carlo Trajectory Fan Chart: {ticker}", fontsize=13, fontweight="bold")
        plt.xlabel(f"Trading Days ({horizon_label})", fontsize=11)
        plt.ylabel(f"Portfolio Value ({currency})", fontsize=11)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()