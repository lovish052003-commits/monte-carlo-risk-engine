"""
Monte Carlo Risk & Valuation Engine
====================================
Quantitative Monte Carlo simulation engine and Value at Risk (VaR) analyzer
meeting institutional quantitative finance standards.

Features:
- Geometric Brownian Motion (GBM) with configurable drift (Zero, Risk-Free, Historical)
- Dynamic Risk-Free Rate detection (~4.0% for USD, ~6.8% for INR .NS/.BO)
- Non-parametric Moving Block Bootstrap (5-day blocks) preserving ARCH volatility clustering
- Asymptotic Monte Carlo Standard Error for empirical VaR order statistics
- Multi-confidence VaR & Expected Shortfall (90%, 95%, 99%)
- Path-wise Maximum Drawdown distribution
- Dual-currency FX time-series adjustment
- Interactive Plotly visualizations with logarithmic Y-axis and percentile fan bands
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Tuple


CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "GBp": "p",
    "JPY": "¥",
    "CAD": "CA$",
    "AUD": "A$",
    "CHF": "CHF ",
    "CNY": "¥",
    "HKD": "HK$",
}

# In-memory price cache to avoid redundant Yahoo Finance rate limits
_PRICE_CACHE: Dict[str, Tuple[float, pd.Series, Any, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour TTL


def detect_currency(ticker: str, stock_obj: yf.Ticker) -> Tuple[str, str]:
    """
    Detect the operational quote currency and symbol of the ticker.
    Ensures NSE/BSE tickers (.NS, .BO) and international stocks are correctly identified.
    Handles London Stock Exchange GBp (pence) vs GBP.
    """
    currency = None
    try:
        if hasattr(stock_obj, "fast_info") and stock_obj.fast_info:
            currency = stock_obj.fast_info.currency
    except Exception:
        pass

    if not currency:
        try:
            info = stock_obj.info or {}
            currency = info.get("currency")
        except Exception:
            pass

    # Universal exchange heuristics if metadata is missing
    if not currency:
        upper = ticker.upper()
        if upper.endswith(".NS") or upper.endswith(".BO"):
            currency = "INR"
        elif upper.endswith(".L"):
            currency = "GBP"
        elif upper.endswith(".DE") or upper.endswith(".PA"):
            currency = "EUR"
        elif upper.endswith(".TO"):
            currency = "CAD"
        elif upper.endswith(".AX"):
            currency = "AUD"
        else:
            currency = "USD"

    currency = currency.strip()
    # Handle London Stock Exchange pence quote
    if currency == "GBp" or currency.lower() == "gbp":
        symbol = "£"
        currency = "GBP"
    else:
        currency = currency.upper()
        symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")

    return currency, symbol


def get_default_risk_free_rate(ticker: str, currency: str = "USD") -> float:
    """
    Dynamically resolve the prevailing benchmark risk-free rate based on asset currency/exchange.
    - USD tickers: ~4.0% (US Treasury benchmark)
    - INR tickers (.NS, .BO): ~6.8% (RBI 10Y Sovereign benchmark)
    - GBP / LSE (.L): ~4.5% (UK Gilt)
    - EUR (.DE, .PA): ~3.0% (Bund)
    """
    upper = ticker.upper().strip()
    curr = currency.upper().strip()
    if upper.endswith(".NS") or upper.endswith(".BO") or curr == "INR":
        return 0.068
    elif curr == "GBP" or upper.endswith(".L"):
        return 0.045
    elif curr == "EUR" or upper.endswith(".DE") or upper.endswith(".PA"):
        return 0.030
    else:
        return 0.040


def compute_lookback_metadata(prices: pd.Series) -> Dict[str, Any]:
    """
    Rigorously calculate the actual historical lookback period and observations.
    Never imply 5 years if fewer years are available.
    """
    valid_days = len(prices)
    start_date = prices.index[0].strftime("%Y-%m-%d")
    end_date = prices.index[-1].strftime("%Y-%m-%d")
    calendar_days = (prices.index[-1] - prices.index[0]).days
    lookback_years = calendar_days / 365.25

    if lookback_years >= 4.75 and valid_days >= 1150:
        lookback_label = "5-Year Historical"
    elif lookback_years < 2.0:
        lookback_label = f"IPO-to-Date / {lookback_years:.1f} Years"
    else:
        lookback_label = f"{lookback_years:.1f}-Year Historical"

    lookback_desc = f"{start_date} to {end_date} ({valid_days:,} trading days, ~{lookback_years:.1f} years, {lookback_label})"
    return {
        "start_date": start_date,
        "end_date": end_date,
        "calendar_days": calendar_days,
        "lookback_years": round(lookback_years, 2),
        "lookback_label": lookback_label,
        "lookback_desc": lookback_desc,
        "observations": valid_days,
    }


def fetch_historical_prices(ticker: str, period: str = "5y") -> Tuple[pd.Series, str, Any]:
    """
    Fetch and sanitize adjusted historical closing prices with in-memory caching.
    Uses auto_adjust=True to adjust for stock splits, dividends, and capital actions.
    """
    if not ticker or not ticker.strip():
        raise ValueError("Stock ticker cannot be empty.")

    clean_ticker = ticker.strip().upper()
    cache_key = f"{clean_ticker}_{period}"
    now = time.time()

    if cache_key in _PRICE_CACHE:
        cached_time, cached_series, cached_stock, cached_meta = _PRICE_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_series.copy(), cached_meta["lookback_desc"], cached_stock

    stock = yf.Ticker(clean_ticker)

    try:
        data = stock.history(period=period, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Failed to query market data for ticker '{clean_ticker}': {str(e)}")

    if data is None or data.empty:
        try:
            data = stock.history(period=period)
        except Exception as e:
            raise ValueError(f"Failed to query market data for ticker '{clean_ticker}': {str(e)}")

    if data is None or data.empty:
        raise ValueError(f"No historical price data found for ticker '{clean_ticker}'. Please verify the symbol.")

    if "Close" in data.columns:
        close_series = data["Close"].copy()
    elif "Adj Close" in data.columns:
        close_series = data["Adj Close"].copy()
    else:
        raise ValueError(f"No closing price column found for ticker '{clean_ticker}'.")

    # Filter valid positive numbers, sort chronologically, remove duplicate timestamps
    close_series = close_series.dropna()
    close_series = close_series[close_series > 0]
    close_series = close_series[~close_series.index.duplicated(keep="last")]
    close_series = close_series.sort_index()

    valid_days = len(close_series)
    if valid_days < 30:
        raise ValueError(
            f"Insufficient valid historical data ({valid_days} trading days) for ticker '{clean_ticker}'. "
            "A minimum of 30 trading days is required for reliable statistical estimation."
        )

    # If quoted in GBp (pence on LSE), normalize to GBP
    raw_currency = getattr(stock.fast_info, "currency", None) if hasattr(stock, "fast_info") else None
    if raw_currency == "GBp":
        close_series = close_series / 100.0

    meta = compute_lookback_metadata(close_series)
    _PRICE_CACHE[cache_key] = (now, close_series, stock, meta)

    return close_series, meta["lookback_desc"], stock


def convert_price_series_to_currency(
    prices: pd.Series,
    from_curr: str,
    to_curr: str,
    period: str = "5y"
) -> Tuple[pd.Series, float]:
    """
    Convert price series into the investor's requested currency using historical FX data.
    Captures dual-asset risk (underlying stock volatility + FX exchange rate volatility).
    """
    from_c = from_curr.upper().strip()
    to_c = to_curr.upper().strip()
    if from_c == to_c:
        return prices, 1.0

    pair1 = f"{from_c}{to_c}=X"
    fx_series = None
    try:
        fx_stock = yf.Ticker(pair1)
        fx_hist = fx_stock.history(period=period, auto_adjust=True)
        if not fx_hist.empty and "Close" in fx_hist.columns and len(fx_hist["Close"]) > 10:
            fx_series = fx_hist["Close"].copy()
    except Exception:
        fx_series = None

    if fx_series is None or fx_series.empty:
        pair2 = f"{to_c}{from_c}=X"
        try:
            fx_stock = yf.Ticker(pair2)
            fx_hist = fx_stock.history(period=period, auto_adjust=True)
            if not fx_hist.empty and "Close" in fx_hist.columns and len(fx_hist["Close"]) > 10:
                fx_series = 1.0 / fx_hist["Close"].copy()
        except Exception:
            fx_series = None

    if fx_series is None or fx_series.empty:
        raise ValueError(
            f"Foreign exchange rate between {from_c} and {to_c} is unavailable. "
            "Cannot perform cross-currency conversion safely."
        )

    # Normalize tz and index alignment
    p_idx = prices.index.tz_localize(None).normalize() if prices.index.tz is not None else prices.index.normalize()
    fx_idx = fx_series.index.tz_localize(None).normalize() if fx_series.index.tz is not None else fx_series.index.normalize()

    p_clean = prices.copy()
    p_clean.index = p_idx
    fx_clean = fx_series.copy()
    fx_clean.index = fx_idx
    fx_clean = fx_clean[~fx_clean.index.duplicated(keep="last")]

    # Reindex FX to match price trading days using forward-fill
    fx_reindexed = fx_clean.reindex(p_clean.index, method="ffill").bfill()
    converted_prices = (p_clean * fx_reindexed).dropna()
    converted_prices = converted_prices[converted_prices > 0]

    latest_fx = float(fx_reindexed.iloc[-1])
    return converted_prices, latest_fx


def calculate_log_returns_and_parameters(prices: pd.Series) -> Dict[str, Any]:
    """
    Calculate daily logarithmic returns, annualized volatility, and empirical drift.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    returns = returns[np.isfinite(returns)]

    if len(returns) < 29:
        raise ValueError("Insufficient returns data after log difference calculation.")

    mu_log_daily = float(returns.mean())
    sigma_daily = float(returns.std(ddof=1))

    mu_log_annual = mu_log_daily * 252.0
    sigma_annual = sigma_daily * np.sqrt(252.0)

    # Continuous arithmetic price drift: mu_gbm = mu_log + 0.5 * sigma^2
    mu_gbm_daily = mu_log_daily + 0.5 * (sigma_daily ** 2)
    mu_gbm_annual = mu_gbm_daily * 252.0

    return {
        "mu_log_daily": mu_log_daily,
        "sigma_daily": sigma_daily,
        "log_drift_daily": mu_log_daily,
        "mu_log_annual": mu_log_annual,
        "sigma_annual": sigma_annual,
        "log_drift_annual": mu_log_annual,
        "mu_gbm_daily": mu_gbm_daily,
        "mu_gbm_annual": mu_gbm_annual,
        "returns_series": returns,
        "observations": len(returns),
        # Backward compatibility aliases
        "drift_daily": mu_log_daily,
        "drift_annual": mu_log_annual,
        "mu_daily": mu_log_daily,
        "mu_annual": mu_log_annual,
    }


def resolve_log_drift(
    params: Dict[str, Any],
    mode: str = "zero",
    rf: float = 0.04
) -> Tuple[float, float, str]:
    """
    Resolve the daily log drift under the specified drift mode.
    
    Modes:
    - 'zero': Zero expected arithmetic excess return (mu = 0.0). Conservative standard for risk.
              GBM log drift = -0.5 * sigma_daily^2.
    - 'risk_free': Risk-neutral drift equal to annual risk-free rate rf.
                   GBM log drift = (rf / 252) - 0.5 * sigma_daily^2.
    - 'historical': Observed empirical mean log return directly.
                    GBM log drift = mu_log_daily.
    """
    sigma_daily = params["sigma_daily"]
    mode_clean = (mode or "zero").lower().strip()

    if mode_clean == "historical":
        return params["log_drift_daily"], params["mu_gbm_annual"], "Historical Drift"
    elif mode_clean == "risk_free":
        mu_daily = rf / 252.0
        log_drift = mu_daily - 0.5 * (sigma_daily ** 2)
        return log_drift, rf, f"Risk-Free ({rf*100:.1f}%)"
    else:  # "zero" (default)
        log_drift = -0.5 * (sigma_daily ** 2)
        return log_drift, 0.0, "Zero Drift"


def simulate_gbm_portfolio(
    investment_amount: float,
    current_stock_price: float,
    log_drift_daily: Optional[float] = None,
    sigma_daily: float = 0.0,
    time_horizon: int = 252,
    simulations: int = 10000,
    seed: Optional[int] = 42,
    drift_daily: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate Geometric Brownian Motion paths for Portfolio Value and Stock Price.
    Constructs paths from cumulative log increments.
    Returns: (portfolio_paths, stock_paths, shares)
    """
    if investment_amount <= 0:
        raise ValueError("Investment amount must be strictly greater than 0.")
    if current_stock_price <= 0:
        raise ValueError("Current stock price must be strictly greater than 0.")
    if time_horizon <= 0:
        raise ValueError("Time horizon must be at least 1 trading day.")
    if simulations <= 0:
        raise ValueError("Simulations count must be greater than 0.")

    effective_log_drift = drift_daily if drift_daily is not None else log_drift_daily
    if effective_log_drift is None:
        raise ValueError("Must specify log_drift_daily (or drift_daily).")

    shares = investment_amount / current_stock_price
    rng = np.random.default_rng(seed)

    # Generate daily log increments: (time_horizon, simulations)
    log_increments = effective_log_drift + sigma_daily * rng.standard_normal((time_horizon, simulations))
    
    # Cumulative log increments with 0 at Day 0
    cum_log = np.vstack([np.zeros((1, simulations)), np.cumsum(log_increments, axis=0)])
    growth_factors = np.exp(cum_log)

    portfolio_paths = investment_amount * growth_factors
    stock_paths = current_stock_price * growth_factors

    if np.max(np.abs(portfolio_paths[0, :] - investment_amount)) > 1e-4:
        raise RuntimeError("Day 0 portfolio value deviates from investment amount.")

    return portfolio_paths, stock_paths, shares


def simulate_block_bootstrap_portfolio(
    investment_amount: float,
    returns: pd.Series,
    log_drift_daily: float,
    time_horizon: int = 252,
    simulations: int = 10000,
    block_size: int = 5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Non-parametric Moving Block Bootstrap (resampling contiguous 5-day chunks).
    Preserves autocorrelation, volatility clustering, and ARCH effects.
    """
    rng = np.random.default_rng(seed)
    r = returns.to_numpy()
    M = len(r)
    b = min(max(1, block_size), M)

    # Re-center empirical returns onto target log drift
    r_centered = r - np.mean(r) + log_drift_daily
    num_blocks = int(np.ceil(time_horizon / b))
    max_start = M - b

    if max_start <= 0:
        # Fallback to standard i.i.d. bootstrap if history is shorter than block size
        boot_increments = rng.choice(r_centered, size=(time_horizon, simulations), replace=True)
    else:
        # Vectorized block sampling:
        # Sample starting indices for each block: shape (num_blocks, simulations)
        start_idx = rng.integers(0, max_start + 1, size=(num_blocks, simulations))
        # Offsets 0..b-1: shape (b, 1, 1)
        offsets = np.arange(b).reshape(b, 1, 1)
        # Full contiguous blocks: shape (b, num_blocks, simulations)
        block_idx = (start_idx[np.newaxis, :, :] + offsets).reshape(num_blocks * b, simulations)
        # Slice to exact required time horizon
        boot_increments = r_centered[block_idx[:time_horizon, :]]

    cum_log = np.vstack([np.zeros((1, simulations)), np.cumsum(boot_increments, axis=0)])
    portfolio_paths = investment_amount * np.exp(cum_log)
    return portfolio_paths


# Backward compatibility alias
simulate_bootstrap_portfolio = simulate_block_bootstrap_portfolio


def compute_var_standard_error(terminal_values: np.ndarray, alpha: float) -> float:
    """
    Asymptotic Monte Carlo Standard Error of empirical VaR (quantile order statistic).
    SE(q_alpha) = sqrt(alpha * (1 - alpha) / N) / f(q_alpha)
    where 1/f(q_alpha) is the empirical quantile spread / (2 * delta).
    Quantifies the simulation noise (Monte Carlo error) of the VaR estimate.
    """
    N = len(terminal_values)
    if N < 50:
        return 0.0

    delta = 0.015  # 1.5% neighborhood around alpha
    low_p = max(0.001, alpha - delta)
    high_p = min(0.999, alpha + delta)

    q_low = float(np.percentile(terminal_values, low_p * 100.0))
    q_high = float(np.percentile(terminal_values, high_p * 100.0))
    spread = q_high - q_low
    if spread <= 0:
        return 0.0

    inv_density = spread / (high_p - low_p)
    se = float(np.sqrt((alpha * (1.0 - alpha)) / N) * inv_density)
    return se


def calculate_risk_statistics(
    portfolio_paths: np.ndarray,
    investment_amount: float,
    confidence_level: int = 95
) -> Dict[str, Any]:
    """
    Calculate comprehensive risk and return statistics across all simulated paths.
    Supports 90%, 95%, and 99% VaR / CVaR, Monte Carlo Standard Error, and Maximum Drawdown.
    """
    terminal_values = portfolio_paths[-1, :]
    n_sims = len(terminal_values)

    terminal_mean = float(np.mean(terminal_values))
    terminal_median = float(np.median(terminal_values))
    
    p1 = float(np.percentile(terminal_values, 1))
    p5 = float(np.percentile(terminal_values, 5))
    p10 = float(np.percentile(terminal_values, 10))
    p25 = float(np.percentile(terminal_values, 25))
    p50 = float(np.percentile(terminal_values, 50))
    p75 = float(np.percentile(terminal_values, 75))
    p90 = float(np.percentile(terminal_values, 90))
    p95 = float(np.percentile(terminal_values, 95))
    p99 = float(np.percentile(terminal_values, 99))

    prob_loss = float(np.count_nonzero(terminal_values < investment_amount) / n_sims)
    expected_return = float((terminal_mean / investment_amount) - 1.0)
    median_return = float((terminal_median / investment_amount) - 1.0)

    # 90% VaR & CVaR (alpha = 0.10)
    var_90 = float(max(0.0, investment_amount - p10))
    tail_90 = terminal_values[terminal_values <= p10]
    cvar_90 = float(max(0.0, investment_amount - np.mean(tail_90 if len(tail_90) > 0 else [p10])))
    var_90_se = compute_var_standard_error(terminal_values, alpha=0.10)

    # 95% VaR & CVaR (alpha = 0.05)
    var_95 = float(max(0.0, investment_amount - p5))
    tail_95 = terminal_values[terminal_values <= p5]
    cvar_95 = float(max(0.0, investment_amount - np.mean(tail_95 if len(tail_95) > 0 else [p5])))
    var_95_se = compute_var_standard_error(terminal_values, alpha=0.05)

    # 99% VaR & CVaR (alpha = 0.01)
    var_99 = float(max(0.0, investment_amount - p1))
    tail_99 = terminal_values[terminal_values <= p1]
    cvar_99 = float(max(0.0, investment_amount - np.mean(tail_99 if len(tail_99) > 0 else [p1])))
    var_99_se = compute_var_standard_error(terminal_values, alpha=0.01)

    # Path-wise Maximum Drawdown (peak-to-trough drop along each simulated trajectory)
    running_max = np.maximum.accumulate(portfolio_paths, axis=0)
    drawdowns = (portfolio_paths - running_max) / running_max
    max_dd_per_path = np.min(drawdowns, axis=0)  # negative values
    median_max_dd = float(np.median(max_dd_per_path))
    worst_5pct_max_dd = float(np.percentile(max_dd_per_path, 5))

    # Active metrics corresponding to user-selected confidence level
    if confidence_level == 99:
        active_var = var_99
        active_cvar = cvar_99
        active_threshold = p1
        active_se = var_99_se
    elif confidence_level == 90:
        active_var = var_90
        active_cvar = cvar_90
        active_threshold = p10
        active_se = var_90_se
    else:  # default 95
        active_var = var_95
        active_cvar = cvar_95
        active_threshold = p5
        active_se = var_95_se

    return {
        "terminal_mean": terminal_mean,
        "terminal_median": terminal_median,
        "terminal_p1": p1,
        "terminal_p5": p5,
        "terminal_p10": p10,
        "terminal_p25": p25,
        "terminal_p50": p50,
        "terminal_p75": p75,
        "terminal_p90": p90,
        "terminal_p95": p95,
        "terminal_p99": p99,
        "probability_of_loss": prob_loss,
        "probability_of_loss_pct": prob_loss * 100.0,
        "expected_return": expected_return,
        "expected_return_pct": expected_return * 100.0,
        "median_return": median_return,
        "median_return_pct": median_return * 100.0,
        "confidence_level": confidence_level,
        "active_threshold": active_threshold,
        "var_active": active_var,
        "var_active_pct": (active_var / investment_amount) * 100.0,
        "cvar_active": active_cvar,
        "cvar_active_pct": (active_cvar / investment_amount) * 100.0,
        "var_se": active_se,
        "var_se_pct": (active_se / investment_amount) * 100.0,
        "var_90": var_90,
        "var_90_pct": (var_90 / investment_amount) * 100.0,
        "cvar_90": cvar_90,
        "cvar_90_pct": (cvar_90 / investment_amount) * 100.0,
        "var_90_se": var_90_se,
        "var_95": var_95,
        "var_95_pct": (var_95 / investment_amount) * 100.0,
        "cvar_95": cvar_95,
        "cvar_95_pct": (cvar_95 / investment_amount) * 100.0,
        "expected_shortfall_95": cvar_95,
        "expected_shortfall_95_pct": (cvar_95 / investment_amount) * 100.0,
        "var_95_se": var_95_se,
        "var_99": var_99,
        "var_99_pct": (var_99 / investment_amount) * 100.0,
        "cvar_99": cvar_99,
        "cvar_99_pct": (cvar_99 / investment_amount) * 100.0,
        "var_99_se": var_99_se,
        "median_max_drawdown_pct": median_max_dd * 100.0,
        "worst_max_drawdown_pct": worst_5pct_max_dd * 100.0,
    }


def build_simulation_chart(
    portfolio_paths: np.ndarray,
    investment_amount: float,
    ticker: str,
    currency: str,
    currency_symbol: str,
    time_horizon: int,
    simulations: int,
    num_paths_to_plot: int = 100,
    lookback_label: str = "",
    drift_desc: str = ""
) -> str:
    """
    Build interactive Plotly chart exactly matching Picture 2:
    - 5%–95% confidence fan band (cyan fill)
    - 25%–75% core fan band (deeper cyan fill)
    - 100 visible stochastic simulation paths (rgba(148, 163, 184, 0.18))
    - Prominent green Median Path (P50) line
    - Dashed amber Initial Investment baseline
    - Top-right horizontal legend with Median Path (P50), 25% - 75% Band, 5% - 95% Band
    - Title: Monte Carlo Trajectory Fan Chart: {ticker}
    - Subtitle: {simulations:,} Sims | {time_horizon}d Horizon | {lookback_label} | {drift_desc}
    """
    days = list(range(time_horizon + 1))
    paths_subset = portfolio_paths[:, :num_paths_to_plot]

    p5_curve = np.percentile(portfolio_paths, 5, axis=1)
    p25_curve = np.percentile(portfolio_paths, 25, axis=1)
    median_curve = np.median(portfolio_paths, axis=1)
    p75_curve = np.percentile(portfolio_paths, 75, axis=1)
    p95_curve = np.percentile(portfolio_paths, 95, axis=1)

    fig = go.Figure()

    # Outer Fan Band: 5th to 95th Percentile
    fig.add_trace(go.Scatter(
        x=days, y=p95_curve,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=days, y=p5_curve,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.12)",
        name="5% - 95% Band",
        hoverinfo="skip"
    ))

    # Inner Fan Band: 25th to 75th Percentile
    fig.add_trace(go.Scatter(
        x=days, y=p75_curve,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=days, y=p25_curve,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.25)",
        name="25% - 75% Band",
        hoverinfo="skip"
    ))

    # 100 stochastic simulation paths (clearly visible network flowing through the bands)
    for i in range(min(num_paths_to_plot, paths_subset.shape[1])):
        fig.add_trace(go.Scatter(
            x=days,
            y=paths_subset[:, i],
            mode="lines",
            line=dict(width=0.8, color="rgba(148, 163, 184, 0.18)"),
            hoverinfo="skip",
            showlegend=False
        ))

    # Baseline: Initial Investment line
    fig.add_hline(
        y=investment_amount,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Initial: {currency_symbol}{investment_amount:,.2f}",
        annotation_position="bottom right",
        annotation_font=dict(color="#f59e0b", size=10)
    )

    # Median Path across ALL simulations
    fig.add_trace(go.Scatter(
        x=days,
        y=median_curve,
        mode="lines",
        line=dict(width=2.5, color="#10b981"),
        name="Median Path (P50)",
        hovertemplate=f"Day %{{x}}: {currency_symbol}%{{y:,.2f}}<extra>Median</extra>"
    ))

    lookback_part = f"{lookback_label} | " if lookback_label else ""
    drift_part = f"{drift_desc}" if drift_desc else "Zero Drift (Pure Volatility)"
    subtitle_text = f"{simulations:,} Sims | {time_horizon}d Horizon | {lookback_part}{drift_part}"

    fig.update_layout(
        title=dict(
            text=f"<b>Monte Carlo Trajectory Fan Chart: {ticker}</b><br>"
                 f"<span style='font-size: 12px; color: #94a3b8; font-weight: normal;'>"
                 f"{subtitle_text}"
                 f"</span>",
            font=dict(size=17, color="#f8fafc")
        ),
        xaxis=dict(
            title=dict(text="Trading Days", font=dict(color="#94a3b8")),
            gridcolor="#18181b",
            zerolinecolor="#27272a",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            title=dict(text=f"Portfolio Value ({currency})", font=dict(color="#94a3b8")),
            gridcolor="#18181b",
            zerolinecolor="#27272a",
            tickfont=dict(color="#94a3b8"),
            tickprefix=currency_symbol,
            autorange=True
        ),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=60, r=30, t=75, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1")
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color="#ffffff",
            font_size=13
        )
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_terminal_histogram(
    terminal_values: np.ndarray,
    investment_amount: float,
    active_var: float,
    active_cvar: float,
    confidence_level: int,
    currency_symbol: str,
    currency: str,
    ticker: str
) -> str:
    """
    Build terminal portfolio value distribution histogram highlighting the VaR & CVaR tail risk.
    """
    var_cutoff = investment_amount - active_var
    cvar_cutoff = investment_amount - active_cvar
    median_val = float(np.median(terminal_values))

    fig = go.Figure()

    # Main distribution histogram
    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=75,
        marker=dict(
            color="rgba(56, 189, 248, 0.45)",
            line=dict(color="#38bdf8", width=0.5)
        ),
        name="Terminal Distribution",
        hovertemplate=f"Range: {currency_symbol}%{{x:,.0f}}<br>Paths: %{{y}}<extra></extra>"
    ))

    # VaR cutoff line
    fig.add_vline(
        x=var_cutoff,
        line_dash="dash",
        line_color="#f43f5e",
        line_width=2.5,
        annotation_text=f"{confidence_level}% VaR Threshold: {currency_symbol}{var_cutoff:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color="#f43f5e", size=11)
    )

    # CVaR cutoff line
    fig.add_vline(
        x=cvar_cutoff,
        line_dash="dot",
        line_color="#e11d48",
        line_width=2.0,
        annotation_text=f"{confidence_level}% CVaR (Tail Mean): {currency_symbol}{cvar_cutoff:,.0f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#e11d48", size=10)
    )

    # Initial Investment line
    fig.add_vline(
        x=investment_amount,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.8,
        annotation_text=f"Initial: {currency_symbol}{investment_amount:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=11)
    )

    # Median line
    fig.add_vline(
        x=median_val,
        line_dash="solid",
        line_color="#10b981",
        line_width=2.0,
        annotation_text=f"Median: {currency_symbol}{median_val:,.0f}",
        annotation_position="bottom right",
        annotation_font=dict(color="#10b981", size=11)
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Terminal Portfolio Value Distribution: {ticker}</b><br>"
                 f"<span style='font-size: 12px; color: #94a3b8; font-weight: normal;'>"
                 f"10,000 Simulated Outcomes | Shaded Tail = Loss Exceeding {confidence_level}% VaR"
                 f"</span>",
            font=dict(size=17, color="#f8fafc")
        ),
        xaxis=dict(
            title=dict(text=f"Simulated Terminal Portfolio Value ({currency})", font=dict(color="#94a3b8")),
            gridcolor="#18181b",
            zerolinecolor="#27272a",
            tickfont=dict(color="#94a3b8"),
            tickprefix=currency_symbol
        ),
        yaxis=dict(
            title=dict(text="Frequency (Simulation Paths)", font=dict(color="#94a3b8")),
            gridcolor="#18181b",
            zerolinecolor="#27272a",
            tickfont=dict(color="#94a3b8")
        ),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=60, r=30, t=75, b=50),
        hovermode="x",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color="#ffffff",
            font_size=13
        )
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def run_monte_carlo_engine(
    ticker: str,
    investment_amount: float,
    time_horizon: int = 252,
    simulations: int = 10000,
    requested_currency: Optional[str] = None,
    seed: Optional[int] = 42,
    drift_mode: str = "zero",
    confidence_level: int = 95,
    rf_rate: Optional[float] = None,
    method: str = "both"
) -> Dict[str, Any]:
    """
    Main orchestration function for the Monte Carlo Risk Engine.
    Executes both parametric GBM and 5-day moving Block Bootstrap simulations.
    """
    # 1. Fetch historical price series
    prices, lookback_desc, stock = fetch_historical_prices(ticker, period="5y")
    lookback_meta = compute_lookback_metadata(prices)
    stock_currency, default_symbol = detect_currency(ticker, stock)

    # 2. Foreign Exchange time-series adjustment if requested
    active_currency = stock_currency
    active_symbol = default_symbol
    fx_rate = 1.0

    if requested_currency and requested_currency.strip().upper() != stock_currency:
        req_c = requested_currency.strip().upper()
        prices_adj, fx_rate = convert_price_series_to_currency(prices, stock_currency, req_c, period="5y")
        active_currency = req_c
        active_symbol = CURRENCY_SYMBOLS.get(active_currency, f"{active_currency} ")
        current_price = float(prices_adj.iloc[-1])
        working_prices = prices_adj
    else:
        current_price = float(prices.iloc[-1])
        working_prices = prices

    # 3. Parameter estimation
    params = calculate_log_returns_and_parameters(working_prices)

    # 4. Dynamic Risk-Free Rate benchmark resolution
    if rf_rate is None or rf_rate == 0.05:
        # Dynamically set 4.0% for USD and 6.8% for INR tickers
        effective_rf = get_default_risk_free_rate(ticker, active_currency)
    else:
        effective_rf = rf_rate

    # 5. Drift resolution
    resolved_log_drift, resolved_arithmetic_drift, drift_desc = resolve_log_drift(
        params=params,
        mode=drift_mode,
        rf=effective_rf
    )

    # 6. Parametric GBM Simulation
    portfolio_paths, stock_paths, shares = simulate_gbm_portfolio(
        investment_amount=investment_amount,
        current_stock_price=current_price,
        log_drift_daily=resolved_log_drift,
        sigma_daily=params["sigma_daily"],
        time_horizon=time_horizon,
        simulations=simulations,
        seed=seed
    )
    gbm_stats = calculate_risk_statistics(portfolio_paths, investment_amount, confidence_level=confidence_level)

    # 7. Non-parametric 5-Day Moving Block Bootstrap Simulation (preserving ARCH effects)
    boot_paths = simulate_block_bootstrap_portfolio(
        investment_amount=investment_amount,
        returns=params["returns_series"],
        log_drift_daily=resolved_log_drift,
        time_horizon=time_horizon,
        simulations=simulations,
        block_size=5,
        seed=seed
    )
    boot_stats = calculate_risk_statistics(boot_paths, investment_amount, confidence_level=confidence_level)

    # 8. Generate Interactive Visualizations
    plot_html = build_simulation_chart(
        portfolio_paths=portfolio_paths,
        investment_amount=investment_amount,
        ticker=ticker.strip().upper(),
        currency=active_currency,
        currency_symbol=active_symbol,
        time_horizon=time_horizon,
        simulations=simulations,
        num_paths_to_plot=100,
        lookback_label=lookback_meta["lookback_label"],
        drift_desc=drift_desc
    )

    hist_html = build_terminal_histogram(
        terminal_values=portfolio_paths[-1, :],
        investment_amount=investment_amount,
        active_var=gbm_stats["var_active"],
        active_cvar=gbm_stats["cvar_active"],
        confidence_level=confidence_level,
        currency_symbol=active_symbol,
        currency=active_currency,
        ticker=ticker.strip().upper()
    )

    # Dynamic horizon descriptor
    if time_horizon == 1:
        horizon_label = "1-Day (Basel)"
    elif time_horizon == 10:
        horizon_label = "10-Day (Basel)"
    elif time_horizon == 252:
        horizon_label = "1-Year (252-Day)"
    else:
        horizon_label = f"{time_horizon}-Day"

    return {
        # Core identification & inputs
        "ticker": ticker.strip().upper(),
        "initial_investment": round(investment_amount, 2),
        "portfolio_value": round(investment_amount, 2),  # Backward compatibility
        "time_horizon": time_horizon,
        "horizon_label": horizon_label,
        "simulations": simulations,
        "confidence_level": confidence_level,
        "drift_mode": drift_mode,
        "drift_desc": drift_desc,
        "rf_rate": round(effective_rf, 4),
        "stock_price": round(current_price, 2),
        "initial_price": round(current_price, 2),  # Backward compatibility
        "stock_price_converted": round(current_price, 2),
        "shares": round(shares, 4),
        "currency": active_currency,
        "currency_symbol": active_symbol,
        "stock_currency": stock_currency,
        "fx_rate": fx_rate,

        # Historical statistical assumptions
        "historical_period": lookback_meta["lookback_desc"],
        "actual_historical_period": f"{lookback_meta['start_date']} to {lookback_meta['end_date']} ({lookback_meta['observations']:,} observations, {lookback_meta['lookback_years']} years)",
        "lookback_label": lookback_meta["lookback_label"],
        "observations": lookback_meta["observations"],
        "lookback_years": lookback_meta["lookback_years"],
        "annualized_mean_log_return": round(params["mu_log_annual"], 4),
        "annualized_volatility": round(params["sigma_annual"], 4),
        "gbm_log_drift_annual": round(resolved_log_drift * 252.0, 4),
        "gbm_arithmetic_drift_annual": round(resolved_arithmetic_drift, 4),
        "annualized_drift": round(resolved_log_drift * 252.0, 4),

        # Active Risk & Return (GBM Model)
        "terminal_mean": round(gbm_stats["terminal_mean"], 2),
        "mean_simulated_terminal_value": round(gbm_stats["terminal_mean"], 2),
        "expected_portfolio_value": round(gbm_stats["terminal_mean"], 2),  # Backward compatibility
        "terminal_median": round(gbm_stats["terminal_median"], 2),
        "median_terminal_value": round(gbm_stats["terminal_median"], 2),
        "terminal_p1": round(gbm_stats["terminal_p1"], 2),
        "terminal_p5": round(gbm_stats["terminal_p5"], 2),
        "terminal_p10": round(gbm_stats["terminal_p10"], 2),
        "terminal_p25": round(gbm_stats["terminal_p25"], 2),
        "terminal_p50": round(gbm_stats["terminal_p50"], 2),
        "terminal_p75": round(gbm_stats["terminal_p75"], 2),
        "terminal_p90": round(gbm_stats["terminal_p90"], 2),
        "terminal_p95": round(gbm_stats["terminal_p95"], 2),
        "terminal_p99": round(gbm_stats["terminal_p99"], 2),
        "worst_5_pct_terminal_value": round(gbm_stats["terminal_p5"], 2),
        "probability_of_loss": round(gbm_stats["probability_of_loss"], 4),
        "probability_of_loss_pct": round(gbm_stats["probability_of_loss_pct"], 2),
        "expected_return": round(gbm_stats["expected_return"], 4),
        "expected_return_pct": round(gbm_stats["expected_return_pct"], 2),
        "median_return": round(gbm_stats["median_return"], 4),
        "median_return_pct": round(gbm_stats["median_return_pct"], 2),

        # Active VaR / CVaR metrics and Monte Carlo Standard Error
        "var_active": round(gbm_stats["var_active"], 2),
        "var_active_pct": round(gbm_stats["var_active_pct"], 2),
        "cvar_active": round(gbm_stats["cvar_active"], 2),
        "cvar_active_pct": round(gbm_stats["cvar_active_pct"], 2),
        "var_se": round(gbm_stats["var_se"], 2),
        "var_se_pct": round(gbm_stats["var_se_pct"], 2),
        "active_threshold": round(gbm_stats["active_threshold"], 2),

        # Standard 90%, 95%, 99% tables
        "var_90": round(gbm_stats["var_90"], 2),
        "var_90_pct": round(gbm_stats["var_90_pct"], 2),
        "cvar_90": round(gbm_stats["cvar_90"], 2),
        "cvar_90_pct": round(gbm_stats["cvar_90_pct"], 2),
        "var_90_se": round(gbm_stats["var_90_se"], 2),
        "var_95": round(gbm_stats["var_95"], 2),
        "var_95_pct": round(gbm_stats["var_95_pct"], 2),
        "cvar_95": round(gbm_stats["expected_shortfall_95"], 2),
        "cvar_95_pct": round(gbm_stats["expected_shortfall_95_pct"], 2),
        "expected_shortfall_95": round(gbm_stats["expected_shortfall_95"], 2),
        "expected_shortfall_95_pct": round(gbm_stats["expected_shortfall_95_pct"], 2),
        "var_95_se": round(gbm_stats["var_95_se"], 2),
        "var_99": round(gbm_stats["var_99"], 2),
        "var_99_pct": round(gbm_stats["var_99_pct"], 2),
        "cvar_99": round(gbm_stats["cvar_99"], 2),
        "cvar_99_pct": round(gbm_stats["cvar_99_pct"], 2),
        "var_99_se": round(gbm_stats["var_99_se"], 2),

        # Maximum Drawdown distribution
        "median_max_drawdown_pct": round(gbm_stats["median_max_drawdown_pct"], 2),
        "worst_max_drawdown_pct": round(gbm_stats["worst_max_drawdown_pct"], 2),

        # Non-parametric Block Bootstrap comparison metrics (5-day blocks, ARCH effects)
        "bootstrap": {
            "terminal_mean": round(boot_stats["terminal_mean"], 2),
            "terminal_median": round(boot_stats["terminal_median"], 2),
            "probability_of_loss_pct": round(boot_stats["probability_of_loss_pct"], 2),
            "var_active": round(boot_stats["var_active"], 2),
            "var_active_pct": round(boot_stats["var_active_pct"], 2),
            "cvar_active": round(boot_stats["cvar_active"], 2),
            "cvar_active_pct": round(boot_stats["cvar_active_pct"], 2),
            "var_se": round(boot_stats["var_se"], 2),
            "var_95": round(boot_stats["var_95"], 2),
            "var_95_pct": round(boot_stats["var_95_pct"], 2),
            "cvar_95": round(boot_stats["expected_shortfall_95"], 2),
            "cvar_95_pct": round(boot_stats["expected_shortfall_95_pct"], 2),
            "terminal_p5": round(boot_stats["terminal_p5"], 2),
            "worst_5_pct_terminal_value": round(boot_stats["terminal_p5"], 2),
            "median_max_drawdown_pct": round(boot_stats["median_max_drawdown_pct"], 2),
            "worst_max_drawdown_pct": round(boot_stats["worst_max_drawdown_pct"], 2),
        },

        # Visualizations
        "plot_html": plot_html,
        "hist_html": hist_html,
    }
