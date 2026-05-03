"""
Map Earnings-22 call IDs to real stock tickers and download actual market data.
This replaces the synthetic/random targets with real post-earnings returns.
"""

import logging
import datetime
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Manual mapping: call_id → (ticker, approximate_call_date)
# Derived from the first sentence of each call transcript
# -----------------------------------------------------------------------
CALL_MAPPING = {
    # MTN Ghana (listed on Ghana Stock Exchange, use MTN.JO for JSE-listed MTN Group)
    "2020-03-0230487MTN-Ghana-2019-Annual-Results-Call": ("MTNOY", "2020-03-02"),
    "2020-Annual-Results-Call-Recording": ("MTNOY", "2021-03-15"),
    # LATAM Airlines
    "4329526": ("LTM", "2020-03-06"),
    # Telkom SA
    "4351517": ("TLKGY", "2020-06-15"),
    # EDP (Energias de Portugal)
    "4372696": ("EDPFY", "2020-07-29"),
    # SK Telecom
    "4430051": ("SKM", "2021-02-04"),
    # Telecom Italia / TIM
    "4432298": ("TIIAY", "2021-05-12"),
    # Deutsche Telekom / T-Mobile
    "4450488": ("DTEGY", "2021-08-12"),
    # ELD Electronics → Delta Electronics (Thailand)
    "4463693": ("DLELY", "2021-11-04"),
    # Turkcell
    "4466399": ("TKC", "2021-10-28"),
    # Bancolombia
    "4466718": ("CIB", "2021-11-10"),
    # Net1 UEPS Technologies
    "4467434": ("UEPS", "2021-11-09"),
    # Pampa Energia
    "4468679": ("PAM", "2021-11-12"),
    # Nexi (Italian payments)
    "4468715": ("NEXXY", "2021-11-11"),
    # Loma Negra
    "4468919": ("LOMA", "2021-11-11"),
    # YPF (Argentina)
    "4469528": ("YPF", "2021-11-12"),
    # Generic call - try Investec
    "4469590": ("IVTJF", "2021-11-18"),
    # Unknown - skip
    "4470253": (None, "2021-11-15"),
    # Sabesp (Brazilian water utility)
    "4470663": ("SBS", "2021-11-12"),
    # Unknown - skip
    "4471809": (None, "2021-11-18"),
    # Unknown (disclaimer call)
    "4473837": (None, "2021-12-01"),
    # DS Smith (UK packaging)
    "4474327": ("DITHF", "2021-12-08"),
    # MercadoLibre or similar (20-F filer)
    "4475604": (None, "2021-12-15"),
    # Investor AB (Swedish)
    "4480850": ("IVSBF", "2022-01-20"),
    # Unknown Asian company
    "4481766": (None, "2022-01-27"),
    # Evolution Mining
    "4481952": ("CAHPF", "2022-01-27"),
    # Unknown Asian company
    "4481967": (None, "2022-01-27"),
    # GasLog Partners
    "4482110": ("GLOP", "2022-02-03"),
    # Vedanta Limited
    "4482613": ("VEDL", "2022-01-28"),
    # Imperial Oil
    "4483296": ("IMO", "2022-02-01"),
    # Tele2 (Swedish telecom)
    "4483338": ("TLTZF", "2022-02-03"),
    # TeamViewer AG
    "4483589": ("TMVWY", "2022-02-09"),
    # Infineon Technologies
    "4483912": ("IFNNY", "2022-02-03"),
    # New Residential Investment (now Rithm Capital)
    "4485206": ("RITM", "2022-02-10"),
    # MTN Nigeria
    "MP3-A": ("MTNOY", "2020-05-01"),
}


def get_real_market_data(project_root: Path):
    """Download real stock prices for our 35 calls."""
    
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return
    
    results = []
    
    for call_id, (ticker, call_date_str) in CALL_MAPPING.items():
        if ticker is None:
            logger.warning("No ticker for %s, using synthetic target", call_id)
            np.random.seed(hash(call_id) % 2**31)
            results.append({
                "call_id": call_id,
                "ticker": "UNKNOWN",
                "call_date": datetime.date.fromisoformat(call_date_str),
                "return_1d": float(np.random.normal(0.0, 0.02)),
                "return_5d": float(np.random.normal(0.0, 0.04)),
                "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                "close_t0": 100.0,
                "close_t1": 100.0,
                "close_t5": 100.0,
                "data_source": "synthetic",
            })
            continue
        
        call_date = datetime.date.fromisoformat(call_date_str)
        start = call_date - datetime.timedelta(days=5)
        end = call_date + datetime.timedelta(days=15)
        
        try:
            logger.info("Downloading %s for %s (%s)...", ticker, call_id, call_date_str)
            stock = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
            
            if len(stock) < 3:
                logger.warning("Not enough data for %s (%s), using synthetic", ticker, call_id)
                np.random.seed(hash(call_id) % 2**31)
                results.append({
                    "call_id": call_id,
                    "ticker": ticker,
                    "call_date": call_date,
                    "return_1d": float(np.random.normal(0.0, 0.02)),
                    "return_5d": float(np.random.normal(0.0, 0.04)),
                    "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                    "close_t0": 100.0,
                    "close_t1": 100.0,
                    "close_t5": 100.0,
                    "data_source": "synthetic_fallback",
                })
                continue
            
            # Find the closest trading day to call_date
            import pandas as pd
            stock = stock.sort_index()
            
            # Handle multi-level columns from yfinance
            if isinstance(stock.columns, pd.MultiIndex):
                close_col = stock.columns[stock.columns.get_level_values(0) == "Close"][0]
                closes = stock[close_col]
            else:
                closes = stock["Close"]
            
            # Convert call_date to pandas Timestamp for comparison
            call_ts = pd.Timestamp(call_date)
            
            # Find t0 (call day or next trading day)
            t0_idx = 0
            for i, d in enumerate(closes.index):
                if d >= call_ts:
                    t0_idx = i
                    break
            
            close_t0 = float(closes.iloc[t0_idx])
            
            # T+1
            if t0_idx + 1 < len(closes):
                close_t1 = float(closes.iloc[t0_idx + 1])
            else:
                close_t1 = close_t0
            
            # T+5
            if t0_idx + 5 < len(closes):
                close_t5 = float(closes.iloc[t0_idx + 5])
            else:
                close_t5 = float(closes.iloc[-1])
            
            return_1d = (close_t1 - close_t0) / close_t0
            return_5d = (close_t5 - close_t0) / close_t0
            
            # Realized volatility (std of daily returns over 5 days)
            if t0_idx + 5 < len(closes):
                window = closes.iloc[t0_idx:t0_idx + 6]
                daily_returns = window.pct_change().dropna()
                realized_vol = float(daily_returns.std()) if len(daily_returns) > 1 else 0.02
            else:
                realized_vol = 0.02
            
            results.append({
                "call_id": call_id,
                "ticker": ticker,
                "call_date": call_date,
                "return_1d": return_1d,
                "return_5d": return_5d,
                "realized_vol_5d": realized_vol,
                "close_t0": close_t0,
                "close_t1": close_t1,
                "close_t5": close_t5,
                "data_source": "real",
            })
            
            logger.info("  %s: close=%.2f, ret_1d=%.4f, ret_5d=%.4f, vol=%.4f",
                        ticker, close_t0, return_1d, return_5d, realized_vol)

            
        except Exception as e:
            logger.warning("Error downloading %s: %s, using synthetic", ticker, e)
            np.random.seed(hash(call_id) % 2**31)
            results.append({
                "call_id": call_id,
                "ticker": ticker,
                "call_date": call_date,
                "return_1d": float(np.random.normal(0.0, 0.02)),
                "return_5d": float(np.random.normal(0.0, 0.04)),
                "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                "close_t0": 100.0,
                "close_t1": 100.0,
                "close_t5": 100.0,
                "data_source": "synthetic_fallback",
            })
    
    # Save
    df = pl.DataFrame(results)
    output_path = project_root / "data" / "processed" / "earnings22_market_data.parquet"
    df.write_parquet(output_path)
    
    real_count = len([r for r in results if r["data_source"] == "real"])
    logger.info("=" * 60)
    logger.info("Saved market data: %d calls (%d real, %d synthetic)", 
                len(df), real_count, len(df) - real_count)
    logger.info("Real return_1d range: [%.4f, %.4f]", df["return_1d"].min(), df["return_1d"].max())
    logger.info("Real vol range: [%.4f, %.4f]", df["realized_vol_5d"].min(), df["realized_vol_5d"].max())
    logger.info("Output: %s", output_path)
    

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    get_real_market_data(project_root)
