"""Yahoo Finance data fetcher with disk caching."""

import hashlib
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFetcher:
    def __init__(self, cache_dir: str = "data/raw", cache_expiry_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_expiry_hours = cache_expiry_hours
        os.makedirs(cache_dir, exist_ok=True)

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: Optional[str] = None,
        field: str = "Adj Close",
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for a list of tickers."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._cache_path(tickers, start, end, field)
        if self._cache_valid(cache_path):
            logger.info("Loading Yahoo data from cache: %s", cache_path)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        df = self._fetch_with_retry(tickers, start, end, field)

        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        logger.info("Cached Yahoo data to %s", cache_path)
        return df

    def fetch_ohlcv(
        self,
        tickers: List[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch full OHLCV data (multi-level columns)."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._cache_path(tickers, start, end, "ohlcv")
        if self._cache_valid(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        with open(cache_path, "wb") as f:
            pickle.dump(raw, f)
        return raw

    def _fetch_with_retry(
        self,
        tickers: List[str],
        start: str,
        end: str,
        field: str,
        retries: int = 3,
        delay: float = 2.0,
    ) -> pd.DataFrame:
        for attempt in range(retries):
            try:
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    raise ValueError(f"Empty response for tickers: {tickers}")

                if isinstance(raw.columns, pd.MultiIndex):
                    if field in raw.columns.get_level_values(0):
                        df = raw[field]
                    else:
                        df = raw["Close"]
                else:
                    df = raw[[field]] if field in raw.columns else raw[["Close"]]
                    df.columns = tickers[:1]

                # Ensure single-ticker result is a proper DataFrame
                if isinstance(df, pd.Series):
                    df = df.to_frame(name=tickers[0])

                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                logger.info("Fetched %d rows for %d tickers from Yahoo", len(df), len(tickers))
                return df

            except Exception as exc:
                logger.warning("Yahoo fetch attempt %d failed: %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))

        raise RuntimeError(f"Failed to fetch Yahoo data after {retries} attempts")

    def _cache_path(self, tickers, start, end, field) -> str:
        key = f"{sorted(tickers)}-{start}-{end}-{field}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"yahoo_{h}.pkl")

    def _cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age < timedelta(hours=self.cache_expiry_hours)
