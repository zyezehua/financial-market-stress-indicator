"""Master data pipeline: fetch, align, and persist all raw data."""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import yaml

from .fetchers.fred import FredFetcher
from .fetchers.yahoo import YahooFetcher

logger = logging.getLogger(__name__)


def _load_config(settings_path: str, assets_path: str) -> tuple:
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    with open(assets_path) as f:
        assets = yaml.safe_load(f)
    return settings, assets


def _flatten_yahoo_tickers(assets: dict) -> list:
    """Collect all Yahoo tickers from assets.yaml into a flat list."""
    tickers = []
    y = assets.get("yahoo", {})
    for category, content in y.items():
        if isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, str):
                    tickers.append(v)
                elif isinstance(v, list):
                    tickers.extend(v)
        elif isinstance(content, list):
            tickers.extend(content)
        elif isinstance(content, str):
            tickers.append(content)
    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


def _flatten_fred_series(assets: dict) -> list:
    series = []
    for category, content in assets.get("fred", {}).items():
        series.extend(content.values())
    return list(dict.fromkeys(series))


class DataPipeline:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        assets_path: str = "config/assets.yaml",
        fred_api_key: Optional[str] = None,
    ):
        self.settings, self.assets = _load_config(settings_path, assets_path)

        cache_dir = self.settings["data"]["cache_dir"]
        expiry = self.settings["data"]["cache_expiry_hours"]

        self.yahoo = YahooFetcher(cache_dir=cache_dir, cache_expiry_hours=expiry)

        key = fred_api_key or os.getenv(self.settings["fred"]["api_key_env"], "")
        if not key:
            logger.warning(
                "FRED API key not found. Set %s env var or pass fred_api_key.",
                self.settings["fred"]["api_key_env"],
            )
        self.fred = FredFetcher(api_key=key, cache_dir=cache_dir, cache_expiry_hours=expiry)

        self.start_date = self.settings["data"]["start_date"]
        self.processed_dir = self.settings["data"]["processed_dir"]
        os.makedirs(self.processed_dir, exist_ok=True)

    def run(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        start = start or self.start_date
        end = end or datetime.today().strftime("%Y-%m-%d")

        logger.info("Running data pipeline: %s → %s", start, end)

        yahoo_tickers = _flatten_yahoo_tickers(self.assets)
        fred_series = _flatten_fred_series(self.assets)

        prices = self.yahoo.fetch(yahoo_tickers, start=start, end=end)
        macro = self.fred.fetch(fred_series, start=start, end=end)

        prices, macro = self._align(prices, macro)

        self._save(prices, os.path.join(self.processed_dir, "prices.parquet"))
        self._save(macro, os.path.join(self.processed_dir, "macro.parquet"))

        logger.info(
            "Pipeline complete. prices=%s macro=%s",
            prices.shape,
            macro.shape,
        )
        return {"prices": prices, "macro": macro}

    def update(self) -> Dict[str, pd.DataFrame]:
        """Incremental update: fetch new data and append to existing processed files."""
        prices_path = os.path.join(self.processed_dir, "prices.parquet")
        macro_path  = os.path.join(self.processed_dir, "macro.parquet")

        if os.path.exists(prices_path):
            existing_prices = pd.read_parquet(prices_path)
            existing_macro  = pd.read_parquet(macro_path)
            # Start one day after the last date we have
            last_date = existing_prices.index[-1]
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            end   = datetime.today().strftime("%Y-%m-%d")

            if start > end:
                logger.info("Data already up to date (%s). No fetch needed.", last_date.date())
                return {"prices": existing_prices, "macro": existing_macro}

            logger.info("Incremental update from %s → %s", start, end)

            yahoo_tickers = _flatten_yahoo_tickers(self.assets)
            fred_series   = _flatten_fred_series(self.assets)

            new_prices = self.yahoo.fetch(yahoo_tickers, start=start, end=end)
            new_macro  = self.fred.fetch(fred_series, start=start, end=end)

            if new_prices.empty:
                logger.info("No new market data yet for %s (market may be closed).", start)
                return {"prices": existing_prices, "macro": existing_macro}

            # Append and deduplicate (keep latest values for any overlapping dates)
            prices = pd.concat([existing_prices, new_prices])
            prices = prices[~prices.index.duplicated(keep="last")].sort_index()
            macro  = pd.concat([existing_macro, new_macro])
            macro  = macro[~macro.index.duplicated(keep="last")].sort_index().ffill()

            self._save(prices, prices_path)
            self._save(macro, macro_path)
            logger.info("Update complete. Total rows: prices=%d macro=%d", len(prices), len(macro))
            return {"prices": prices, "macro": macro}

        # No existing data — run full pipeline
        return self.run()

    @staticmethod
    def _align(
        prices: pd.DataFrame, macro: pd.DataFrame
    ) -> tuple:
        """Align both DataFrames to the intersection of business days."""
        prices.index = pd.to_datetime(prices.index)
        macro.index = pd.to_datetime(macro.index)

        common = prices.index.intersection(macro.index)
        if common.empty:
            # Fall back to union with forward-fill if no exact overlap
            combined_idx = prices.index.union(macro.index)
            prices = prices.reindex(combined_idx).ffill()
            macro = macro.reindex(combined_idx).ffill()
            return prices, macro

        prices = prices.loc[common]
        macro = macro.loc[common]
        # Forward-fill remaining NaNs (FRED series sometimes lag)
        macro = macro.ffill()
        prices = prices.ffill()
        return prices, macro

    @staticmethod
    def _save(df: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        logger.info("Saved %s rows to %s", len(df), path)

    @staticmethod
    def load_prices(processed_dir: str = "data/processed") -> pd.DataFrame:
        return pd.read_parquet(os.path.join(processed_dir, "prices.parquet"))

    @staticmethod
    def load_macro(processed_dir: str = "data/processed") -> pd.DataFrame:
        return pd.read_parquet(os.path.join(processed_dir, "macro.parquet"))
