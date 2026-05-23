"""FRED data fetcher with disk caching and business-day resampling."""

import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FredFetcher:
    def __init__(
        self,
        api_key: str,
        cache_dir: str = "data/raw",
        cache_expiry_hours: int = 24,
    ):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_expiry_hours = cache_expiry_hours
        os.makedirs(cache_dir, exist_ok=True)
        self._fred = None

    @property
    def fred(self):
        if self._fred is None:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def fetch(
        self,
        series_ids: List[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch multiple FRED series, resampled to business days."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._cache_path(series_ids, start, end)
        if self._cache_valid(cache_path):
            logger.info("Loading FRED data from cache: %s", cache_path)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        frames: Dict[str, pd.Series] = {}
        for sid in series_ids:
            try:
                s = self.fred.get_series(sid, observation_start=start, observation_end=end)
                s.name = sid
                frames[sid] = self._to_business_day(s)
                logger.info("Fetched FRED series %s: %d obs", sid, len(s))
            except Exception as exc:
                logger.warning("Failed to fetch FRED series %s: %s", sid, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames.values(), axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        return df

    def fetch_one(self, series_id: str, start: str, end: Optional[str] = None) -> pd.Series:
        df = self.fetch([series_id], start, end)
        if df.empty or series_id not in df.columns:
            return pd.Series(name=series_id, dtype=float)
        return df[series_id]

    def _to_business_day(self, s: pd.Series) -> pd.Series:
        """Forward-fill a series onto a business-day index."""
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")]
        bday_idx = pd.bdate_range(s.index.min(), s.index.max())
        return s.reindex(bday_idx).ffill()

    def _cache_path(self, series_ids: List[str], start: str, end: str) -> str:
        key = f"{sorted(series_ids)}-{start}-{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"fred_{h}.pkl")

    def _cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age < timedelta(hours=self.cache_expiry_hours)
