"""
Lightweight on-disk feature store with versioning.

Features are persisted in Parquet format (partitioned by symbol/interval) to
avoid recomputation across training runs and to make feature lineage
traceable via metadata. The store records the feature version, horizon,
source time range, and computation time for reproducibility.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

DEFAULT_FEATURE_STORE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "feature_store"
)


class FeatureStore:
    """
    Simple versioned feature store backed by Parquet files.

    Files are partitioned by ``symbol`` and ``interval`` to enable selective
    refreshes. A sidecar JSON file keeps lineage metadata so training can
    verify compatibility before reusing cached features.
    """

    def __init__(self, root: str = DEFAULT_FEATURE_STORE_DIR, version: str = "v1") -> None:
        self.root = os.path.abspath(root)
        self.version = version

    def _paths(self, symbol: str, interval: str, horizon: int) -> tuple[str, str]:
        directory = os.path.join(self.root, symbol.upper(), interval)
        filename = f"{symbol.upper()}_{interval}_h{horizon}_{self.version}.parquet"
        return directory, os.path.join(directory, filename)

    @staticmethod
    def _meta_path(parquet_path: str) -> str:
        return f"{parquet_path}.meta.json"

    def load(
        self,
        symbol: str,
        interval: str,
        horizon: int,
        expected_end: Optional[str] = None,
        min_rows: Optional[int] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[dict[str, Any]]]:
        """
        Load cached features if they match the current version and freshness.
        """
        _, parquet_path = self._paths(symbol, interval, horizon)
        meta_path = self._meta_path(parquet_path)
        if not os.path.exists(parquet_path) or not os.path.exists(meta_path):
            return None, None

        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None, None

        if meta.get("version") != self.version:
            return None, None
        if expected_end and meta.get("source_end") != expected_end:
            return None, None

        df = pd.read_parquet(parquet_path)
        if min_rows and len(df) < min_rows:
            return None, None
        return df, meta

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        horizon: int,
        feature_version: str,
        source_start: str,
        source_end: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Persist features plus lineage metadata.
        """
        directory, parquet_path = self._paths(symbol, interval, horizon)
        os.makedirs(directory, exist_ok=True)
        df.to_parquet(parquet_path, index=True)

        meta = {
            "version": self.version,
            "feature_version": feature_version,
            "symbol": symbol.upper(),
            "interval": interval,
            "horizon": horizon,
            "source_start": source_start,
            "source_end": source_end,
            "rows": len(df),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = self._meta_path(parquet_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return parquet_path, meta
