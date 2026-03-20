"""
Training script for the XGBoost crypto trend prediction models.

Usage
-----
    python -m src.train

One model per symbol is trained using historical 1-hour Binance klines and
saved to the ``models/`` directory at the project root.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from sklearn.metrics import accuracy_score, roc_auc_score

from .data_fetcher import SYMBOLS, fetch_klines
from .feature_engineering import build_features, FEATURE_VERSION
from .feature_store import FeatureStore, DEFAULT_FEATURE_STORE_DIR
from .model import CryptoTrendModel, XGBOOST_PARAMS


TARGET_CONFIGS = [
    {"name": "primitive", "target_col": "target", "variant": None},
]

MIN_TRAIN_SAMPLES = 100
DEFAULT_HORIZON = 4

REGULARIZATION_GRID = [
    {
        **XGBOOST_PARAMS,
        "max_depth": 2,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "min_child_weight": 10,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
    },
    {
        **XGBOOST_PARAMS,
        "max_depth": 3,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_weight": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
    },
    {
        **XGBOOST_PARAMS,
        "max_depth": 2,
        "n_estimators": 800,
        "learning_rate": 0.02,
        "reg_alpha": 3.0,
        "reg_lambda": 15.0,
        "min_child_weight": 15,
        "subsample": 0.5,
        "colsample_bytree": 0.4,
        "gamma": 3.0,
    },
    {
        **XGBOOST_PARAMS,
        "max_depth": 4,
        "n_estimators": 150,
        "learning_rate": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
]


def _safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return None


def evaluate_baselines(df_feat, target_col: str) -> dict:
    """Compare simple baselines (persistence, momentum, crossover)."""
    y_true = df_feat[target_col].values
    required_cols = {"return_24h", "roc_6", "ema_7_21_cross"}
    if missing := required_cols - set(df_feat.columns):
        return {
            "missing_features": {
                "columns": sorted(missing),
                "accuracy": None,
                "auc": None,
            }
        }
    baselines = {
        "positive_return_24h": (df_feat["return_24h"] > 0).astype(int).values,
        "momentum_roc6": (df_feat["roc_6"] > 0).astype(int).values,
        "ema7_gt_ema21": (df_feat["ema_7_21_cross"] > 0).astype(int).values,
    }
    results: dict[str, dict[str, float | None]] = {}
    for name, preds in baselines.items():
        acc = float(accuracy_score(y_true, preds))
        auc = _safe_auc(y_true, preds)
        results[name] = {"accuracy": acc, "auc": auc}
    return results


def train_symbol(
    symbol: str,
    interval: str = "1h",
    limit: int = 5000,
    n_splits: int = 5,
    verbose: bool = True,
    use_feature_store: bool = True,
    feature_store_dir: Optional[str] = None,
    incremental: bool = False,
    incremental_window: int = 500,
    incremental_rounds: int = 200,
    bayes_trials: int = 20,
) -> dict:
    """
    Fetch data, build features, train and save a model for *symbol*.

    Parameters
    ----------
    symbol : str
        Trading-pair symbol (e.g. 'BTCUSDT').
    interval : str
        Binance kline interval.
    limit : int
        Number of historical candles to fetch (fetched in 1 000-sized batches).
    n_splits : int
        Walk-forward cross-validation folds.
    verbose : bool
        Print progress to stdout.
    use_feature_store : bool
        Whether to reuse or persist features via the on-disk feature store.
    feature_store_dir : str | None
        Custom directory for the feature store (default: project-level store).
    incremental : bool
        If True, attempt an incremental update of an existing model rather than
        a full retrain.
    incremental_window : int
        Number of most recent samples to use for incremental updates.
    incremental_rounds : int
        Additional boosting rounds when performing incremental updates.
    bayes_trials : int
        Optuna trials to run when augmenting the grid search with Bayesian search.

    Returns
    -------
    dict
        Training metrics, e.g. ``{"symbol": ..., "oof_auc": ..., "train_accuracy": ..., "test_accuracy": ...}``.
    """
    if verbose:
        print(f"\nTraining model for {symbol} …")

    df_raw = fetch_klines(symbol, interval=interval, limit=limit)
    btc_ref = None
    if symbol.upper() != "BTCUSDT":
        btc_ref = fetch_klines("BTCUSDT", interval=interval, limit=limit)

    store = FeatureStore(
        root=feature_store_dir or DEFAULT_FEATURE_STORE_DIR, version=FEATURE_VERSION
    )
    df_feat = None
    cached_from_store = False
    latest_raw = df_raw.index.max()
    if use_feature_store and not df_raw.empty:
        cached, meta = store.load(
            symbol,
            interval,
            horizon=DEFAULT_HORIZON,
            expected_end=latest_raw.isoformat(),
            min_rows=MIN_TRAIN_SAMPLES,
        )
        if cached is not None:
            df_feat = cached
            cached_from_store = True
            if verbose:
                print(
                    f"  [{symbol}] loaded {len(df_feat)} cached feature rows "
                    f"(version={meta.get('feature_version')})"
                )

    if df_feat is None:
        df_feat = build_features(
            df_raw,
            horizon=DEFAULT_HORIZON,
            symbol=symbol,
            interval=interval,
            limit=limit,
            btc_df=btc_ref,
        )
        if use_feature_store and latest_raw is not None:
            store.save(
                df_feat,
                symbol=symbol,
                interval=interval,
                horizon=DEFAULT_HORIZON,
                feature_version=FEATURE_VERSION,
                source_start=df_raw.index.min().isoformat(),
                source_end=latest_raw.isoformat(),
            )
    df_feat = df_feat.sort_index()

    if len(df_feat) < MIN_TRAIN_SAMPLES:
        msg = (
            f"  WARNING: only {len(df_feat)} samples for {symbol}. "
            "Skipping to avoid unreliable training."
        )
        print(msg, file=sys.stderr)
        return [
            {
                "symbol": symbol,
                "target": cfg["name"],
                "status": "skipped",
                "reason": msg,
            }
            for cfg in TARGET_CONFIGS
        ]

    results = []
    for cfg in TARGET_CONFIGS:
        target_col = cfg["target_col"]
        variant = cfg["variant"]
        model = CryptoTrendModel(
            symbol=symbol,
            variant=variant,
            target_column=target_col,
            importance_threshold=0.01,
            val_gap=24,
        )

        if incremental:
            try:
                model.load()
                window = min(len(df_feat), max(incremental_window, MIN_TRAIN_SAMPLES))
                update_df = df_feat.tail(window)
                inc_metrics = model.incremental_fit(
                    update_df,
                    extra_rounds=incremental_rounds,
                    verbose=verbose,
                )
                path = model.save()
                baselines = evaluate_baselines(df_feat, target_col=target_col)
                if verbose:
                    print(
                        f"  [{symbol}] incremental update applied with "
                        f"{len(update_df)} rows and +{incremental_rounds} rounds"
                    )
                results.append(
                    {
                        "symbol": symbol,
                        "target": cfg["name"],
                        "status": "incremental",
                        "model_path": path,
                        "baselines": baselines,
                        "feature_cache": cached_from_store,
                        **inc_metrics,
                    }
                )
                continue
            except FileNotFoundError:
                if verbose:
                    print(
                        f"  [{symbol}] no existing model found; falling back to full training."
                    )
        metrics = model.train(
            df_feat,
            n_splits=n_splits,
            verbose=verbose,
            target_column=target_col,
            param_grid=REGULARIZATION_GRID,
            bayes_trials=bayes_trials,
        )
        path = model.save()
        baselines = evaluate_baselines(df_feat, target_col=target_col)

        if verbose:
            train_acc = metrics.get("train_accuracy")
            test_acc = metrics.get("test_accuracy")
            train_prec = metrics.get("train_precision")
            test_prec = metrics.get("test_precision")
            train_rec = metrics.get("train_recall")
            test_rec = metrics.get("test_recall")
            train_f1 = metrics.get("train_f1")
            test_f1 = metrics.get("test_f1")
            train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
            test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            train_prec_fmt = f"{train_prec:.4f}" if train_prec is not None else "N/A"
            test_prec_fmt = f"{test_prec:.4f}" if test_prec is not None else "N/A"
            train_rec_fmt = f"{train_rec:.4f}" if train_rec is not None else "N/A"
            test_rec_fmt = f"{test_rec:.4f}" if test_rec is not None else "N/A"
            train_f1_fmt = f"{train_f1:.4f}" if train_f1 is not None else "N/A"
            test_f1_fmt = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
            print(
                f"  Metrics ({cfg['name']}) — "
                f"train: acc={train_acc_fmt} prec={train_prec_fmt} rec={train_rec_fmt} f1={train_f1_fmt}  "
                f"test: acc={test_acc_fmt} prec={test_prec_fmt} rec={test_rec_fmt} f1={test_f1_fmt}"
            )
            if metrics.get("search_summary"):
                ss = metrics["search_summary"]
                print(
                    f"    search: best={ss.get('best_source')} "
                    f"grid_best={ss.get('grid_best_auc')} bayes_best={ss.get('bayes_best_auc')}"
                )
            if metrics.get("performance_delta") is not None:
                print(
                    f"    walk-forward drift: {metrics['performance_delta']:+.4f} "
                    "(positive = improvement from first to last test window)"
                )
            print(f"  Feature columns used ({cfg['name']}): {len(metrics['features'])}")

        if verbose:
            print(f"  Model saved to {path}")

        results.append(
            {
                "symbol": symbol,
                "target": cfg["name"],
                "status": "ok",
                "model_path": path,
                "baselines": baselines,
                "feature_cache": cached_from_store,
                **metrics,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for crypto trend prediction."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to train (default: all 10).",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Binance kline interval (default: 1h).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Number of historical candles per symbol (walk-back up to limit).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Time-series CV folds (default: 5).",
    )
    parser.add_argument(
        "--bayes-trials",
        type=int,
        default=20,
        help="Optuna Bayesian search trials to run alongside the grid search.",
    )
    parser.add_argument(
        "--no-feature-store",
        action="store_true",
        help="Disable the feature store cache (recompute features every run).",
    )
    parser.add_argument(
        "--feature-store-dir",
        default=None,
        help="Custom directory for persisted features (default: project feature_store/).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Apply an incremental update to an existing model instead of full retraining.",
    )
    parser.add_argument(
        "--incremental-window",
        type=int,
        default=500,
        help="Rows from the tail of the dataset used for incremental updates.",
    )
    parser.add_argument(
        "--incremental-rounds",
        type=int,
        default=200,
        help="Additional boosting rounds when incrementally updating.",
    )
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        results.extend(
            train_symbol(
                symbol,
                interval=args.interval,
                limit=args.limit,
                n_splits=args.splits,
                use_feature_store=not args.no_feature_store,
                feature_store_dir=args.feature_store_dir,
                incremental=args.incremental,
                incremental_window=args.incremental_window,
                incremental_rounds=args.incremental_rounds,
                bayes_trials=args.bayes_trials,
            )
        )

    print("\n=== Training Summary ===")
    for r in results:
        auc = f"{r['oof_auc']:.4f}" if r.get("oof_auc") is not None else "N/A"
        train_acc = r.get("train_accuracy")
        test_acc = r.get("test_accuracy")
        train_prec = r.get("train_precision")
        test_prec = r.get("test_precision")
        train_rec = r.get("train_recall")
        test_rec = r.get("test_recall")
        train_f1 = r.get("train_f1")
        test_f1 = r.get("test_f1")
        train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
        test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        train_prec_fmt = f"{train_prec:.4f}" if train_prec is not None else "N/A"
        test_prec_fmt = f"{test_prec:.4f}" if test_prec is not None else "N/A"
        train_rec_fmt = f"{train_rec:.4f}" if train_rec is not None else "N/A"
        test_rec_fmt = f"{test_rec:.4f}" if test_rec is not None else "N/A"
        train_f1_fmt = f"{train_f1:.4f}" if train_f1 is not None else "N/A"
        test_f1_fmt = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
        print(
            f"  {r['symbol']:<10} [{r['target']:<11}] "
            f"OOF AUC={auc}  "
            f"train(acc/prec/rec/f1)={train_acc_fmt}/{train_prec_fmt}/{train_rec_fmt}/{train_f1_fmt}  "
            f"test(acc/prec/rec/f1)={test_acc_fmt}/{test_prec_fmt}/{test_rec_fmt}/{test_f1_fmt}  "
            f"status={r['status']}"
        )
        if r.get("baselines"):
            for name, stats in r["baselines"].items():
                auc_val = stats.get("auc")
                auc_fmt = f"{auc_val:.4f}" if auc_val is not None else "N/A"
                acc_val = stats.get("accuracy")
                acc_fmt = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                extra = ""
                if name == "missing_features":
                    cols = ",".join(stats.get("columns", []))
                    if cols:
                        extra = f" (missing: {cols})"
                print(
                    f"      baseline {name:<16} acc={acc_fmt} auc={auc_fmt}{extra}"
                )


if __name__ == "__main__":
    main()
