"""
XGBoost wrapper for the crypto trend prediction model.

Each symbol gets its own trained XGBoostClassifier. Models are saved to /
loaded from disk using joblib so that predictions can be served without
re-training.
"""

from __future__ import annotations

import os
import warnings
from inspect import signature
import joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV

from .feature_engineering import FEATURE_COLUMNS

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

XGBOOST_PARAMS: dict = {
    "n_estimators": 1200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 4,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

MIN_VAL_SAMPLES = 10
MIN_VAL_RATIO = 0.1
MAX_VAL_RATIO = 0.2
IMPORTANCE_ESTIMATORS_CAP = 300
IMPORTANCE_ESTIMATORS_BASE = 200
STABILITY_CV_CUTOFF = 1.5
MIN_REGIME_SAMPLES = 20
MIN_STABLE_FEATURES = 10
BULL_BEAR_THRESHOLD = 1.0
VOLATILITY_THRESHOLD_METHOD = "median"

_FIT_PARAMS = signature(XGBClassifier.fit).parameters
SUPPORTS_CALLBACKS = "callbacks" in _FIT_PARAMS
SUPPORTS_EARLY_STOPPING = "early_stopping_rounds" in _FIT_PARAMS


class CryptoTrendModel:
    """
    A thin wrapper around ``XGBClassifier`` that adds cross-validation
    reporting, persistence helpers and a clean predict API.

    Parameters
    ----------
    symbol : str
        The trading-pair symbol this model is trained for.
    model_dir : str
        Directory used for saving / loading model files.
    params : dict | None
        XGBoost hyper-parameters.  Defaults to ``XGBOOST_PARAMS``.
    variant : str | None
        Optional variant tag used to distinguish multiple targets per symbol.
    target_column : str
        Name of the target column inside the feature DataFrame.
    """

    def __init__(
        self,
        symbol: str,
        model_dir: str = DEFAULT_MODEL_DIR,
        params: dict | None = None,
        top_features: int = 20,
        early_stopping_rounds: int = 100,
        variant: str | None = None,
        target_column: str = "target",
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        val_gap: int = 0,
        n_bag_models: int = 3,
    ) -> None:
        self.symbol = symbol
        self.model_dir = model_dir
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.top_features = top_features
        self.early_stopping_rounds = early_stopping_rounds
        self.variant = variant
        self.target_column = target_column
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.val_gap = val_gap
        self.n_bag_models = n_bag_models
        self._model: XGBClassifier | None = None
        self._feature_columns: list[str] | None = None
        self._bag_models: list[XGBClassifier] = []

    def _select_top_features(
        self, df: pd.DataFrame, candidate_columns: list[str]
    ) -> list[str]:
        """Select most stable predictors via correlation with the target."""
        corr = (
            df[candidate_columns + [self.target_column]]
            .corr()[self.target_column]
            .drop(self.target_column)
            .abs()
            .dropna()
        )
        ranked = corr.sort_values(ascending=False)
        keep = ranked.head(self.top_features).index.tolist()
        if keep:
            return keep
        warnings.warn(
            f"[{self.symbol}] No usable correlations for feature selection; "
            f"using all {len(candidate_columns)} features."
        )
        return candidate_columns

    def _drop_correlated(
        self, df: pd.DataFrame, columns: list[str], threshold: float
    ) -> list[str]:
        """Remove highly correlated features to reduce redundancy."""
        if not columns:
            return columns
        corr_matrix = df[columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        to_drop = {
            column
            for column in upper_tri.columns
            if any(upper_tri[column].fillna(0) > threshold)
        }
        return [c for c in columns if c not in to_drop]

    def _rfecv_features(
        self, df: pd.DataFrame, columns: list[str], params: dict, n_splits: int
    ) -> list[str]:
        """Recursive feature elimination with time-series CV."""
        if len(columns) <= 2:
            return columns
        estimator = XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rfecv.fit(df[columns].values, df[self.target_column].values)
        mask = getattr(rfecv, "support_", None)
        if mask is None or not mask.any():
            return columns
        return [col for col, keep in zip(columns, mask) if keep]

    def _prune_with_importance(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        params: dict,
        threshold: float,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Use XGBoost feature importances to drop uninformative predictors."""
        quick_params = params.copy()
        quick_params["n_estimators"] = min(
            quick_params.get("n_estimators", IMPORTANCE_ESTIMATORS_BASE),
            IMPORTANCE_ESTIMATORS_CAP,
        )
        model = XGBClassifier(**quick_params)
        model.fit(
            df[feature_columns].values,
            df[self.target_column].values,
            verbose=False,
        )
        importances = model.feature_importances_
        ranked = sorted(
            zip(feature_columns, importances), key=lambda kv: kv[1], reverse=True
        )
        keep = [feat for feat, score in ranked if score > threshold]
        if not keep:
            keep = feature_columns
        return keep, ranked

    def _walk_forward_splits(self, n_samples: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Walk-forward validation splits with fixed validation window."""
        if n_samples < 2:
            return []
        val_size = max(MIN_VAL_SAMPLES, int(n_samples * MIN_VAL_RATIO))
        max_allowed = max(int(n_samples * MAX_VAL_RATIO), 1)
        val_size = min(val_size, max_allowed)

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(n_splits):
            end_val = n_samples - (n_splits - fold - 1) * val_size
            start_val = end_val - val_size
            if start_val <= 0 or start_val >= end_val:
                continue
            gap = max(int(self.val_gap), 0)
            if start_val <= gap:
                continue
            train_end = max(start_val - gap, 0)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(start_val, end_val)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((train_idx, val_idx))
        return splits

    def _cross_val_auc(
        self,
        df: pd.DataFrame,
        candidate_columns: list[str],
        params: dict,
        n_splits: int,
        verbose: bool,
        label: str | None = None,
        threshold: float | None = None,
        importance_log: dict[str, list[float]] | None = None,
        regime_scores: dict[str, list[float]] | None = None,
    ) -> tuple[float, list[float]]:
        splits = self._walk_forward_splits(len(df), n_splits)
        if not splits:
            return float("nan"), []

        oof_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            selected = self._select_top_features(train_df, candidate_columns)
            if threshold is not None:
                selected, _ = self._prune_with_importance(
                    train_df, selected, params, threshold=threshold
                )

            X_tr = train_df[selected].values
            X_val = val_df[selected].values
            y_tr = train_df[self.target_column].values
            y_val = val_df[self.target_column].values

            fold_model = XGBClassifier(**params)
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            if self.early_stopping_rounds and len(y_val) > 0:
                if SUPPORTS_CALLBACKS:
                    fit_kwargs["callbacks"] = [
                        EarlyStopping(
                            rounds=self.early_stopping_rounds,
                            save_best=True,
                            metric_name="auc",
                        )
                    ]
                elif SUPPORTS_EARLY_STOPPING:
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fold_model.fit(
                X_tr,
                y_tr,
                **fit_kwargs,
            )
            if importance_log is not None and hasattr(fold_model, "feature_importances_"):
                for feat, score in zip(selected, fold_model.feature_importances_):
                    importance_log.setdefault(feat, []).append(float(score))
            if regime_scores is not None:
                self._update_regime_scores(
                    regime_scores, fold_model, val_df, selected
                )
            proba = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            oof_aucs.append(auc)
            if verbose:
                prefix = f"[{self.symbol}]"
                if label:
                    prefix += f" {label}"
                print(f"  {prefix} fold {fold + 1}/{len(splits)}  AUC={auc:.4f}")

        return float(np.mean(oof_aucs)), oof_aucs

    def _regime_masks(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Define simple market regimes (trend + volatility)."""
        masks: dict[str, pd.Series] = {}
        if "price_over_ema_200" in df.columns:
            price = df["price_over_ema_200"]
            masks["bull"] = price >= BULL_BEAR_THRESHOLD
            masks["bear"] = price < BULL_BEAR_THRESHOLD
        if "realized_volatility_24" in df.columns:
            vol = df["realized_volatility_24"]
            if VOLATILITY_THRESHOLD_METHOD == "median":
                threshold = vol.median()
            elif VOLATILITY_THRESHOLD_METHOD == "mean":
                threshold = vol.mean()
            else:
                threshold = vol.median()
            masks["high_vol"] = vol >= threshold
            masks["low_vol"] = vol < threshold
        return masks

    def _update_regime_scores(
        self,
        store: dict[str, list[float]],
        model: XGBClassifier,
        val_df: pd.DataFrame,
        selected: list[str],
    ) -> None:
        """Track validation AUC per regime to measure stability."""
        if not selected or self.target_column not in val_df:
            return
        regimes = self._regime_masks(val_df)
        if not regimes:
            return
        proba = model.predict_proba(val_df[selected].values)[:, 1]
        y_true = val_df[self.target_column].values
        for name, mask in regimes.items():
            if mask is None or mask.sum() < MIN_REGIME_SAMPLES:
                continue
            try:
                auc = roc_auc_score(y_true[mask], proba[mask])
            except ValueError:
                continue
            store.setdefault(name, []).append(float(auc))

    def _regime_feature_consistency(
        self, df: pd.DataFrame, features: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Measure how stable feature/target relationships are across regimes.
        """
        regimes = self._regime_masks(df)
        consistency: dict[str, dict[str, float]] = {}
        if not regimes:
            return consistency
        for feat in features:
            corrs: list[float] = []
            for mask in regimes.values():
                subset = df.loc[mask, [feat, self.target_column]].dropna()
                if len(subset) < MIN_REGIME_SAMPLES:
                    continue
                corr = subset[feat].corr(subset[self.target_column])
                if pd.isna(corr):
                    continue
                corrs.append(abs(float(corr)))
            if not corrs:
                continue
            mean_corr = float(np.mean(corrs))
            cv = float(np.std(corrs) / (mean_corr + 1e-6))
            consistency[feat] = {
                "regime_corr_mean": mean_corr,
                "regime_corr_cv": cv,
                "regime_score": 1.0 / (1.0 + cv),
            }
        return consistency

    def _summarize_stability(
        self,
        importance_log: dict[str, list[float]],
        regime_scores: dict[str, list[float]],
        regime_consistency: dict[str, dict[str, float]],
    ) -> dict:
        """Aggregate stability metrics for downstream filtering/reporting."""
        feature_stability: dict[str, dict[str, float]] = {}
        for feat, scores in importance_log.items():
            if not scores:
                continue
            mean_imp = float(np.mean(scores))
            std_imp = float(np.std(scores))
            cv = float(std_imp / (mean_imp + 1e-6))
            base_score = mean_imp / (1.0 + cv)
            regime_score = regime_consistency.get(feat, {}).get("regime_score", 1.0)
            feature_stability[feat] = {
                "mean_importance": mean_imp,
                "std_importance": std_imp,
                "cv": cv,
                "regime_score": regime_score,
                "stability_score": base_score * regime_score,
            }
        regime_summary = {
            name: {"mean_auc": float(np.mean(vals)), "std_auc": float(np.std(vals))}
            for name, vals in regime_scores.items()
            if vals
        }
        return {
            "feature_stability": feature_stability,
            "regime_auc": regime_summary,
        }

    def _apply_stability_filter(
        self,
        features: list[str],
        stability: dict[str, dict[str, dict[str, float]]],
        min_keep: int = MIN_STABLE_FEATURES,
    ) -> list[str]:
        """Down-weight or drop unstable features across regimes/time."""
        if not features:
            return features
        stability_map = stability.get("feature_stability", {}) if stability else {}
        scored: list[tuple[str, float]] = []
        for feat in features:
            info = stability_map.get(feat)
            is_unstable = bool(info and info.get("cv", 0) > STABILITY_CV_CUTOFF)
            has_buffer = len(features) > min_keep
            if is_unstable and has_buffer:
                continue
            score = info.get("stability_score", 0.0) if info else 0.0
            scored.append((feat, score))
        if not scored:
            return features
        scored_sorted = sorted(scored, key=lambda kv: kv[1], reverse=True)
        keep_n = max(min_keep, len(scored_sorted))
        return [feat for feat, _ in scored_sorted[:keep_n]]

    def _walk_forward_optimization(
        self,
        df: pd.DataFrame,
        features: list[str],
        params: dict,
        train_ratio: float = 0.6,
        window_ratio: float = 0.1,
    ) -> list[dict]:
        """
        Train/validate/test on rolling windows to monitor degradation over time.
        """
        n = len(df)
        if n < MIN_VAL_SAMPLES * 3:
            return []
        train_size = max(int(n * train_ratio), MIN_VAL_SAMPLES * 2)
        window_size = max(int(n * window_ratio), MIN_VAL_SAMPLES)
        results: list[dict] = []
        start = 0
        while start + train_size + window_size < n:
            train_end = start + train_size
            val_end = train_end + window_size
            test_end = min(val_end + window_size, n)
            train_df = df.iloc[start:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:test_end]

            window_model = XGBClassifier(**params)
            window_model.fit(
                train_df[features].values, train_df[self.target_column].values, verbose=False
            )

            def _safe_auc(sub_df: pd.DataFrame) -> float:
                if len(sub_df) < MIN_VAL_SAMPLES:
                    return float("nan")
                try:
                    return float(
                        roc_auc_score(
                            sub_df[self.target_column].values,
                            window_model.predict_proba(sub_df[features].values)[:, 1],
                        )
                    )
                except ValueError:
                    return float("nan")

            val_auc = _safe_auc(val_df)
            test_auc = _safe_auc(test_df)
            results.append(
                {
                    "train_range": (int(start), int(train_end)),
                    "val_range": (int(train_end), int(val_end)),
                    "test_range": (int(val_end), int(test_end)),
                    "val_auc": val_auc,
                    "test_auc": test_auc,
                }
            )
            start += window_size
        return results

    def _bayes_optimize(
        self,
        df: pd.DataFrame,
        columns: list[str],
        n_splits: int,
        n_trials: int = 20,
        timeout: int | None = None,
        base_params: dict | None = None,
    ) -> dict:
        """Bayesian hyper-parameter search via Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = (base_params or self.params).copy()
            params.update(
                {
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
                    "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "n_estimators": trial.suggest_int("n_estimators", 400, 1600),
                }
            )
            mean_auc, _ = self._cross_val_auc(
                df,
                columns,
                params,
                n_splits=n_splits,
                verbose=False,
                label="bayes",
                threshold=self.importance_threshold,
            )
            return mean_auc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return {**self.params, **study.best_params}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        verbose: bool = True,
        target_column: str | None = None,
        param_grid: list[dict] | None = None,
        importance_threshold: float | None = None,
        bayes_trials: int = 15,
    ) -> dict:
        """
        Train on a feature DataFrame that contains a target column.

        Uses walk-forward cross-validation to report out-of-fold AUC before
        fitting the final model on all available data. Optionally runs a
        small hyper-parameter grid search and prunes uninformative features
        using XGBoost feature importances.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with ``FEATURE_COLUMNS`` columns and a target column.
        n_splits : int
            Number of folds for ``TimeSeriesSplit``.
        verbose : bool
            Whether to print per-fold metrics.
        target_column : str | None
            Which target column to use (defaults to current ``self.target_column``).
        param_grid : list[dict] | None
            Optional grid of parameter dicts to search; best mean OOF AUC wins.
        importance_threshold : float | None
            Drop features whose importance is at or below this threshold after
            a quick importance-only fit. Defaults to ``self.importance_threshold``.

        Returns
        -------
        dict
            Metrics including mean out-of-fold ROC-AUC, train/test accuracy,
            and the chosen parameter set.
        """
        if target_column:
            self.target_column = target_column
        threshold = (
            importance_threshold
            if importance_threshold is not None
            else self.importance_threshold
        )

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        available = self._drop_correlated(df, available, self.correlation_threshold)
        y = df[self.target_column].values

        # Handle class imbalance by up-weighting the minority class.
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        if pos + neg == 0:
            scale_pos_weight = 1.0
        elif pos == 0:
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = neg / pos

        base_params = self.params.copy()
        base_params["scale_pos_weight"] = scale_pos_weight
        candidate_specs: list[tuple[dict, str]] = []
        grid_params = param_grid if param_grid else [base_params]
        for idx, params in enumerate(grid_params, start=1):
            candidate = params.copy()
            candidate.setdefault("scale_pos_weight", scale_pos_weight)
            candidate_specs.append((candidate, f"grid#{idx}"))

        # Bayesian optimization to enrich candidate pool
        if bayes_trials and bayes_trials > 0:
            try:
                bayes_params = self._bayes_optimize(
                    df, available, n_splits=n_splits, n_trials=bayes_trials, base_params=base_params
                )
                bayes_params.setdefault("scale_pos_weight", scale_pos_weight)
                candidate_specs.append((bayes_params, "bayes"))
            except Exception as exc:
                if verbose:
                    print(f"  [{self.symbol}] Optuna search failed: {exc}")
        cv_results: list[dict] = []
        best_idx = 0
        best_mean_auc = -np.inf

        for idx, (params, source) in enumerate(candidate_specs):
            mean_auc, fold_aucs = self._cross_val_auc(
                df,
                available,
                params,
                n_splits=n_splits,
                verbose=verbose,
                label=source,
                threshold=threshold,
            )
            cv_results.append(
                {"params": params, "mean_auc": mean_auc, "folds": fold_aucs, "source": source}
            )
            score = mean_auc if not np.isnan(mean_auc) else -np.inf
            if score > best_mean_auc:
                best_idx = idx
                best_mean_auc = score

        self.params = candidate_specs[best_idx][0]
        best_source = candidate_specs[best_idx][1]
        mean_auc = best_mean_auc
        best_folds = cv_results[best_idx]["folds"]
        if verbose and len(candidate_specs) > 1:
            print(
                f"  [{self.symbol}] best candidate={best_source} "
                f"({best_idx + 1}/{len(candidate_specs)}) mean OOF AUC={mean_auc:.4f}"
            )

        if verbose:
            print(f"  [{self.symbol}] mean OOF AUC = {mean_auc:.4f}")

        grid_aucs = [r["mean_auc"] for r in cv_results if r.get("source", "").startswith("grid")]
        bayes_aucs = [r["mean_auc"] for r in cv_results if r.get("source") == "bayes"]
        search_summary = {
            "best_source": best_source,
            "grid_best_auc": float(np.nanmax(grid_aucs)) if grid_aucs else None,
            "bayes_best_auc": float(np.nanmax(bayes_aucs)) if bayes_aucs else None,
        }

        # Determine feature set on full data for final training
        selected = self._select_top_features(df, available)
        if len(selected) > 2:
            selected = self._drop_correlated(df, selected, self.correlation_threshold)
        if len(selected) > 2:
            selected = self._rfecv_features(
                df, selected, self.params, n_splits=n_splits
            )
        importance_ranking: list[tuple[str, float]] = []
        if threshold is not None:
            selected, importance_ranking = self._prune_with_importance(
                df, selected, self.params, threshold=threshold
            )
            if verbose:
                print(
                    f"  [{self.symbol}] importance pruning kept {len(selected)}/{len(available)} features"
                )

        # Stability metrics across time/regimes
        importance_log: dict[str, list[float]] = {}
        regime_auc_log: dict[str, list[float]] = {}
        self._cross_val_auc(
            df,
            selected,
            self.params,
            n_splits=n_splits,
            verbose=False,
            label="stability",
            threshold=threshold,
            importance_log=importance_log,
            regime_scores=regime_auc_log,
        )
        regime_consistency = self._regime_feature_consistency(df, selected)
        stability_report = self._summarize_stability(
            importance_log, regime_auc_log, regime_consistency
        )
        stability_filtered = self._apply_stability_filter(
            selected,
            stability_report,
            min_keep=max(MIN_STABLE_FEATURES, len(selected) // 2),
        )
        if stability_filtered:
            selected = stability_filtered

        walk_forward_report = self._walk_forward_optimization(
            df, selected, self.params
        )
        performance_drift = None
        if walk_forward_report:
            test_aucs = [
                w["test_auc"]
                for w in walk_forward_report
                if w.get("test_auc") is not None and not np.isnan(w.get("test_auc"))
            ]
            if len(test_aucs) >= 2:
                performance_drift = test_aucs[0] - test_aucs[-1]

        X = df[selected].values
        y = df[self.target_column].values

        # Re-train on the full data set
        # Hold out a modest validation slice: at least MIN_VAL_SAMPLES or
        # MIN_VAL_RATIO of the data, but never more than MAX_VAL_RATIO so most
        # history remains for training.
        min_required = max(MIN_VAL_SAMPLES, int(len(X) * MIN_VAL_RATIO))
        max_allowed = max(int(len(X) * MAX_VAL_RATIO), 1)
        val_size = min(min_required, max_allowed)
        if len(X) - val_size < 1:
            val_size = max(1, len(X) - 1)
        gap = max(int(self.val_gap), 0)
        val_start = len(X) - val_size
        desired_train_end = val_start - gap
        min_train = max(50, val_size, 1)
        max_train = max(val_start - 1, 1)
        # Clamp train_end to enforce gap (no overlap), minimum train size, and bounds
        train_end = min(max(desired_train_end, min_train), max_train)
        X_train, X_val = X[:train_end], X[val_start:]
        y_train, y_val = y[:train_end], y[val_start:]

        self._model = XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if len(X_val) > 0 else []
        fit_kwargs = {
            "eval_set": eval_set,
            "verbose": False,
        }
        if eval_set and self.early_stopping_rounds:
            if SUPPORTS_CALLBACKS:
                fit_kwargs["callbacks"] = [
                    EarlyStopping(
                        rounds=self.early_stopping_rounds,
                        save_best=True,
                        metric_name="auc",
                    )
                ]
            elif SUPPORTS_EARLY_STOPPING:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        # Bagging ensemble with different seeds (size = n_bag_models); base random_state is defined in XGBOOST_PARAMS
        self._bag_models = []
        for seed in range(self.n_bag_models):
            bag_params = self.params.copy()
            bag_params["random_state"] = bag_params.get("random_state", 42) + seed
            bag_model = XGBClassifier(**bag_params)
            bag_model.fit(X_train, y_train, **fit_kwargs)
            self._bag_models.append(bag_model)
        # Keep the first model for backward compatibility with save/load APIs
        self._model = self._bag_models[0]

        self._feature_columns = selected

        train_accuracy: float | None = None
        test_accuracy: float | None = None
        train_precision: float | None = None
        train_recall: float | None = None
        train_f1: float | None = None
        test_precision: float | None = None
        test_recall: float | None = None
        test_f1: float | None = None

        def _compute_prf(true, pred):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true,
                pred,
                average="binary",
                zero_division=0,
            )
            return float(precision), float(recall), float(f1)
        if len(X_train) > 0:
            train_pred = self._model.predict(X_train)
            train_accuracy = float(accuracy_score(y_train, train_pred))
            train_precision, train_recall, train_f1 = _compute_prf(y_train, train_pred)
        if len(X_val) > 0:
            test_pred = self._model.predict(X_val)
            test_accuracy = float(accuracy_score(y_val, test_pred))
            test_precision, test_recall, test_f1 = _compute_prf(y_val, test_pred)

        feature_importance = []
        if hasattr(self._model, "feature_importances_"):
            feature_importance = sorted(
                zip(selected, self._model.feature_importances_),
                key=lambda kv: kv[1],
                reverse=True,
            )

        if verbose:
            acc_msg = []
            if train_accuracy is not None:
                acc_msg.append(
                    f"train acc={train_accuracy:.4f} "
                    f"prec={train_precision:.4f} rec={train_recall:.4f} f1={train_f1:.4f}"
                )
            if test_accuracy is not None:
                acc_msg.append(
                    f"test acc={test_accuracy:.4f} "
                    f"prec={test_precision:.4f} rec={test_recall:.4f} f1={test_f1:.4f}"
                )
            if acc_msg:
                print(f"  [{self.symbol}] " + "  ".join(acc_msg))

        return {
            "oof_auc": mean_auc,
            "fold_aucs": best_folds,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "params": self.params,
            "features": selected,
            "importance_ranking": importance_ranking or feature_importance,
            "target_column": self.target_column,
            "search_summary": search_summary,
            "stability_report": stability_report,
            "walk_forward": walk_forward_report,
            "performance_drift": performance_drift,
        }

    def incremental_fit(
        self,
        df: pd.DataFrame,
        extra_rounds: int = 200,
        verbose: bool = True,
        target_column: str | None = None,
    ) -> dict:
        """
        Incrementally update an existing model using new data.

        Uses the existing booster as a warm-start via ``xgb_model`` so that only
        additional boosting rounds are added. This is useful for near-real-time
        updates between periodic full retraining cycles.
        """
        if target_column:
            self.target_column = target_column
        if not self._feature_columns:
            self._feature_columns = [
                c for c in df.columns if c != self.target_column and c in FEATURE_COLUMNS
            ]
        features = [c for c in self._feature_columns if c in df.columns]
        if not features:
            raise RuntimeError("No feature columns available for incremental update.")

        X = df[features].values
        y = df[self.target_column].values
        val_size = max(int(len(X) * MIN_VAL_RATIO), MIN_VAL_SAMPLES)
        val_size = min(val_size, max(int(len(X) * MAX_VAL_RATIO), 1))
        if len(X) <= val_size:
            val_size = max(1, len(X) - 1)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        updated_models: list[XGBClassifier] = []
        models_to_update = self._bag_models or []
        if not models_to_update and self._model:
            models_to_update = [self._model]

        for model in models_to_update:
            params = model.get_params()
            base_estimators = params.get("n_estimators", self.params.get("n_estimators", 200))
            params["n_estimators"] = base_estimators + extra_rounds
            updated = XGBClassifier(**params)
            booster = None
            try:
                booster = model.get_booster()
            except Exception:
                booster = None
            fit_kwargs = {
                "verbose": False,
            }
            if booster is not None:
                fit_kwargs["xgb_model"] = booster
            updated.fit(X_train, y_train, **fit_kwargs)
            updated_models.append(updated)

        if not updated_models:
            # If no pre-existing model, fall back to a small fresh fit.
            fallback_params = self.params.copy()
            fallback_params["n_estimators"] = fallback_params.get("n_estimators", 200) + extra_rounds
            updated = XGBClassifier(**fallback_params)
            updated.fit(X_train, y_train, verbose=False)
            updated_models.append(updated)

        self._bag_models = updated_models
        self._model = updated_models[0]
        self.params = self._model.get_params()

        inc_auc = None
        try:
            proba = self._model.predict_proba(X_val)[:, 1]
            inc_auc = float(roc_auc_score(y_val, proba))
        except ValueError:
            inc_auc = None

        metrics = {
            "oof_auc": inc_auc,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "features": features,
            "params": self.params,
            "target_column": self.target_column,
        }
        if verbose:
            print(
                f"  [{self.symbol}] incremental fit complete "
                f"(train={len(X_train)}, val={len(X_val)}, auc={inc_auc})"
            )
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return the probability of an upward price move for each row in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the same feature columns used during training.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability of the positive class (price goes up).
        """
        if self._model is None:
            raise RuntimeError(
                f"Model for {self.symbol} is not trained. "
                "Call train() or load() first."
            )
        if not self._feature_columns:
            raise RuntimeError(
                "Model feature columns not set or empty. "
                "Ensure train() or load() has completed successfully."
            )
        available = [c for c in self._feature_columns if c in df.columns]
        X = df[available].values
        models = self._bag_models or [self._model]
        probas = [m.predict_proba(X)[:, 1] for m in models]
        return np.mean(probas, axis=0)

    def predict_latest(self, df: pd.DataFrame) -> float:
        """
        Predict the probability of a price rise using only the most recent row.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (the last row is used).

        Returns
        -------
        float
            Probability in [0, 1].
        """
        proba = self.predict_proba(df.iloc[[-1]])
        return float(proba[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        suffix = f"_{self.variant}" if self.variant else ""
        return os.path.join(self.model_dir, f"{self.symbol}{suffix}.joblib")

    def save(self) -> str:
        """Persist model to disk. Returns the saved file path."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "model": self._model,
            "bag_models": self._bag_models,
            "feature_columns": self._feature_columns,
            "symbol": self.symbol,
            "params": self.params,
            "variant": self.variant,
            "target_column": self.target_column,
        }
        path = self._model_path()
        joblib.dump(payload, path)
        return path

    def load(self) -> "CryptoTrendModel":
        """Load a previously saved model from disk. Returns self."""
        path = self._model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved model found for {self.symbol} at {path}. "
                "Run train.py first."
            )
        payload = joblib.load(path)
        self._model = payload["model"]
        self._bag_models = payload.get("bag_models", [self._model])
        self._feature_columns = payload["feature_columns"]
        self.params = payload.get("params", self.params)
        self.variant = payload.get("variant", self.variant)
        self.target_column = payload.get("target_column", self.target_column)
        return self
