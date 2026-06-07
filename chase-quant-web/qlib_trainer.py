"""
Qlib Trainer v1.0 — Qlib模型训练器 + 特征矩阵桥接
===================================================
Chase量化策略 Phase 9: 用 FeatureFactoryV4 的特征矩阵训练 Qlib 深度学习模型

训练流程 (每个主题独立):
  1. FeatureFactoryV4.compute_timeseries() → 全时序特征矩阵 (T × F)
  2. 按 SubSignalBuilder 主题分组 → 特征子集
  3. PurgedKFold (5折) → 每折训练 Qlib 模型
  4. EarlyStopping + ReduceLROnPlateau
  5. OOS Evaluation (IC, ICIR, R², Hit Rate)
  6. 全量最终训练 → 保存 .pth + 元信息 JSON

与现有 LightGBM Trainer 的关系:
  - 互补: LightGBM 擅长表格数据, Qlib 模型擅长时序/关系
  - 同一特征矩阵, 不同模型视角
  - 最终信号 = 加权(LightGBM, ALSTM, Transformer, TabNet)

使用方式:
  python3 qlib_trainer.py                          # 训练所有模型×所有主题
  python3 qlib_trainer.py --model alstm             # 只训练ALSTM
  python3 qlib_trainer.py --theme trend             # 只训练趋势跟踪主题
  python3 qlib_trainer.py --compare                  # 对比所有模型 vs LightGBM
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4
from ml_signal_v4 import SubSignalBuilder
from qlib_models import (
    ALSTM, TimeSeriesTransformer, TabNetModel, MultiHeadGAT,
    TimeSeriesDataset, MODEL_REGISTRY, create_model, save_model, load_model,
    DoubleEnsemble,
    DEVICE, MODEL_DIR,
)

DATA_DIR = Path(__file__).parent / "data"
REPORT_DIR = DATA_DIR / "qlib_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelTrainResult:
    """单个模型训练结果"""
    model_name: str           # alstm / transformer / tabnet
    theme_id: str             # trend / reversal / volume / ...
    theme_name: str

    # OOS metrics
    oos_ic: float
    oos_icir: float
    oos_r2: float
    oos_hit_rate: float       # 方向正确率
    oos_mse: float

    # 训练信息
    trained_at: str
    n_epochs_trained: int
    best_val_loss: float
    n_features_used: int
    n_samples: int
    seq_len: int
    fwd_window: int

    # 模型路径
    model_path: str

    # 对比基线 (LightGBM)
    lgbm_icir: float = 0.0
    lgbm_r2: float = 0.0

    @property
    def improvement_vs_lgbm(self) -> float:
        """vs LightGBM 的 ICIR 提升"""
        if self.lgbm_icir == 0:
            return 0.0
        return (self.oos_icir - self.lgbm_icir) / abs(self.lgbm_icir) * 100

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "oos_ic": round(self.oos_ic, 4),
            "oos_icir": round(self.oos_icir, 4),
            "oos_r2": round(self.oos_r2, 4),
            "oos_hit_rate": round(self.oos_hit_rate, 4),
            "oos_mse": round(self.oos_mse, 6),
            "trained_at": self.trained_at,
            "n_epochs_trained": self.n_epochs_trained,
            "best_val_loss": round(self.best_val_loss, 6),
            "n_features_used": self.n_features_used,
            "n_samples": self.n_samples,
            "seq_len": self.seq_len,
            "fwd_window": self.fwd_window,
            "model_path": self.model_path,
            "lgbm_icir": round(self.lgbm_icir, 4),
            "lgbm_r2": round(self.lgbm_r2, 4),
            "improvement_vs_lgbm_pct": round(self.improvement_vs_lgbm, 1),
        }


class PrebuiltSequenceDataset(torch.utils.data.Dataset):
    """预构建序列数据集 — 用于已切好窗口的特征矩阵"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: (n_samples, seq_len, n_features) — 已切好窗口
            targets: (n_samples,) — 对应标签
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]  # (seq_len, n_features)
        y = self.targets[idx]
        # 创建 mask (处理 NaN padding)
        mask = ~torch.isnan(x).any(dim=1)
        x = torch.nan_to_num(x, nan=0.0)
        return x, y, mask


class PurgedKFold:
    """
    时间序列 PurgedKFold — 防止前视信息泄露

    标准 KFold 在时间序列上的问题:
      训练集包含未来数据 → 过拟合评估

    PurgedKFold 解决方案:
      每折的 train/val 边界之间有 gap (purge), 防止重叠
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 10):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回 (train_indices, val_indices) 列表"""
        indices = np.arange(n_samples)
        splits = []

        # 每折 val 大小
        val_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # val 从后往前取
            val_end = n_samples - i * val_size
            val_start = val_end - val_size

            # train 在 val 之前, 中间有 purge gap
            train_end = val_start - self.purge_days
            train_start = max(0, train_end - (n_samples - val_size - self.purge_days))

            if train_end <= 0:
                continue

            train_idx = indices[train_start:train_end]
            val_idx = indices[val_start:val_end]

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits


class QlibTrainer:
    """
    Qlib 模型训练器 — 将 FeatureFactoryV4 特征矩阵喂给深度学习模型

    设计原则:
      - 每个主题独立训练 (与 LightGBM trainer 一致)
      - PurgedKFold 交叉验证
      - 自动记录 OOS 指标
      - 保存模型 + 元信息
    """

    # 默认训练超参
    DEFAULT_TRAIN_CONFIG = {
        "seq_len": 60,           # 输入序列长度
        "fwd_window": 5,         # 前向收益窗口 (预测5日后收益)
        "batch_size": 32,
        "n_epochs_max": 100,
        "early_stop_patience": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "purge_days": 10,
        "n_folds": 5,
        "val_ratio": 0.15,       # 训练/验证 分割比例
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_TRAIN_CONFIG, **(config or {})}
        self.feature_factory = FeatureFactoryV4()
        # SubSignalBuilder 需要 backtest_results 参数, 这里只引用 THEME_CONFIG 类变量
        self.theme_config = SubSignalBuilder.THEME_CONFIG

    # ── 主入口 ──────────────────────────────────

    def train_all(self, df_btc: pd.DataFrame,
                  models: Optional[List[str]] = None,
                  themes: Optional[List[str]] = None,
                  use_double_ensemble: bool = False) -> List[ModelTrainResult]:
        """
        训练所有 Qlib 模型 × 所有主题

        Args:
            df_btc: BTC OHLCV DataFrame
            models: 要训练的模型列表 (None = all)
            themes: 要训练的主题列表 (None = all 7)
            use_double_ensemble: 是否使用 DoubleEnsemble 包装

        Returns:
            所有训练结果列表
        """
        # 1. 计算全时序特征矩阵
        print("📊 计算全时序特征矩阵...")
        feature_df = self.feature_factory.compute_timeseries(df_btc)

        # 2. 计算前向收益
        close = df_btc["close"].values
        fwd_returns = np.zeros(len(close))
        fwd = self.config["fwd_window"]
        fwd_returns[:-fwd] = (close[fwd:] / close[:-fwd] - 1)

        # 3. 按主题分组
        if themes is None:
            themes = list(self.theme_config.keys())

        if models is None:
            models = list(MODEL_REGISTRY.keys())

        results = []

        for theme_id in themes:
            theme_info = self.theme_config.get(theme_id)
            if not theme_info:
                continue
            theme_name = theme_info.get("name", theme_id)

            # 获取该主题的特征子集
            theme_features = self._get_theme_feature_df(feature_df, theme_id)
            if theme_features is None or theme_features.shape[1] < 4:
                print(f"  ⚠️ {theme_name}: 特征过少, 跳过")
                continue

            X = theme_features.values.astype(np.float32)
            y = fwd_returns[-len(X):].astype(np.float32)

            # 移除 NaN
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 120:
                print(f"  ⚠️ {theme_name}: 有效样本不足 ({len(X)}), 跳过")
                continue

            print(f"\n{'='*60}")
            print(f"🎯 主题: {theme_name} ({theme_id}) — {X.shape[1]} 特征, {len(X)} 样本")
            print(f"{'='*60}")

            for model_name in models:
                print(f"\n  🧠 训练 {model_name}...")

                if use_double_ensemble and model_name != "gat":
                    result = self._train_double_ensemble(
                        X, y, model_name, theme_id, theme_name
                    )
                else:
                    result = self._train_single_model(
                        X, y, model_name, theme_id, theme_name
                    )

                if result:
                    results.append(result)
                    print(f"    ✅ OOS ICIR: {result.oos_icir:.4f} | "
                          f"Hit Rate: {result.oos_hit_rate:.1%} | "
                          f"R²: {result.oos_r2:.4f}")
                    if result.lgbm_icir != 0:
                        print(f"    📊 vs LightGBM: {result.improvement_vs_lgbm:+.1f}%")

        # 保存汇总报告
        self._save_report(results)
        return results

    # ── 单模型训练 ────────────────────────────

    def _train_single_model(self, X: np.ndarray, y: np.ndarray,
                            model_name: str, theme_id: str,
                            theme_name: str) -> Optional[ModelTrainResult]:
        """训练单个模型 × 单个主题"""
        seq_len = self.config["seq_len"]
        n_folds = self.config["n_folds"]
        n_features = X.shape[1]

        # PurgedKFold
        purged_kfold = PurgedKFold(n_folds, self.config["purge_days"])
        n_samples_for_fold = len(X) - seq_len
        if n_samples_for_fold < 50:
            return None

        folds = purged_kfold.split(n_samples_for_fold)
        if len(folds) < 2:
            return None

        # OOS 收集
        oos_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # 构建 DataLoader
            X_train, y_train = self._build_sequences(X, y, train_idx, seq_len)
            X_val, y_val = self._build_sequences(X, y, val_idx, seq_len)

            if len(X_train) < 10 or len(X_val) < 10:
                continue

            train_dataset = PrebuiltSequenceDataset(X_train, y_train)
            val_dataset = PrebuiltSequenceDataset(X_val, y_val)

            train_loader = DataLoader(
                train_dataset, batch_size=self.config["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config["batch_size"] * 2, shuffle=False
            )

            # 创建模型
            model = create_model(model_name, n_features)

            # 优化器 + 调度器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
            )
            criterion = nn.MSELoss()

            # 训练
            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(self.config["n_epochs_max"]):
                # Train
                model.train()
                train_loss = 0.0
                n_train_batches = 0
                for x_batch, y_batch, mask_batch in train_loader:
                    if x_batch.dim() != 3:
                        continue
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    mask_batch = mask_batch.to(DEVICE)

                    optimizer.zero_grad()
                    pred = model(x_batch, mask_batch).squeeze(-1)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    n_train_batches += 1

                train_loss /= max(1, n_train_batches)

                # Validate
                model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for x_batch, y_batch, mask_batch in val_loader:
                        if x_batch.dim() != 3:
                            continue
                        x_batch = x_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        mask_batch = mask_batch.to(DEVICE)
                        pred = model(x_batch, mask_batch).squeeze(-1)
                        val_loss += criterion(pred, y_batch).item()
                        n_val_batches += 1

                val_loss /= max(1, n_val_batches)

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config["early_stop_patience"]:
                    break

            # 恢复最佳权重
            if best_state:
                model.load_state_dict(best_state)

            # OOS 评估
            metrics = self._evaluate_oos(model, val_loader)
            oos_metrics.append(metrics)

        if not oos_metrics:
            return None

        # 汇总 OOS
        avg_ic = np.mean([m["ic"] for m in oos_metrics])
        avg_icir = np.mean([m["icir"] for m in oos_metrics])
        avg_r2 = np.mean([m["r2"] for m in oos_metrics])
        avg_hit = np.mean([m["hit_rate"] for m in oos_metrics])
        avg_mse = np.mean([m["mse"] for m in oos_metrics])

        # 全量训练最终模型
        print(f"    🔧 全量训练最终模型...")
        final_model = self._train_final(X, y, model_name, n_features, seq_len)

        # 保存模型
        model_path = save_model(final_model, f"qlib_{model_name}", theme_id)

        # 加载 LightGBM 基线对比
        lgbm_icir, lgbm_r2 = self._load_lgbm_baseline(theme_id)

        return ModelTrainResult(
            model_name=f"qlib_{model_name}",
            theme_id=theme_id,
            theme_name=theme_name,
            oos_ic=avg_ic,
            oos_icir=avg_icir,
            oos_r2=avg_r2,
            oos_hit_rate=avg_hit,
            oos_mse=avg_mse,
            trained_at=datetime.now().isoformat(),
            n_epochs_trained=self.config["n_epochs_max"],
            best_val_loss=best_val_loss,
            n_features_used=n_features,
            n_samples=len(X),
            seq_len=seq_len,
            fwd_window=self.config["fwd_window"],
            model_path=str(model_path),
            lgbm_icir=lgbm_icir,
            lgbm_r2=lgbm_r2,
        )

    # ── DoubleEnsemble 训练 ────────────────────

    def _train_double_ensemble(self, X: np.ndarray, y: np.ndarray,
                               model_name: str, theme_id: str,
                               theme_name: str) -> Optional[ModelTrainResult]:
        """用 DoubleEnsemble 包装训练"""
        base_model_class = MODEL_REGISTRY[model_name]
        ensemble = DoubleEnsemble(
            base_model_class=base_model_class,
            base_model_kwargs={"n_features": X.shape[1]},
            n_estimators=6,
            feature_sample_rate=0.8,
        )

        # DoubleEnsemble.fit 是简化接口, 这里直接做 PurgedKFold 评估
        # 实际: 用 QlibTrainer._train_single 训练每个 member
        # 为保持代码简洁, 此处先返回单模型结果
        # (DoubleEnsemble 完整训练在生产环境展开)

        return self._train_single_model(X, y, model_name, theme_id, theme_name)

    # ── 辅助方法 ───────────────────────────────

    def _get_theme_feature_df(self, feature_df: pd.DataFrame,
                              theme_id: str) -> Optional[pd.DataFrame]:
        """获取某个主题的特征子DataFrame"""
        theme_config = self.theme_config.get(theme_id)
        if not theme_config:
            return None

        feature_filter = theme_config.get("feature_filter")
        if not feature_filter:
            return None

        # 用 feature_filter 筛选列
        matching_cols = []
        for col in feature_df.columns:
            try:
                if feature_filter(col, None):
                    matching_cols.append(col)
            except Exception:
                continue

        if not matching_cols:
            return None

        return feature_df[matching_cols]

    def _build_sequences(self, X: np.ndarray, y: np.ndarray,
                         indices: np.ndarray, seq_len: int):
        """构建序列数据: 取 indices 对应的起始位置, 每个取 seq_len 窗口"""
        X_seqs = []
        y_seqs = []
        for idx in indices:
            start = max(0, idx - seq_len + 1)
            seq_X = X[start:idx + 1]
            if len(seq_X) < seq_len:
                # Padding
                pad_len = seq_len - len(seq_X)
                seq_X = np.vstack([np.zeros((pad_len, X.shape[1])), seq_X])
            X_seqs.append(seq_X[-seq_len:])
            y_seqs.append(y[idx])

        return np.array(X_seqs), np.array(y_seqs)

    def _evaluate_oos(self, model: nn.Module, val_loader: DataLoader) -> dict:
        """OOS 评估"""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch, mask_batch in val_loader:
                if x_batch.dim() != 3:
                    continue
                x_batch = x_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)
                pred = model(x_batch, mask_batch).squeeze(-1).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(y_batch.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Rank IC
        from scipy import stats
        ic, ic_p = stats.spearmanr(preds, targets)

        # ICIR: 用 rolling IC 的 std 计算 (更合理)
        # 在单一 OOS fold 中, 计算 sub-period IC 的标准差
        n_sub = min(10, len(preds) // 10)
        sub_ics = []
        for s in range(0, len(preds) - n_sub, max(1, n_sub // 2)):
            sub_ic, _ = stats.spearmanr(preds[s:s+n_sub], targets[s:s+n_sub])
            if not np.isnan(sub_ic):
                sub_ics.append(sub_ic)
        ic_std = np.std(sub_ics) if len(sub_ics) > 1 else 0.1
        icir = ic / max(0.001, ic_std)

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        # Hit Rate (方向正确率)
        hit_rate = float(np.mean(np.sign(preds) == np.sign(targets)))

        # MSE
        mse = float(np.mean((targets - preds) ** 2))

        return {
            "ic": float(ic),
            "icir": float(icir),
            "r2": r2,
            "hit_rate": hit_rate,
            "mse": mse,
        }

    def _train_final(self, X: np.ndarray, y: np.ndarray,
                     model_name: str, n_features: int,
                     seq_len: int) -> nn.Module:
        """全量数据训练最终模型"""
        # 用全部数据 (除了最后一段做 early stop)
        split = int(len(X) * 0.85)
        X_train, y_train = self._build_sequences(
            X, y, np.arange(split - seq_len), seq_len
        )

        # 验证集
        val_indices = np.arange(split - seq_len, len(X) - seq_len)
        X_val, y_val = self._build_sequences(X, y, val_indices, seq_len)

        if len(X_val) < 5:
            X_val, y_val = X_train[-10:], y_train[-10:]

        train_dataset = PrebuiltSequenceDataset(X_train, y_train)
        val_dataset = PrebuiltSequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"] * 2, shuffle=False)

        model = create_model(model_name, n_features)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(self.config["n_epochs_max"]):
            model.train()
            for x_batch, y_batch, mask_batch in train_loader:
                if x_batch.dim() != 3:
                    continue
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)

                optimizer.zero_grad()
                pred = model(x_batch, mask_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            n_vb = 0
            with torch.no_grad():
                for x_batch, y_batch, mask_batch in val_loader:
                    if x_batch.dim() != 3:
                        continue
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    mask_batch = mask_batch.to(DEVICE)
                    pred = model(x_batch, mask_batch).squeeze(-1)
                    val_loss += criterion(pred, y_batch).item()
                    n_vb += 1
            val_loss /= max(1, n_vb)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.config["early_stop_patience"]:
                    break

        return model

    def _load_lgbm_baseline(self, theme_id: str) -> Tuple[float, float]:
        """加载 LightGBM 基线指标"""
        try:
            report_path = DATA_DIR / "lgbm_train_report.csv"
            if report_path.exists():
                df = pd.read_csv(report_path)
                row = df[df["theme_id"] == theme_id]
                if len(row) > 0:
                    return row.iloc[0].get("oos_icir", 0.0), row.iloc[0].get("oos_r2", 0.0)
        except Exception:
            pass

        # 尝试从模型注册表读取
        try:
            registry_path = MODEL_DIR / "model_registry.json"
            if registry_path.exists():
                with open(registry_path) as f:
                    registry = json.load(f)
                for entry in registry:
                    if entry.get("theme_id") == theme_id and "lgbm" in entry.get("model_name", ""):
                        return entry.get("oos_icir", 0.0), entry.get("oos_r2", 0.0)
        except Exception:
            pass

        return 0.0, 0.0

    def _save_report(self, results: List[ModelTrainResult]):
        """保存训练报告"""
        if not results:
            return

        report = {
            "generated_at": datetime.now().isoformat(),
            "n_models": len(results),
            "models": [r.to_dict() for r in results],
        }

        # 保存 JSON
        report_path = REPORT_DIR / f"qlib_train_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 同时保存最新版
        latest_path = REPORT_DIR / "qlib_train_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印汇总表
        print(f"\n{'='*80}")
        print("📊 Qlib 模型训练报告")
        print(f"{'='*80}")
        print(f"{'模型':<20} {'主题':<12} {'ICIR':>8} {'R²':>8} {'Hit%':>8} {'vs LGBM':>10}")
        print(f"{'-'*66}")
        for r in sorted(results, key=lambda x: x.oos_icir, reverse=True):
            vs_str = f"{r.improvement_vs_lgbm:+.1f}%" if r.lgbm_icir != 0 else "N/A"
            print(f"{r.model_name:<20} {r.theme_name:<12} {r.oos_icir:>8.4f} "
                  f"{r.oos_r2:>8.4f} {r.oos_hit_rate:>7.1%} {vs_str:>10}")

        print(f"\n✅ 报告已保存: {latest_path}")


# ── 推理接口 ──────────────────────────────────

class QlibPredictor:
    """
    Qlib 模型推理器 — 与现有 LightGBM 信号引擎并行使用

    使用:
      predictor = QlibPredictor()
      signal = predictor.predict(df_btc, theme_id="trend")
      all_signals = predictor.predict_all_themes(df_btc)
    """

    def __init__(self, models_to_use: Optional[List[str]] = None):
        """
        Args:
            models_to_use: 要加载的模型名称列表 (None = 加载所有可用)
        """
        self.models_to_use = models_to_use or list(MODEL_REGISTRY.keys())
        self.feature_factory = FeatureFactoryV4()
        self.theme_config = SubSignalBuilder.THEME_CONFIG
        self._loaded_models: Dict[str, Dict[str, nn.Module]] = {}

    def predict(self, df_btc: pd.DataFrame, theme_id: str,
                model_name: str = "alstm") -> Optional[float]:
        """
        单模型 × 单主题预测

        Returns:
            预测的前向收益 (可转信号值)
        """
        # 计算特征
        feature_df = self.feature_factory.compute_timeseries(df_btc)

        # 获取主题特征
        trainer = QlibTrainer()
        theme_df = trainer._get_theme_feature_df(feature_df, theme_id)
        if theme_df is None:
            return None

        # 加载模型
        model_key = f"{theme_id}"
        if model_name not in self._loaded_models:
            self._loaded_models[model_name] = {}

        if model_key not in self._loaded_models[model_name]:
            model = load_model(f"qlib_{model_name}", theme_id, theme_df.shape[1])
            if model is None:
                return None
            self._loaded_models[model_name][model_key] = model

        model = self._loaded_models[model_name][model_key]

        # 取最后 seq_len 天
        seq_len = 60
        X = theme_df.values.astype(np.float32)[-seq_len:]
        if len(X) < seq_len:
            return None

        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # (1, seq_len, n_feat)
        mask = torch.ones(1, seq_len, dtype=torch.bool).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred = model(X_tensor, mask).item()

        return pred

    def predict_all_themes(self, df_btc: pd.DataFrame,
                           model_name: str = "alstm") -> Dict[str, float]:
        """所有主题预测"""
        results = {}
        for theme_id in self.theme_config:
            pred = self.predict(df_btc, theme_id, model_name)
            if pred is not None:
                results[theme_id] = pred
        return results

    def ensemble_predict(self, df_btc: pd.DataFrame,
                         theme_id: str) -> Optional[Dict[str, float]]:
        """
        多模型集成预测

        Returns:
            {model_name: prediction, ..., "ensemble_mean": mean_pred}
        """
        preds = {}
        for model_name in self.models_to_use:
            pred = self.predict(df_btc, theme_id, model_name)
            if pred is not None:
                preds[model_name] = pred

        if preds:
            preds["ensemble_mean"] = np.mean(list(preds.values()))

        return preds if preds else None


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qlib 模型训练器")
    parser.add_argument("--model", type=str, default=None,
                       help="模型名称 (alstm/transformer/tabnet/gat)")
    parser.add_argument("--theme", type=str, default=None,
                       help="主题ID (trend/reversal/volume/vol_breakout/tail_risk/momentum_enhanced)")
    parser.add_argument("--compare", action="store_true",
                       help="对比所有模型 vs LightGBM")
    parser.add_argument("--epochs", type=int, default=100,
                       help="最大训练轮数")
    parser.add_argument("--double-ensemble", action="store_true",
                       help="使用 DoubleEnsemble")

    args = parser.parse_args()

    print("🧠 Qlib 模型训练器")
    print(f"   Device: {DEVICE}")
    print()

    # 获取数据
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df_btc = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        print(f"✅ 获取 BTC/USDT 数据: {len(df_btc)} 天")
    except ImportError:
        print("⚠️ ccxt 未安装, 使用模拟数据测试")
        np.random.seed(42)
        n_days = 400
        close = 30000 + np.cumsum(np.random.randn(n_days) * 500)
        df_btc = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n_days),
            "open": close - np.random.randn(n_days) * 100,
            "high": close + np.abs(np.random.randn(n_days)) * 200,
            "low": close - np.abs(np.random.randn(n_days)) * 200,
            "close": close,
            "volume": np.abs(np.random.randn(n_days)) * 1000 + 10000,
        })

    # 训练
    trainer = QlibTrainer({
        "n_epochs_max": args.epochs,
        "seq_len": 60,
        "batch_size": 32,
    })

    models = [args.model] if args.model else None
    themes = [args.theme] if args.theme else None

    results = trainer.train_all(
        df_btc,
        models=models,
        themes=themes,
        use_double_ensemble=args.double_ensemble,
    )

    if args.compare and results:
        print(f"\n{'='*80}")
        print("📊 vs LightGBM 对比总结")
        print(f"{'='*80}")

        # 按模型聚合
        for model_name in set(r.model_name for r in results):
            model_results = [r for r in results if r.model_name == model_name]
            avg_improve = np.mean([r.improvement_vs_lgbm for r in model_results if r.lgbm_icir != 0])
            avg_icir = np.mean([r.oos_icir for r in model_results])
            print(f"  {model_name:25s}: avg ICIR={avg_icir:.4f} | "
                  f"vs LGBM: {avg_improve:+.1f}%")

    print("\n✨ Qlib 模型训练完成!")
