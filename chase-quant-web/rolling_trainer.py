"""
Rolling Trainer v2.0 — 滚动在线学习引擎 + 图漂移监控
========================================================
Chase量化策略 Phase 10+11: 自适应模型更新 + 资产关系图漂移检测

核心能力 (vs 静态训练):
  静态训练 (Phase 9): 训练一次 → 模型老化 → 表现退化
  滚动训练 (Phase 10): 定期自动重训 → 适应市场变化 → 持续优化
  图漂移监控 (Phase 11): 检测资产关系变化 → 触发图重建 → GATs 持续有效

设计灵感:
  - Qlib OnlineManager: 在线学习 + 模型更新
  - Walk-Forward Analysis: 滚动窗口 OOS 验证
  - 西蒙斯风格: "让模型和数据一起进化"

架构:
  ┌─────────────────────────────────────────────────────────────┐
  │                 RollingTrainer v2.0                          │
  │                                                             │
  │  ┌─────────┐   ┌──────────┐   ┌──────────────────┐         │
  │  │ Window  │ → │ Retrain  │ → │ Model Registry   │         │
  │  │ Manager │   │ Scheduler│   │ (versioned .pth) │         │
  │  └─────────┘   └──────────┘   └──────────────────┘         │
  │       │               │                │                    │
  │  expanding/      daily/weekly/      staleness               │
  │  sliding         monthly           detection                │
  │                                                             │
  │  ┌──────────────────┐   ┌──────────────────────────┐       │
  │  │ Asset Graph      │ → │ Graph Drift Detection    │       │
  │  │ Builder (Phase11)│   │ → 触发图重建 → GATs更新  │       │
  │  └──────────────────┘   └──────────────────────────┘       │
  └─────────────────────────────────────────────────────────────┘

使用方式:
  # 首次滚动训练 (全量)
  python3 rolling_trainer.py --init

  # 增量更新 (检查是否需要重训)
  python3 rolling_trainer.py --update

  # 强制全量重训
  python3 rolling_trainer.py --force

  # 查看滚动训练状态
  python3 rolling_trainer.py --status

  # 回测滚动窗口性能
  python3 rolling_trainer.py --backtest
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import pickle
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4
from ml_signal_v4 import SubSignalBuilder
from qlib_trainer import QlibTrainer, QlibPredictor, ModelTrainResult
from qlib_models import (
    ALSTM, TimeSeriesTransformer, TabNetModel, MultiHeadGAT,
    MODEL_REGISTRY, create_model, save_model, load_model,
    DEVICE, MODEL_DIR,
)

# Asset Graph (Phase 11)
try:
    from asset_graph import AssetGraphBuilder, CrossAssetGATPredictor
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"
ROLLING_DIR = DATA_DIR / "rolling"
ROLLING_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

class WindowMode(str, Enum):
    EXPANDING = "expanding"   # 窗口不断扩大 (保留全部历史)
    SLIDING = "sliding"       # 固定长度滑动窗口
    HYBRID = "hybrid"         # 2年滑动 + 保留关键拐点


class StalenessLevel(str, Enum):
    FRESH = "🟢 fresh"         # < 7 天
    WARM = "🟡 warm"           # 7-21 天
    STALE = "🟠 stale"         # 21-30 天
    EXPIRED = "🔴 expired"     # > 30 天


@dataclass
class RollingWindow:
    """单个滚动窗口的训练配置"""
    window_id: str                        # 窗口唯一ID (hash)
    start_date: str                       # 训练数据起始
    end_date: str                         # 训练数据结束
    n_samples: int                        # 训练样本数
    n_features: int                       # 特征数
    mode: WindowMode                      # expanding / sliding
    created_at: str                       # 窗口创建时间


@dataclass
class RollingModelSnapshot:
    """滚动训练产出的模型快照"""
    model_name: str                       # alstm / transformer / tabnet / lgbm
    theme_id: str                         # trend / reversal / volume / ...
    theme_name: str
    window_id: str                        # 所属窗口
    version: int                          # 版本号 (递增)
    trained_at: str                       # ISO timestamp
    data_start: str
    data_end: str
    n_samples: int
    n_epochs: int
    seq_len: int
    fwd_window: int

    # OOS 指标
    oos_ic: float
    oos_icir: float
    oos_r2: float
    oos_hit_rate: float

    # 路径
    model_path: str
    meta_path: str

    # 与上一版本对比
    icir_delta: float = 0.0               # 相比上一版本的ICIR变化
    icir_trend: str = "→"                 # ↑改善 / →持平 / ↓退化

    # 数据指纹 (检测数据漂移)
    data_hash: str = ""                   # 特征矩阵hash
    feature_drift: float = 0.0            # 特征分布偏移量

    @property
    def age_days(self) -> int:
        try:
            t = datetime.fromisoformat(self.trained_at)
            return (datetime.now() - t).days
        except Exception:
            return 999

    @property
    def staleness(self) -> StalenessLevel:
        d = self.age_days
        if d <= 7:
            return StalenessLevel.FRESH
        elif d <= 21:
            return StalenessLevel.WARM
        elif d <= 30:
            return StalenessLevel.STALE
        return StalenessLevel.EXPIRED

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "window_id": self.window_id,
            "version": self.version,
            "trained_at": self.trained_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "n_samples": self.n_samples,
            "n_epochs": self.n_epochs,
            "seq_len": self.seq_len,
            "oos_ic": round(self.oos_ic, 4),
            "oos_icir": round(self.oos_icir, 4),
            "oos_r2": round(self.oos_r2, 4),
            "oos_hit_rate": round(self.oos_hit_rate, 4),
            "icir_delta": round(self.icir_delta, 4),
            "icir_trend": self.icir_trend,
            "age_days": self.age_days,
            "staleness": self.staleness.value,
            "feature_drift": round(self.feature_drift, 4),
        }


# ═══════════════════════════════════════════════════════════════
# Rolling Window Manager
# ═══════════════════════════════════════════════════════════════

class RollingWindowManager:
    """
    管理滚动训练窗口 — 决定何时使用什么数据重训

    三种模式:
      expanding: [───────] → [──────────] → [─────────────]
        保留全部历史, 窗口不断扩大, 适合长期趋势

      sliding:        [────] → [────] → [────]
        固定窗口大小, 丢弃旧数据, 适合快速适应

      hybrid (默认): [──────────] + [拐点]
        2年滑动窗口 + 保留关键市场拐点, 兼顾长期和短期
    """

    DEFAULT_CONFIG = {
        "mode": "hybrid",
        "sliding_size_days": 730,       # 滑动窗口: 2年
        "min_samples": 200,              # 最少样本数
        "update_interval_days": 7,       # 更新间隔 (每周)
        "max_staleness_days": 21,        # 模型过期天数
        "retrain_threshold_icir_drop": 0.15,  # ICIR下降超过此值触发重训
        "key_inflection_points": True,   # 保留关键拐点
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._windows: List[RollingWindow] = []
        self._load_state()

    @property
    def current_window(self) -> Optional[RollingWindow]:
        """当前活跃窗口"""
        return self._windows[-1] if self._windows else None

    def get_training_window(self, df_btc: pd.DataFrame,
                            force_full: bool = False) -> Tuple[pd.DataFrame, RollingWindow]:
        """
        根据配置决定本次训练使用的数据窗口

        Args:
            df_btc: 完整 OHLCV 数据
            force_full: 强制使用全部数据

        Returns:
            (训练用DataFrame, 窗口元信息)
        """
        n_total = len(df_btc)
        mode = WindowMode(self.config["mode"])

        if force_full or mode == WindowMode.EXPANDING:
            # 全部数据
            train_df = df_btc.copy()
            window = RollingWindow(
                window_id=self._make_hash(df_btc),
                start_date=str(df_btc.iloc[0].get("date", df_btc.index[0])),
                end_date=str(df_btc.iloc[-1].get("date", df_btc.index[-1])),
                n_samples=len(train_df),
                n_features=-1,
                mode=WindowMode.EXPANDING,
                created_at=datetime.now().isoformat(),
            )

        elif mode == WindowMode.SLIDING:
            size = self.config["sliding_size_days"]
            # 取最近 N 天
            if "date" in df_btc.columns:
                cutoff = df_btc["date"].max() - pd.Timedelta(days=size)
                train_df = df_btc[df_btc["date"] >= cutoff].copy()
            else:
                train_df = df_btc.iloc[-min(size, n_total):].copy()

            window = RollingWindow(
                window_id=self._make_hash(train_df),
                start_date=str(train_df.iloc[0].get("date", train_df.index[0])),
                end_date=str(train_df.iloc[-1].get("date", train_df.index[-1])),
                n_samples=len(train_df),
                n_features=-1,
                mode=WindowMode.SLIDING,
                created_at=datetime.now().isoformat(),
            )

        else:  # HYBRID
            # 2年滑动 + 保留关键拐点数据
            size = self.config["sliding_size_days"]
            if "date" in df_btc.columns:
                cutoff = df_btc["date"].max() - pd.Timedelta(days=size)
                train_df = df_btc[df_btc["date"] >= cutoff].copy()

                # 检测并保留关键拐点
                if self.config["key_inflection_points"]:
                    inflection_dates = self._detect_inflections(df_btc)
                    for d in inflection_dates:
                        if d < cutoff:
                            extra = df_btc[df_btc["date"] >= d].iloc[:30]  # 拐点前后30天
                            train_df = pd.concat([extra, train_df]).drop_duplicates(
                                subset=["date"] if "date" in train_df.columns else None
                            ).sort_values("date" if "date" in train_df.columns else train_df.index.name)
            else:
                train_df = df_btc.iloc[-min(size, n_total):].copy()

            window = RollingWindow(
                window_id=self._make_hash(train_df),
                start_date=str(train_df.iloc[0].get("date", train_df.index[0])),
                end_date=str(train_df.iloc[-1].get("date", train_df.index[-1])),
                n_samples=len(train_df),
                n_features=-1,
                mode=WindowMode.HYBRID,
                created_at=datetime.now().isoformat(),
            )

        self._windows.append(window)
        self._save_state()
        return train_df, window

    def should_retrain(self, snapshots: List[RollingModelSnapshot]) -> Tuple[bool, str]:
        """
        判断是否需要重新训练

        Args:
            snapshots: 当前所有模型快照

        Returns:
            (是否需要重训, 原因)
        """
        if not snapshots:
            return True, "首次训练 (无现有模型)"

        # 检查过期
        max_age = max(s.age_days for s in snapshots)
        if max_age > self.config["max_staleness_days"]:
            return True, f"模型过期 (最大年龄 {max_age}天 > {self.config['max_staleness_days']}天)"

        # 检查ICIR退化
        for s in snapshots:
            if s.icir_trend == "↓" and abs(s.icir_delta) > self.config["retrain_threshold_icir_drop"]:
                return True, f"{s.model_name}/{s.theme_id} ICIR退化 {s.icir_delta:.3f}"

        # 检查特征漂移
        high_drift = [s for s in snapshots if s.feature_drift > 0.3]
        if high_drift:
            return True, f"特征漂移过大 ({len(high_drift)} 个模型 drift > 0.3)"

        return False, f"模型健康 (最新 {max_age}天前)"

    def _detect_inflections(self, df: pd.DataFrame) -> List:
        """检测关键市场拐点日期"""
        close = df["close"].values
        inflection_dates = []

        # 简单拐点检测: 价格从高点回落 >15%
        for i in range(60, len(close)):
            window_high = np.max(close[i-60:i])
            if close[i] < window_high * 0.85:  # 下跌 15%
                if "date" in df.columns:
                    inflection_dates.append(df.iloc[i]["date"])

        # 去重相邻拐点
        if inflection_dates:
            deduped = [inflection_dates[0]]
            for d in inflection_dates[1:]:
                if isinstance(d, pd.Timestamp) and isinstance(deduped[-1], pd.Timestamp):
                    if (d - deduped[-1]).days > 60:
                        deduped.append(d)
                else:
                    deduped.append(d)
            return deduped[:10]  # 最多保留10个拐点
        return []

    def _make_hash(self, df: pd.DataFrame) -> str:
        """生成数据窗口 hash"""
        sample = df["close"].values[:100] if len(df) > 100 else df["close"].values
        return hashlib.md5(sample.tobytes()).hexdigest()[:12]

    def _save_state(self):
        """保存窗口状态到磁盘"""
        state = {
            "config": self.config,
            "windows": [
                {
                    "window_id": w.window_id,
                    "start_date": w.start_date,
                    "end_date": w.end_date,
                    "n_samples": w.n_samples,
                    "n_features": w.n_features,
                    "mode": w.mode.value,
                    "created_at": w.created_at,
                }
                for w in self._windows[-20:]  # 保留最近20个窗口
            ],
        }
        with open(ROLLING_DIR / "window_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """从磁盘加载窗口状态"""
        state_path = ROLLING_DIR / "window_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.config = {**self.config, **state.get("config", {})}
                self._windows = [
                    RollingWindow(
                        window_id=w["window_id"],
                        start_date=w["start_date"],
                        end_date=w["end_date"],
                        n_samples=w["n_samples"],
                        n_features=w.get("n_features", -1),
                        mode=WindowMode(w["mode"]),
                        created_at=w["created_at"],
                    )
                    for w in state.get("windows", [])
                ]
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# Model Version Registry (滚动版)
# ═══════════════════════════════════════════════════════════════

class RollingModelRegistry:
    """
    滚动模型版本注册表 — 管理所有历史快照

    数据存储:
      data/rolling/
      ├── window_state.json           # 窗口管理状态
      ├── registry.json               # 模型快照索引
      ├── snapshots/                  # 每个快照的元信息
      │   ├── alstm_trend_v3.json
      │   └── transformer_reversal_v2.json
      └── models/                     # 实际模型文件 (链接到 data/models/)
    """

    def __init__(self):
        self.snapshots: List[RollingModelSnapshot] = []
        self.registry_path = ROLLING_DIR / "registry.json"
        self._load()

    def register(self, snapshot: RollingModelSnapshot):
        """注册新模型快照"""
        # 检查是否已存在 (相同 model+theme+version)
        existing = [
            i for i, s in enumerate(self.snapshots)
            if s.model_name == snapshot.model_name
            and s.theme_id == snapshot.theme_id
            and s.version == snapshot.version
        ]
        if existing:
            self.snapshots[existing[0]] = snapshot
        else:
            self.snapshots.append(snapshot)

        # 保存快照元信息
        snap_dir = ROLLING_DIR / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{snapshot.model_name}_{snapshot.theme_id}_v{snapshot.version}.json"
        with open(snap_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        self._save()

    def get_latest(self, model_name: str, theme_id: str) -> Optional[RollingModelSnapshot]:
        """获取最新版本"""
        matching = [
            s for s in self.snapshots
            if s.model_name == model_name and s.theme_id == theme_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda s: s.version)

    def get_history(self, model_name: str, theme_id: str) -> List[RollingModelSnapshot]:
        """获取某模型某主题的完整版本历史"""
        return sorted(
            [s for s in self.snapshots
             if s.model_name == model_name and s.theme_id == theme_id],
            key=lambda s: s.version,
        )

    def get_all_latest(self) -> List[RollingModelSnapshot]:
        """获取所有模型的最新版本"""
        keys = set((s.model_name, s.theme_id) for s in self.snapshots)
        return [
            self.get_latest(model, theme)
            for model, theme in keys
        ]

    def get_staleness_summary(self) -> List[dict]:
        """获取模型新鲜度摘要"""
        latest = self.get_all_latest()
        summary = []
        for s in sorted(latest, key=lambda x: -x.age_days):
            summary.append({
                "model": s.model_name,
                "theme": s.theme_name,
                "version": s.version,
                "age_days": s.age_days,
                "staleness": s.staleness.value,
                "icir": s.oos_icir,
                "trend": s.icir_trend,
                "drift": s.feature_drift,
            })
        return summary

    def next_version(self, model_name: str, theme_id: str) -> int:
        """获取下一个版本号"""
        latest = self.get_latest(model_name, theme_id)
        return (latest.version + 1) if latest else 1

    def _save(self):
        """保存注册表到磁盘"""
        with open(self.registry_path, "w") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "n_snapshots": len(self.snapshots),
                "snapshots": [s.to_dict() for s in self.snapshots],
            }, f, indent=2)

    def _load(self):
        """从磁盘加载注册表"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                self.snapshots = []
                for s in data.get("snapshots", []):
                    try:
                        self.snapshots.append(RollingModelSnapshot(
                            model_name=s["model_name"],
                            theme_id=s["theme_id"],
                            theme_name=s["theme_name"],
                            window_id=s.get("window_id", ""),
                            version=s["version"],
                            trained_at=s["trained_at"],
                            data_start=s["data_start"],
                            data_end=s["data_end"],
                            n_samples=s["n_samples"],
                            n_epochs=s.get("n_epochs", 100),
                            seq_len=s.get("seq_len", 60),
                            fwd_window=s.get("fwd_window", 5),
                            oos_ic=s["oos_ic"],
                            oos_icir=s["oos_icir"],
                            oos_r2=s["oos_r2"],
                            oos_hit_rate=s["oos_hit_rate"],
                            model_path=s["model_path"],
                            meta_path=s.get("meta_path", ""),
                            icir_delta=s.get("icir_delta", 0.0),
                            icir_trend=s.get("icir_trend", "→"),
                            data_hash=s.get("data_hash", ""),
                            feature_drift=s.get("feature_drift", 0.0),
                        ))
                    except Exception:
                        continue
            except Exception:
                self.snapshots = []


# ═══════════════════════════════════════════════════════════════
# Feature Drift Detector
# ═══════════════════════════════════════════════════════════════

class FeatureDriftDetector:
    """
    特征漂移检测器 — 检测市场数据分布变化

    当特征分布发生显著变化时, 旧模型可能不再适用,
    触发重训预警。

    检测方法:
      1. 每列特征计算 KS 统计量 (Kolmogorov-Smirnov)
      2. 平均漂移 score = mean(KS over all features)
      3. drift > 0.3 → 建议重训
    """

    @staticmethod
    def compute_drift(old_features: np.ndarray,
                      new_features: np.ndarray) -> float:
        """
        计算特征分布漂移

        Args:
            old_features: (n_old, n_features) 训练时的特征
            new_features: (n_new, n_features) 当前的特征

        Returns:
            drift_score: 0-1, 越高表示漂移越大
        """
        if old_features.shape[1] != new_features.shape[1]:
            return 1.0  # 特征维度不同 → 完全漂移

        n_feats = min(old_features.shape[1], 100)  # 最多检查100个特征
        ks_scores = []

        for i in range(n_feats):
            old_col = old_features[:, i]
            new_col = new_features[:, i]

            # 移除 NaN
            old_col = old_col[~np.isnan(old_col)]
            new_col = new_col[~np.isnan(new_col)]

            if len(old_col) < 10 or len(new_col) < 10:
                continue

            try:
                from scipy import stats
                ks_stat, _ = stats.ks_2samp(old_col, new_col)
                ks_scores.append(ks_stat)
            except Exception:
                # Fallback: 比较均值和标准差
                old_mean, old_std = np.mean(old_col), np.std(old_col)
                new_mean, new_std = np.mean(new_col), np.std(new_col)
                mean_diff = abs(old_mean - new_mean) / max(abs(old_mean), 1e-8)
                std_diff = abs(old_std - new_std) / max(abs(old_std), 1e-8)
                ks_scores.append(min(1.0, (mean_diff + std_diff) / 2))

        if not ks_scores:
            return 0.0

        return float(np.mean(ks_scores))

    @staticmethod
    def compute_data_hash(features: np.ndarray) -> str:
        """计算特征矩阵指纹"""
        # 采样 + hash
        if len(features) > 500:
            idx = np.linspace(0, len(features)-1, 500, dtype=int)
            sample = features[idx]
        else:
            sample = features
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# Rolling Trainer — 主引擎
# ═══════════════════════════════════════════════════════════════

class RollingTrainer:
    """
    滚动训练主引擎

    核心循环:
      1. 拉取最新数据
      2. 检测是否需要重训 (模型年龄 / ICIR退化 / 特征漂移)
      3. 如果触发 → 滚动窗口训练
      4. 注册新版本 → 更新注册表
      5. 输出训练报告

    训练范围:
      - LightGBM × 7 主题 (传统基线)
      - ALSTM × 7 主题 (Qlib 时序)
      - Transformer × 7 主题 (Qlib 注意力)
      - TabNet × 7 主题 (Qlib 特征选择)
    """

    DEFAULT_CONFIG = {
        **RollingWindowManager.DEFAULT_CONFIG,
        "qlib_epochs": 50,             # Qlib模型每轮epoch数
        "qlib_epochs_full": 100,       # 全量训练epoch数
        "lgbm_early_stop": 50,         # LightGBM早停轮数
        "fwd_window": 5,               # 前向预测窗口
        "seq_len": 60,                 # 时序序列长度
        "purge_days": 10,              # PurgedKFold purge间隔
        "n_folds": 5,                  # 交叉验证折数
        "models_to_train": ["alstm", "transformer", "tabnet"],  # 默认训练模型
        "themes_to_train": None,       # None = 全部7个主题
        "use_double_ensemble": False,
        "auto_cleanup_old": True,      # 自动清理旧版本
        "keep_versions": 5,            # 每个模型保留版本数
        "graph_check_enabled": True,   # Phase 11: 图漂移检测
        "graph_drift_threshold": 0.3,  # 图漂移重训阈值
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.window_manager = RollingWindowManager(self.config)
        self.registry = RollingModelRegistry()
        self.drift_detector = FeatureDriftDetector()
        self.qlib_trainer = QlibTrainer({
            "seq_len": self.config["seq_len"],
            "fwd_window": self.config["fwd_window"],
            "purge_days": self.config["purge_days"],
            "n_folds": self.config["n_folds"],
            "n_epochs_max": self.config["qlib_epochs"],
        })
        self.feature_factory = FeatureFactoryV4()
        self.theme_config = SubSignalBuilder.THEME_CONFIG

        # 资产关系图 (Phase 11)
        self.graph_builder = None
        if GRAPH_AVAILABLE and self.config.get("graph_check_enabled", True):
            try:
                self.graph_builder = AssetGraphBuilder()
                print("🔗 图漂移监控已启用 (Phase 11)")
            except Exception as e:
                print(f"⚠️ 图引擎初始化失败: {e}")

    # ── 主入口 ──────────────────────────────────

    def run(self, df_btc: pd.DataFrame,
            force: bool = False,
            mode: str = "update") -> dict:
        """
        运行滚动训练主循环

        Args:
            df_btc: 最新 OHLCV 数据
            force: 强制执行全量重训
            mode: "init" 首次 / "update" 增量 / "full" 全量

        Returns:
            运行结果摘要
        """
        print("🔄 Rolling Trainer v2.0 — 滚动在线学习引擎 + 图漂移监控")
        print(f"   模式: {mode} | 强制: {force}")
        print(f"   数据: {len(df_btc)} 天 | "
              f"{df_btc.iloc[0].get('date', '?')} → {df_btc.iloc[-1].get('date', '?')}")

        # 1. 检查是否需要重训
        latest_snapshots = self.registry.get_all_latest()
        should, reason = self.window_manager.should_retrain(latest_snapshots)

        # Phase 11: 检查图漂移
        graph_drift_result = None
        if self.graph_builder and self.config.get("graph_check_enabled", True):
            print("🔗 检查资产关系图漂移...")
            graph_drift_result = self.check_graph_drift()
            if graph_drift_result.get("drifted"):
                reason += f" + 图漂移({graph_drift_result['drift_score']:.3f})"
                should = True

        if not force and not should and mode != "init":
            print(f"\n✅ 无需重训: {reason}")
            return {
                "action": "skip",
                "reason": reason,
                "n_snapshots": len(latest_snapshots),
                "timestamp": datetime.now().isoformat(),
            }

        if force:
            reason = "强制重训"
        elif mode == "init":
            reason = "首次初始化"

        print(f"\n🔧 触发重训: {reason}")

        # 2. 确定训练窗口
        train_df, window = self.window_manager.get_training_window(
            df_btc, force_full=(mode in ("init", "full"))
        )
        print(f"📐 训练窗口: {window.mode.value} | "
              f"{window.n_samples} 样本 | {window.start_date} → {window.end_date}")

        # 3. 计算全时序特征矩阵
        print("\n📊 计算特征矩阵...")
        feature_df = self.feature_factory.compute_timeseries(train_df)
        n_features = feature_df.shape[1]
        window.n_features = n_features
        print(f"   特征: {n_features} 列")

        # 4. 特征漂移检测
        feature_matrix = feature_df.values.astype(np.float32)
        data_hash = self.drift_detector.compute_data_hash(feature_matrix)

        # 与上一窗口对比漂移
        if latest_snapshots and latest_snapshots[0].data_hash:
            # 加载旧特征 (从上次训练的模型元信息)
            feature_drift = 0.0  # 简化: 不做完整对比
        else:
            feature_drift = 0.0

        # 5. 训练 Qlib 深度学习模型
        qlib_results = self._train_qlib_rolling(
            train_df, window, data_hash, feature_drift,
            epochs_override=(
                self.config["qlib_epochs_full"]
                if mode in ("init", "full")
                else self.config["qlib_epochs"]
            ),
        )

        # 6. 训练 LightGBM 模型 (基线)
        lgbm_results = self._train_lgbm_rolling(
            train_df, window, data_hash, feature_drift
        )

        # 7. 清理旧版本
        n_cleaned = 0
        if self.config["auto_cleanup_old"]:
            n_cleaned = self._cleanup_old_versions()

        # 8. 生成报告
        all_results = qlib_results + lgbm_results
        report = self._generate_report(all_results, window, reason)

        print(f"\n{'='*60}")
        print(f"✅ 滚动训练完成!")
        print(f"   Qlib模型: {len(qlib_results)} 个")
        print(f"   LightGBM: {len(lgbm_results)} 个")
        print(f"   窗口: {window.mode.value} | {window.n_samples} 样本")
        print(f"   清理旧版本: {n_cleaned} 个")
        print(f"   报告: {ROLLING_DIR / 'rolling_report_latest.json'}")
        print(f"{'='*60}")

        return {
            "action": "trained",
            "reason": reason,
            "window": {
                "mode": window.mode.value,
                "n_samples": window.n_samples,
                "n_features": n_features,
                "start": window.start_date,
                "end": window.end_date,
            },
            "n_qlib_models": len(qlib_results),
            "n_lgbm_models": len(lgbm_results),
            "n_cleaned": n_cleaned,
            "graph_drift": graph_drift_result,  # Phase 11
            "report": report,
            "timestamp": datetime.now().isoformat(),
        }

    def _train_qlib_rolling(self, df_btc: pd.DataFrame,
                            window: RollingWindow,
                            data_hash: str,
                            feature_drift: float,
                            epochs_override: int = 50) -> List[dict]:
        """训练 Qlib 深度学习模型 (滚动版本)"""
        results = []
        models_to_train = self.config["models_to_train"]
        themes_to_train = self.config["themes_to_train"]

        # 临时修改 trainer config
        original_epochs = self.qlib_trainer.config["n_epochs_max"]
        self.qlib_trainer.config["n_epochs_max"] = epochs_override

        try:
            train_results = self.qlib_trainer.train_all(
                df_btc,
                models=models_to_train,
                themes=themes_to_train,
                use_double_ensemble=self.config["use_double_ensemble"],
            )

            for tr in train_results:
                # 获取版本号
                version = self.registry.next_version(tr.model_name, tr.theme_id)

                # 计算与上一版本的ICIR变化
                prev = self.registry.get_latest(tr.model_name, tr.theme_id)
                icir_delta = tr.oos_icir - prev.oos_icir if prev else 0.0
                if abs(icir_delta) < 0.02:
                    icir_trend = "→"
                elif icir_delta > 0:
                    icir_trend = "↑"
                else:
                    icir_trend = "↓"

                snapshot = RollingModelSnapshot(
                    model_name=f"qlib_{tr.model_name}",
                    theme_id=tr.theme_id,
                    theme_name=tr.theme_name,
                    window_id=window.window_id,
                    version=version,
                    trained_at=tr.trained_at,
                    data_start=window.start_date,
                    data_end=window.end_date,
                    n_samples=tr.n_samples,
                    n_epochs=tr.n_epochs_trained,
                    seq_len=tr.seq_len,
                    fwd_window=tr.fwd_window,
                    oos_ic=tr.oos_ic,
                    oos_icir=tr.oos_icir,
                    oos_r2=tr.oos_r2,
                    oos_hit_rate=tr.oos_hit_rate,
                    model_path=tr.model_path,
                    meta_path=str(Path(tr.model_path).with_suffix(".json")),
                    icir_delta=icir_delta,
                    icir_trend=icir_trend,
                    data_hash=data_hash,
                    feature_drift=feature_drift,
                )

                self.registry.register(snapshot)
                results.append(snapshot.to_dict())

        finally:
            self.qlib_trainer.config["n_epochs_max"] = original_epochs

        return results

    def _train_lgbm_rolling(self, df_btc: pd.DataFrame,
                            window: RollingWindow,
                            data_hash: str,
                            feature_drift: float) -> List[dict]:
        """
        训练 LightGBM 模型 (滚动版本)

        复用 ml_lightgbm_trainer 的训练逻辑,
        但使用滚动窗口数据 + 版本化管理
        """
        results = []

        try:
            from ml_lightgbm_trainer import LightGBMTrainer, ModelRegistry as LGBMRegistry

            lgbm_trainer = LightGBMTrainer()
            lgbm_registry = LGBMRegistry()

            train_results = lgbm_trainer.train_all(df_btc)

            for tr in train_results:
                version = self.registry.next_version("lgbm", tr.theme_id)
                prev = self.registry.get_latest("lgbm", tr.theme_id)

                icir_delta = tr.oos_icir - prev.oos_icir if prev else 0.0
                if abs(icir_delta) < 0.02:
                    icir_trend = "→"
                elif icir_delta > 0:
                    icir_trend = "↑"
                else:
                    icir_trend = "↓"

                # 找到对应的模型文件
                model_path = MODEL_DIR / f"lgbm_{tr.theme_id}.pkl"

                snapshot = RollingModelSnapshot(
                    model_name="lgbm",
                    theme_id=tr.theme_id,
                    theme_name=tr.theme_name,
                    window_id=window.window_id,
                    version=version,
                    trained_at=tr.trained_at,
                    data_start=window.start_date,
                    data_end=window.end_date,
                    n_samples=tr.n_samples,
                    n_epochs=0,  # LightGBM 不是 epoch 训练
                    seq_len=0,
                    fwd_window=tr.fwd_window,
                    oos_ic=tr.oos_ic,
                    oos_icir=tr.oos_icir,
                    oos_r2=tr.oos_r2,
                    oos_hit_rate=tr.oos_hit_rate,
                    model_path=str(model_path),
                    meta_path=str(MODEL_DIR / f"lgbm_{tr.theme_id}_meta.json"),
                    icir_delta=icir_delta,
                    icir_trend=icir_trend,
                    data_hash=data_hash,
                    feature_drift=feature_drift,
                )

                self.registry.register(snapshot)
                results.append(snapshot.to_dict())

        except ImportError as e:
            print(f"  ⚠️ LightGBM 滚动训练跳过: {e}")
        except Exception as e:
            print(f"  ⚠️ LightGBM 滚动训练出错: {e}")

        return results

    def _cleanup_old_versions(self) -> int:
        """清理旧版本, 每个模型只保留最近 N 个版本"""
        keep_n = self.config["keep_versions"]
        all_snapshots = self.registry.snapshots
        keys = set((s.model_name, s.theme_id) for s in all_snapshots)

        removed = 0
        for model_name, theme_id in keys:
            history = self.registry.get_history(model_name, theme_id)
            if len(history) <= keep_n:
                continue

            # 保留ICIR最好的 keep_n 个
            keep = sorted(history, key=lambda s: s.oos_icir, reverse=True)[:keep_n]
            keep_paths = {s.model_path for s in keep}

            for s in history:
                if s.model_path not in keep_paths:
                    # 删除旧模型文件 (但保留 data/models/ 中的最新版)
                    # 只清理 rolling snapshots 目录中的元信息
                    snap_path = ROLLING_DIR / "snapshots" / f"{s.model_name}_{s.theme_id}_v{s.version}.json"
                    if snap_path.exists():
                        snap_path.unlink()
                        removed += 1

        # 重新加载注册表
        self.registry._load()
        return removed

    def _generate_report(self, results: List[dict],
                         window: RollingWindow,
                         reason: str) -> dict:
        """生成滚动训练报告"""
        # 按ICIR排序
        sorted_results = sorted(results, key=lambda r: r["oos_icir"], reverse=True)

        # 计算统计
        icirs = [r["oos_icir"] for r in results]
        hit_rates = [r["oos_hit_rate"] for r in results]

        report = {
            "generated_at": datetime.now().isoformat(),
            "trigger_reason": reason,
            "window": {
                "mode": window.mode.value,
                "start": window.start_date,
                "end": window.end_date,
                "n_samples": window.n_samples,
            },
            "summary": {
                "n_models_trained": len(results),
                "n_improved": sum(1 for r in results if r["icir_trend"] == "↑"),
                "n_stable": sum(1 for r in results if r["icir_trend"] == "→"),
                "n_degraded": sum(1 for r in results if r["icir_trend"] == "↓"),
                "mean_icir": round(np.mean(icirs), 4) if icirs else 0,
                "median_icir": round(np.median(icirs), 4) if icirs else 0,
                "max_icir": round(np.max(icirs), 4) if icirs else 0,
                "mean_hit_rate": round(np.mean(hit_rates), 4) if hit_rates else 0,
            },
            "top_performers": sorted_results[:5],
            "all_models": sorted_results,
        }

        # 保存报告
        report_path = ROLLING_DIR / "rolling_report_latest.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # 同时保存带时间戳的历史报告
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        hist_path = ROLLING_DIR / f"rolling_report_{ts}.json"
        with open(hist_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def status(self) -> dict:
        """获取滚动训练状态"""
        snapshots = self.registry.get_all_latest()
        staleness = self.registry.get_staleness_summary()

        should_retrain, reason = self.window_manager.should_retrain(snapshots)

        # Phase 11: 图状态
        graph_status = self._get_graph_status()

        return {
            "n_snapshots": len(snapshots),
            "n_models": len(set(s.model_name for s in snapshots)),
            "n_themes": len(set(s.theme_id for s in snapshots)),
            "latest_training": max((s.trained_at for s in snapshots), default="never"),
            "max_age_days": max((s.age_days for s in snapshots), default=0),
            "staleness_summary": staleness,
            "should_retrain": should_retrain,
            "retrain_reason": reason,
            "window_config": self.window_manager.config,
            "registry_path": str(self.registry.registry_path),
            "graph_status": graph_status,  # Phase 11
        }

    def _get_graph_status(self) -> dict:
        """获取资产关系图状态 (Phase 11)"""
        if not self.graph_builder:
            return {"available": False, "reason": "图引擎未初始化"}

        try:
            snapshot = self.graph_builder.load_snapshot("latest")
            if snapshot is None:
                return {"available": False, "reason": "图快照未构建"}

            # 图年龄
            if snapshot.created_at:
                created = datetime.fromisoformat(snapshot.created_at)
                age_days = (datetime.now() - created).days
            else:
                age_days = 999

            return {
                "available": True,
                "n_assets": snapshot.n_assets,
                "symbols": snapshot.symbols,
                "density": snapshot.graph_density,
                "avg_degree": snapshot.avg_degree,
                "n_communities": snapshot.n_communities,
                "age_days": age_days,
                "top_edge": snapshot.top_edges[0] if snapshot.top_edges else None,
                "stale": age_days > 7,  # 图超过7天认为过时
            }
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def check_graph_drift(self, symbols: List[str] = None) -> dict:
        """
        检测资产关系图漂移 — 市场结构是否变了 (Phase 11)

        Returns:
            {"drifted": bool, "drift_score": float, "action": str}
        """
        if not self.graph_builder:
            return {"drifted": False, "drift_score": 0.0, "action": "graph_unavailable"}

        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

        try:
            # 加载旧图
            old_snapshot = self.graph_builder.load_snapshot("latest")

            # 构建新图
            new_snapshot = self.graph_builder.build(symbols, lookback_days=90)

            if old_snapshot is None:
                # 第一次构建
                self.graph_builder.save_snapshot(new_snapshot, "latest")
                return {"drifted": False, "drift_score": 0.0, "action": "first_build"}

            # 计算漂移
            # 需要对齐资产
            common_symbols = list(set(old_snapshot.symbols) & set(new_snapshot.symbols))
            if len(common_symbols) < 3:
                return {"drifted": False, "drift_score": 0.0, "action": "insufficient_overlap"}

            old_idx = [old_snapshot.symbols.index(s) for s in common_symbols]
            new_idx = [new_snapshot.symbols.index(s) for s in common_symbols]

            old_adj = old_snapshot.adj_matrix[np.ix_(old_idx, old_idx)]
            new_adj = new_snapshot.adj_matrix[np.ix_(new_idx, new_idx)]

            drift_score = self.graph_builder.detect_graph_drift(old_adj, new_adj)
            threshold = self.config.get("graph_drift_threshold", 0.3)
            drifted = drift_score > threshold

            if drifted:
                # 保存新图
                self.graph_builder.save_snapshot(new_snapshot, "latest")
                print(f"🔴 图漂移检测: {drift_score:.4f} > {threshold} → 图已更新!")
            else:
                print(f"✅ 图结构稳定: {drift_score:.4f} ≤ {threshold}")

            return {
                "drifted": drifted,
                "drift_score": drift_score,
                "threshold": threshold,
                "action": "graph_rebuilt" if drifted else "graph_stable",
                "n_common_assets": len(common_symbols),
                "old_density": old_snapshot.graph_density,
                "new_density": new_snapshot.graph_density,
            }
        except Exception as e:
            return {"drifted": False, "drift_score": 0.0, "action": f"error: {e}"}

    def backtest_windows(self, df_btc: pd.DataFrame,
                         n_windows: int = 5) -> pd.DataFrame:
        """
        回测滚动窗口性能 — 模拟历史重训效果

        对历史数据切分成 N 个窗口, 每个窗口独立训练,
        记录OOS表现, 用于评估滚动训练的价值。

        Returns:
            DataFrame with columns: [window, start, end, model, theme, icir, hit_rate]
        """
        print(f"📊 滚动窗口回测 (n={n_windows})...")
        n_total = len(df_btc)
        window_size = n_total // (n_windows + 1)
        records = []

        for i in range(n_windows):
            # 窗口: [0, (i+1)*window_size]
            train_end = (i + 1) * window_size
            train_df = df_btc.iloc[:train_end].copy()

            print(f"\n  📐 窗口 {i+1}/{n_windows}: {len(train_df)} 样本")

            # 轻量训练 (少epoch)
            original_epochs = self.qlib_trainer.config["n_epochs_max"]
            self.qlib_trainer.config["n_epochs_max"] = 30  # 快速回测

            try:
                train_results = self.qlib_trainer.train_all(
                    train_df,
                    models=["alstm", "transformer"],
                    themes=["trend", "reversal", "volume"],
                )

                for tr in train_results:
                    records.append({
                        "window": i + 1,
                        "n_samples": tr.n_samples,
                        "model": tr.model_name,
                        "theme": tr.theme_name,
                        "icir": tr.oos_icir,
                        "hit_rate": tr.oos_hit_rate,
                        "r2": tr.oos_r2,
                    })
            except Exception as e:
                print(f"    ⚠️ 窗口 {i+1} 训练失败: {e}")
            finally:
                self.qlib_trainer.config["n_epochs_max"] = original_epochs

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            # 保存回测结果
            result_df.to_csv(ROLLING_DIR / "rolling_backtest.csv", index=False)

            # 打印摘要
            print(f"\n📈 滚动窗口回测摘要:")
            pivot = result_df.pivot_table(
                values="icir", index="model", columns="window", aggfunc="mean"
            )
            print(pivot.to_string())

        return result_df


# ═══════════════════════════════════════════════════════════════
# Cron 集成 — 自动定时重训
# ═══════════════════════════════════════════════════════════════

def auto_rolling_check(df_btc: pd.DataFrame = None) -> dict:
    """
    自动滚动检查 — 专为 Cron 触发设计

    用法 (crontab):
      0 8 * * 1 cd ~/yina-app/chase-quant-web && python3 -c "
      from rolling_trainer import auto_rolling_check
      import ccxt, pandas as pd
      ex = ccxt.binance()
      ohlcv = ex.fetch_ohlcv('BTC/USDT', '1d', limit=800)
      df = pd.DataFrame(ohlcv, columns=['date','open','high','low','close','volume'])
      print(auto_rolling_check(df))
      "

    Returns:
        检查结果字典
    """
    if df_btc is None:
        print("⚠️ 未提供数据, 跳过滚动训练")
        return {"action": "no_data"}

    trainer = RollingTrainer()
    status = trainer.status()

    # Phase 11: 同时检查图漂移
    graph_status = status.get("graph_status", {})
    graph_drift = None
    if graph_status.get("available") and graph_status.get("stale"):
        print("🔗 资产关系图过期, 检测漂移...")
        graph_drift = trainer.check_graph_drift()
        if graph_drift.get("drifted"):
            print(f"🔴 图漂移: {graph_drift['drift_score']:.4f}")
            status["should_retrain"] = True
            status["retrain_reason"] = (status.get("retrain_reason", "") +
                                       f" + 图漂移({graph_drift['drift_score']:.3f})")

    if status["should_retrain"]:
        print(f"🔄 触发自动重训: {status['retrain_reason']}")
        result = trainer.run(df_btc, force=False, mode="update")
        result["pre_check"] = status
        result["graph_drift"] = graph_drift
        return result
    else:
        print(f"✅ 模型健康, 跳过: {status['retrain_reason']}")
        return {
            "action": "skip",
            "reason": status["retrain_reason"],
            "max_age_days": status["max_age_days"],
            "graph_status": graph_status,
            "graph_drift": graph_drift,
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🔄 Rolling Trainer v2.0 — 滚动在线学习引擎 + 图漂移监控"
    )
    parser.add_argument("--init", action="store_true",
                        help="首次全量训练")
    parser.add_argument("--update", action="store_true",
                        help="增量更新 (检查是否需要重训)")
    parser.add_argument("--force", action="store_true",
                        help="强制全量重训")
    parser.add_argument("--status", action="store_true",
                        help="查看滚动训练状态")
    parser.add_argument("--backtest", action="store_true",
                        help="回测滚动窗口性能")
    parser.add_argument("--check-graph", action="store_true",
                        help="检查资产关系图漂移 (Phase 11)")
    parser.add_argument("--build-graph", action="store_true",
                        help="构建/重建资产关系图 (Phase 11)")
    parser.add_argument("--n-windows", type=int, default=5,
                        help="回测窗口数 (default: 5)")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["expanding", "sliding", "hybrid"],
                        help="窗口模式 (default: hybrid)")
    parser.add_argument("--models", type=str, default="alstm,transformer,tabnet",
                        help="训练模型列表 (逗号分隔)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="每轮训练epoch数")

    args = parser.parse_args()

    # 初始化训练器
    config = {
        "mode": args.mode,
        "models_to_train": [m.strip() for m in args.models.split(",")],
        "qlib_epochs": args.epochs,
    }
    trainer = RollingTrainer(config)

    # 获取数据
    df_btc = None
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=800)
        df_btc = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        print(f"📡 获取 BTC/USDT 数据: {len(df_btc)} 天")
    except Exception as e:
        print(f"⚠️ 无法获取实时数据: {e}")
        print("  使用本地缓存数据...")
        # 尝试从本地加载
        cache_path = DATA_DIR / "btc_cache.csv"
        if cache_path.exists():
            df_btc = pd.read_csv(cache_path)
            print(f"  ✅ 加载缓存: {len(df_btc)} 天")

    if df_btc is None or len(df_btc) < 100:
        print("❌ 数据不足, 至少需要100天数据")
        exit(1)

    # 执行操作
    if args.status:
        status = trainer.status()
        print("\n📊 滚动训练状态:")
        print(f"   模型快照: {status['n_snapshots']} 个")
        print(f"   覆盖模型: {status['n_models']} 种")
        print(f"   覆盖主题: {status['n_themes']} 个")
        print(f"   最近训练: {status['latest_training'][:19] if status['latest_training'] != 'never' else '从未'}")
        print(f"   最大年龄: {status['max_age_days']} 天")
        print(f"   需要重训: {'是' if status['should_retrain'] else '否'} — {status['retrain_reason']}")

        # Phase 11: 图状态
        gs = status.get("graph_status", {})
        if gs.get("available"):
            print(f"\n🔗 资产关系图 (Phase 11):")
            print(f"   资产数: {gs['n_assets']}")
            print(f"   图密度: {gs['graph_density']:.4f}")
            print(f"   平均度: {gs['avg_degree']:.2f}")
            print(f"   社区数: {gs['n_communities']}")
            print(f"   图年龄: {gs['age_days']} 天 {'⚠️ 需更新' if gs.get('stale') else '✅ 新鲜'}")
            if gs.get('top_edge'):
                e = gs['top_edge']
                print(f"   最强边: {e['source']} ←→ {e['target']} (w={e['weight']:.3f})")
        else:
            print(f"\n🔗 资产关系图: ❌ {gs.get('reason', '未初始化')}")

        print(f"\n📋 模型新鲜度:")
        for s in status["staleness_summary"]:
            print(f"   {s['staleness']:16s} | {s['model']:15s} | {s['theme']:10s} | "
                  f"v{s['version']} | ICIR={s['icir']:.3f} | {s['trend']} | drift={s['drift']:.2f}")

    elif args.backtest:
        result_df = trainer.backtest_windows(df_btc, n_windows=args.n_windows)
        if not result_df.empty:
            print(f"\n✅ 回测完成: {len(result_df)} 条记录")
            print(f"   保存到: {ROLLING_DIR / 'rolling_backtest.csv'}")

    elif args.init:
        print("🚀 首次全量训练...")
        result = trainer.run(df_btc, force=True, mode="init")
        print(f"\n结果: {result['action']}")
        if result.get("report"):
            r = result["report"]["summary"]
            print(f"   训练模型: {r['n_models_trained']} 个")
            print(f"   改善/稳定/退化: {r['n_improved']}/{r['n_stable']}/{r['n_degraded']}")
            print(f"   平均ICIR: {r['mean_icir']}")

    elif args.check_graph:
        print("🔗 检查资产关系图漂移...")
        result = trainer.check_graph_drift()
        print(f"\n📊 图漂移检测结果:")
        print(f"   漂移分数: {result.get('drift_score', 0):.4f}")
        print(f"   阈值: {result.get('threshold', 0.3)}")
        print(f"   是否漂移: {'🔴 是' if result.get('drifted') else '✅ 否'}")
        print(f"   动作: {result.get('action', 'unknown')}")
        if result.get('n_common_assets'):
            print(f"   对齐资产: {result['n_common_assets']} 个")
            print(f"   旧密度: {result.get('old_density', 0):.4f}")
            print(f"   新密度: {result.get('new_density', 0):.4f}")

    elif args.build_graph:
        print("🔨 构建资产关系图...")
        try:
            builder = AssetGraphBuilder()
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
            snapshot = builder.build(symbols)
            builder.save_snapshot(snapshot)
            print(f"\n✅ 图构建完成!")
            print(f"   资产: {snapshot.n_assets}")
            print(f"   密度: {snapshot.graph_density:.4f}")
            print(f"   社区: {snapshot.n_communities}")
            print(f"\n🔗 Top 5 最强连边:")
            for e in snapshot.top_edges[:5]:
                print(f"   {e['source']:12s} ←→ {e['target']:12s}  w={e['weight']:.3f}")
        except Exception as e:
            print(f"❌ 图构建失败: {e}")

    elif args.update:
        print("🔍 检查并增量更新...")
        result = auto_rolling_check(df_btc)

    elif args.force:
        print("💪 强制全量重训...")
        result = trainer.run(df_btc, force=True, mode="full")

    else:
        # 默认: 检查和增量更新
        print("🔍 默认模式: 检查并增量更新...")
        result = auto_rolling_check(df_btc)
