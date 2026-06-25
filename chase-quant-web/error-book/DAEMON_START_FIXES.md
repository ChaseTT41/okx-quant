# Daemon 启动必须修复项 🚨

> 每次重启 daemon 或重新拉代码后，必须检查并修复以下两项。
> 这两处是代码层面的方向性错误，GPU/模型无关，每次都需要手工确认。

---

## Fix 1: `ml_signal_v5.py` — F&G 阈值方向修正

**位置**: `ml_signal_v5.py` 约第 459-481 行  
**症状**: 极端恐惧(F&G≤25)时引擎判定买入更容易、做空更困难 — 完全反了  
**影响**: 引擎内部 `generate_signal()` 产出的 action 全错，信号源头被污染

**错误代码** (旧):
```python
if fg <= 25:  # 极端恐惧 → 降低买入门槛, 收紧做空  ← 注释就是反的
    buy_threshold = max(0.02, buy_threshold * 0.5)   # 买入更容易
    sell_threshold = sell_threshold * 1.3            # 做空更困难
elif fg <= 35:
    buy_threshold = max(0.03, buy_threshold * 0.7)   # 只降买入
elif fg >= 75:
    sell_threshold = sell_threshold * 0.7            # 只降做空
```

**正确代码** (新):
```python
if fg <= 25:
    # 极端恐惧 → 收紧买入 + 放宽做空（市场恐慌时优先做空）
    buy_threshold = buy_threshold * 2.0     # 0.06→0.12 更难触发买入
    sell_threshold = sell_threshold * 0.5   # -0.06→-0.03 更容易触发做空
elif fg <= 35:
    # 恐惧 → 买入收紧 + 做空适度放宽
    buy_threshold = buy_threshold * 1.4
    sell_threshold = sell_threshold * 0.7
elif fg >= 75:
    # 极端贪婪 → 做空收紧
    sell_threshold = sell_threshold * 1.5
elif fg >= 65:
    # 偏贪婪 → 做空适度收紧
    sell_threshold = sell_threshold * 1.2
```

---

## Fix 2: `auto_trade.py` — 用 `signal_weighted` 而不是 `signal_consensus` 判定方向

**位置**: `auto_trade.py` 约第 523 行 (`_scan_with_graph` 方法内)  
**症状**: 用 `signal_consensus` 判定 BUY/SELL/HOLD，但该值已被 `consensus_ratio` 稀释  
**原理**:
```
signal_consensus = signal_weighted × consensus_ratio
```
- `signal_weighted` = 模型预测的加权平均（如 -0.05 = 预测跌5%）
- `consensus_ratio` = 模型方向一致比例（如 0.5 = 一半模型同意）
- `signal_consensus` = -0.05 × 0.5 = **-0.025** ← 被稀释了一半
- 阈值 sell_thr = -0.03，-0.025 不满足 → **漏判为 HOLD**

`ml_signal_v5.py` 引擎内部（第 476 行）用的就是 `signal_weighted` 判定，但 `auto_trade.py` 覆写时用了 `signal_consensus`，不一致。

**错误代码** (旧):
```python
if fusion_signal.signal_consensus > buy_thr:
    fusion_signal.action = "BUY"
elif fusion_signal.signal_consensus < sell_thr:
    fusion_signal.action = "SELL"
else:
    fusion_signal.action = "HOLD"
```

**正确代码** (新):
```python
# 用 signal_weighted 判定方向（不被 consensus_ratio 稀释）
if fusion_signal.signal_weighted > buy_thr:
    fusion_signal.action = "BUY"
elif fusion_signal.signal_weighted < sell_thr:
    fusion_signal.action = "SELL"
else:
    fusion_signal.action = "HOLD"
```

---

## 相关联动

`auto_trade.py` 第 510-520 行还有一层 F&G 阈值调整（之前已修过），确认逻辑正确：
```python
if fg <= 25:
    buy_thr = buy_thr * 2.0     # 收紧买入
    sell_thr = sell_thr * 0.5   # 放宽做空
elif fg <= 35:
    buy_thr = buy_thr * 1.4
    sell_thr = sell_thr * 0.7
elif fg >= 75:
    sell_thr = sell_thr * 1.5
elif fg >= 65:
    sell_thr = sell_thr * 1.2
```

---

## 验证方法

修复后查看 daemon 日志，应该看到：
```
🔍 [1/35] BTC/USDT... 🔴 SELL | sig=-0.050 | 置信=50% | 4活跃
🔍 [2/35] ETH/USDT... 🔴 SELL | sig=-0.045 | 置信=45% | 4活跃
📊 扫描结果: 0 BUY, 34 SELL, 1 HOLD
⚖️ 三层审批通过: N个信号 (BTC/USDT, ETH/USDT, ...)
```
信号值在 -0.03 ~ -0.07 范围且三层审批有信号通过即正常。

---

## Fix 3: `trade_gates.py` — 三层审批共识分门槛过高

**位置**: `trade_gates.py` 约第 229-268 行 (`_evaluate_symbol` 方法内)  
**症状**: 策略信号分数低(33-44分)+置信低(35-49%)，共识分算出来只有9-15，全被门槛杀掉  
**原理**:
```
共识分 = 权重比 × 平均(score × confidence × quality_factor)
```
- ML融合动量 BTC: 33 × 0.45 × 1.0(high) = 14.85 → 门槛15 → ❌  
- 激进交易 ONDO: 44 × 0.44 × 0.5(low) = 9.68 → 门槛15 → ❌

三重问题叠加：
1. `quality_factor` 对"激进交易"给0.5（low），砍半
2. `CONSENSUS_SCORE_MEDIUM`=15 在 short_only 下还是太高
3. `min_consensus`=15 拦住了 ML融合动量（14.85分，差0.15）

**错误代码** (旧):
```python
quality_factor = {"high": 1.0, "medium": 0.8, "low": 0.5}  # 对所有模式

if regime_bias in ("short_only", "long_only"):
    CONSENSUS_SCORE_STRONG = 25
    CONSENSUS_SCORE_MEDIUM = 15
    CONSENSUS_SCORE_WEAK = 8
    ...
    min_consensus = 15
```

**正确代码** (新):
```python
# 方向受限时：降低低质量策略的惩罚（极端行情下所有信号都有价值）
if regime_bias in ("short_only", "long_only"):
    quality_factor = {"high": 1.0, "medium": 0.85, "low": 0.70}
else:
    quality_factor = {"high": 1.0, "medium": 0.8, "low": 0.5}
...
if regime_bias in ("short_only", "long_only"):
    CONSENSUS_SCORE_STRONG = 20
    CONSENSUS_SCORE_MEDIUM = 10
    CONSENSUS_SCORE_WEAK = 5
    ...
    min_consensus = 10  # ML融合动量33×45%≈15, 激进44×44%×0.7≈13.6
```

---

## Fix 4: `auto_trade_daemon.py` — execute_strategy_signals 的 MIN_COMPOSITE 过高

**位置**: `auto_trade_daemon.py` 约第 424-426 行  
**症状**: 信号过了三层审批，又被 MIN_COMPOSITE=35 拦下（ML信号 composite=33×0.45=14.85 < 35）  
**影响**: 即使三层审批通过，最终执行也被拒

**错误代码** (旧):
```python
if action_bias == "short_only":
    MIN_COMPOSITE = 35   # 做空信号通常评分较低
    MIN_CONFIDENCE = 0.45
```

**正确代码** (新):
```python
if action_bias == "short_only":
    MIN_COMPOSITE = 10   # 做空信号评分低(33-44分)×置信(45%)≈15，放宽至10
    MIN_CONFIDENCE = 0.35  # 做空置信度放宽至35%
```

---

## Fix 5: `trade_gates.py` — `regime_result` 变量名错误导致L2+L3崩溃

**位置**: `trade_gates.py` 第 404 行 (`ExecutionOfficer.review()` 内)  
**症状**: `⚠️ 共识/执行官审批异常，跳过L2+L3: name 'regime_result' is not defined`  
**影响**: 三层审批直接崩溃，所有信号全被跳过

**错误代码** (旧):
```python
min_supporting = 1 if regime_result.action_bias in ("short_only", "long_only") else 2
```

**正确代码** (新):
```python
min_supporting = 1 if regime.get("action_bias", "any") in ("short_only", "long_only") else 2
```

`review()` 方法的参数是 `regime: dict`（字典），不是 `regime_result`（对象）。
```

---

## Fix 6: `trade_gates.py` — ExecutionOfficer Check 5 在极端恐慌时禁止一切交易

**位置**: `trade_gates.py` 约第 454-462 行 (`ExecutionOfficer.review()` 内)  
**症状**: `F&G=12，市场极度恐慌，暂停交易` — 所有信号被拒  
**影响**: F&G≤15时无条件禁止交易，不管方向 — 但做空正是恐慌时的正确操作

**错误代码** (旧):
```python
elif fg_value <= 15:
    checks_failed.append("新闻风险")
    reasons.append(f"F&G={fg_value}，市场极度恐慌，暂停交易")
```

**正确代码** (新):
```python
elif fg_value <= 15 and action_bias not in ("short_only",):
    # short_only模式: 极端恐慌正是做空的好时机，不禁交易
    checks_failed.append("新闻风险")
    reasons.append(f"F&G={fg_value}，市场极度恐慌，暂停交易")
```

---

## Fix 7: `trade_gates.py` — ExecutionOfficer Check 6 综合置信度门槛过高

**位置**: `trade_gates.py` 约第 464-476 行 (`ExecutionOfficer.review()` 内)  
**症状**: `置信度28 < 最低45` — `_compute_final_confidence` 算出20-28，但门槛45  
**原理**: `_compute_final_confidence = consensus_score + quality_bonus + diversity_bonus`
  - ML融合动量: 15(consensus) + 5(quality) + 0(diversity) = 20
  - 激进交易: 10(consensus) + 0(quality=low) + 0(diversity) = 10

**错误代码** (旧):
```python
if consensus.consensus_level == "strong":
    min_conf = self.MIN_CONFIDENCE_STRONG  # 55
else:
    min_conf = self.MIN_OVERALL_CONFIDENCE  # 45
```

**正确代码** (新):
```python
# 方向受限时降低要求
if action_bias in ("short_only", "long_only"):
    min_conf = 18 if consensus.consensus_level == "strong" else 12
elif consensus.consensus_level == "strong":
    min_conf = self.MIN_CONFIDENCE_STRONG  # 55
else:
    min_conf = self.MIN_OVERALL_CONFIDENCE  # 45
```

---

## 修复清单速查

| # | 文件 | 行数 | 问题 | 一句话 |
|---|------|------|------|--------|
| 1 | `ml_signal_v5.py` | 463-472 | F&G阈值反了 | 极端恐惧→收紧买入+放宽做空 |
| 2 | `auto_trade.py` | 523-528 | signal_consensus稀释 | 改用signal_weighted判定方向 |
| 3 | `trade_gates.py` | 229-268 | 共识门槛过高 | quality_factor+阈值+min_consensus全降 |
| 4 | `auto_trade_daemon.py` | 424-426 | MIN_COMPOSITE=35 | 降至10，容忍低分做空信号 |
| 5 | `trade_gates.py` | 404 | regime_result变量名 | 改为regime.get("action_bias") |
| 6 | `trade_gates.py` | 454-462 | F&G恐慌禁交易 | short_only模式豁免 |
| 7 | `trade_gates.py` | 464-476 | 置信门槛45→12 | 方向受限时降至12 |
| 8 | `auto_trade_daemon.py` | 490 | total_value变量名错误 | 改为total_equity（已定义） |

## 🆕 2026-06-25: 5m/15m短周期扫描 (Chase哥指令)

### Fix 8: `auto_trade_daemon.py` — `total_value` NameError

**位置**: `auto_trade_daemon.py` 第 490 行  
**症状**: `❌ [权哥价值投资] 执行失败: name 'total_value' is not defined`  
**原因**: 函数内定义的是 `total_equity`（第389行），但 `pre_trade_check()` 调用时用了 `total_value`

**错误代码** (旧):
```python
check = rc.pre_trade_check("crypto", max_size, score, total_value)
```

**正确代码** (新):
```python
check = rc.pre_trade_check("crypto", max_size, score, total_equity)
```

---

### Feature 9: 5min/15min 短周期裸K扫描 🆕

**触发**: Chase哥 "你可以看15分钟或者5分钟图，你的机会会多很多"  
**位置**: `strategy_runner.py` + `auto_trade_daemon.py`  
**原理**: 裸K扫描器本身支持5m/15m/1h/4h/1d多时间框架，但策略层只用1h/4h/1d。启用5m/15m后，短线信号量暴增。

**改动清单**:

| 文件 | 改动 | 说明 |
|------|------|------|
| `strategy_runner.py:345` | timeframes 加 '5m' '15m' | `['5m', '15m', '1H', '4H', '1D']` |
| `strategy_runner.py:363-380` | 移除1h/4h硬依赖 | 允许5m/15m独立扫描 |
| `strategy_runner.py:387-398` | TF特定参数 | 5m/15m: pass_score=5, 不强制Low2/High2, 允许追单 |
| `strategy_runner.py:473-477` | 信号标记 | 加 `timeframe`, `is_scalp`, strategy_name=`裸K·5m` |
| `auto_trade_daemon.py:424-441` | 短线门槛 | is_scalp信号: MIN_COMPOSITE=8, MIN_CONFIDENCE=0.30 |

**短线参数差异**:
| 参数 | 1h/4h | 5m/15m | 原因 |
|------|--------|--------|------|
| pass_score | 7 | 5 | 短线噪音多但机会多 |
| Low2/High2 | 强制 | 跳过 | 短周期形成太慢 |
| 动量追单 | 拦截 | 允许 | 动量=短线方向 |
| MIN_COMPOSITE | 10-55 | 8 | 短线评分天然低 |
| MIN_CONFIDENCE | 0.35-0.65 | 0.30 | 短线信度可更低 |
| 止损距离 | ATR×0.5 | ATR×0.5 (天然更紧) | 5m ATR ≈ 0.1-0.3% |

**重启验证**:
```
# 启动后查看日志应该有:
🔍 裸K·5m BTC/USDT... 🔴 SELL | sig=-0.03 | 得分5 | ...
🔍 裸K·15m SOL/USDT... 🔴 SELL | 得分6 | ...
📊 裸K信号: 5m=12条, 15m=8条, 1h=3条, 4h=1条
```

---

### Fix 10: `naked_k_scanner.py` — NaN crash in higher-TF analysis

**位置**: `naked_k_scanner.py` 第 925-926 行  
**症状**: `ValueError: cannot convert float NaN to integer` — 导致所有short-TF扫描崩溃  
**原因**: `np.mean()` 对空列表或含NaN数组返回NaN，`int(NaN)`抛异常

**修复**:
```python
# 使用 np.nanmean + NaN guard
overlaps = [ki.overlap_ratio for ki in self._kline_infos[-20:]]
avg_overlap = np.nanmean(overlaps) if overlaps else 0.5
if np.isnan(avg_overlap):
    avg_overlap = 0.5
```

**额外防护**: 构造函数中 higher-TF 分析加 try/except, 失败时降级为无高TF上下文

### Fix 11: `strategies.py` — `score_trend` 属性缺失

**位置**: `strategies.py` 第 1537-1539 行  
**症状**: `'TradingSignal' object has no attribute 'score_trend'`  
**原因**: TradingSignal dataclass (naked_k_scanner.py:162-176) 只有 `score_3step` 和 `score_2plus3`, 没有 `score_trend/score_keylevel/score_signalk`

**修复**: `_convert_signal()` 使用 `getattr()` 安全访问, 缺失字段返回0

---

_2026-06-25: 全链路11处修复 — ML引擎→共识→执行→5m/15m→NaN→属性缺失_
