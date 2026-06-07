"""
Alpha Miner v1.0 — 自动Alpha挖掘 表达式引擎
=============================================
Chase量化策略 Phase 12: 表达式驱动的Alpha因子自动发现

核心理念:
  用数学表达式定义Alpha因子 → 自动搜索最优参数 → 批量评估 → 入库排名

表达式示例:
  ts_delta(close, 5) / ts_std(close, 20)          # 短期动量 / 波动率
  rank(ts_roc(close, 10))                          # 排名化10日收益
  ts_corr(close, volume, 20) * sign(ts_delta(close, 5))  # 量价相关性 × 方向

三大挖掘策略:
  1. Grid Search:    模板参数网格搜索 — 快, 覆盖面可控
  2. Genetic:        遗传编程进化 — 探索新结构
  3. Random:         语法随机生成 — 发现意外惊喜

评估标准:
  - Rank IC vs 前向收益 (1d/3d/5d/10d/20d)
  - ICIR (IC/IC_std) — 稳定性
  - Long-Short Sharpe (top/bottom quintile)
  - Turnover — 换手率
  - FDR校正 — 多重比较

使用:
  python3 alpha_miner.py --evaluate "ts_delta(close,5)/ts_std(close,20)"
  python3 alpha_miner.py --mine --n 500
  python3 alpha_miner.py --evolve --generations 30
  python3 alpha_miner.py --list
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from scipy import stats
import json
import re
import random
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
ALPHA_DIR = DATA_DIR / "alphas"
ALPHA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Token & AST Types
# ═══════════════════════════════════════════════════════════

class TokenType:
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    CARET = "CARET"
    EOF = "EOF"

@dataclass
class Token:
    type: str
    value: Union[str, float]
    pos: int = 0

@dataclass
class ASTNode:
    """Base AST node"""
    pass

@dataclass
class BinOpNode(ASTNode):
    op: str           # '+', '-', '*', '/', '^'
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryOpNode(ASTNode):
    op: str           # 'neg', 'abs', 'sqrt', 'log', 'sign', 'rank', 'scale'
    operand: ASTNode

@dataclass
class FuncCallNode(ASTNode):
    name: str         # ts_sum, ts_mean, cs_rank, etc.
    args: List[ASTNode]

@dataclass
class VarNode(ASTNode):
    name: str         # open, high, low, close, volume

@dataclass
class ConstNode(ASTNode):
    value: float


# ═══════════════════════════════════════════════════════════
# Expression Parser — Tokenizer + Recursive Descent
# ═══════════════════════════════════════════════════════════

# Recognized functions (name → min_args, max_args)
TS_FUNCTIONS = {
    # Unary time-series (operate on one variable over a window)
    "ts_sum": (2, 2),       # ts_sum(x, d) → rolling sum
    "ts_mean": (2, 2),      # ts_mean(x, d) → rolling mean
    "ts_std": (2, 2),       # ts_std(x, d) → rolling std
    "ts_min": (2, 2),       # ts_min(x, d) → rolling min
    "ts_max": (2, 2),       # ts_max(x, d) → rolling max
    "ts_delta": (2, 2),     # ts_delta(x, d) → x - lag(x, d)
    "ts_delay": (2, 2),     # ts_delay(x, d) → lag(x, d)
    "ts_rank": (2, 2),      # ts_rank(x, d) → rolling percentile rank
    "ts_zscore": (2, 2),    # ts_zscore(x, d) → rolling z-score
    "ts_roc": (2, 2),       # ts_roc(x, d) → rate of change over d
    "ts_ema": (2, 2),       # ts_ema(x, d) → EMA with span d
    "ts_skew": (2, 2),      # ts_skew(x, d) → rolling skewness
    "ts_kurt": (2, 2),      # ts_kurt(x, d) → rolling kurtosis
    "ts_med": (2, 2),       # ts_med(x, d) → rolling median
    "ts_prod": (2, 2),      # ts_prod(x, d) → rolling product
    # Binary time-series
    "ts_corr": (3, 3),      # ts_corr(x, y, d) → rolling correlation
    "ts_cov": (3, 3),       # ts_cov(x, y, d) → rolling covariance
    "ts_beta": (3, 3),      # ts_beta(x, y, d) → rolling beta (slope)
    # Cross-sectional (operate across assets at a point)
    "cs_rank": (1, 1),      # cs_rank(x) → rank percentile across assets
    "cs_zscore": (1, 1),    # cs_zscore(x) → z-score across assets
    "cs_scale": (1, 2),     # cs_scale(x, a=1) → x*a / sum(abs(x))
}

VARIABLES = {"open", "high", "low", "close", "volume", "returns", "log_returns", "vwap"}

UNARY_OPS = {"abs", "sqrt", "log", "sign", "rank", "scale"}


class ParseError(Exception):
    """Expression parse error with position info"""
    def __init__(self, msg: str, pos: int = 0):
        super().__init__(f"Parse error at position {pos}: {msg}")
        self.pos = pos


class AlphaExpressionParser:
    """
    Recursive descent parser for alpha expressions.

    Grammar:
      expr     := term (('+' | '-') term)*
      term     := power (('*' | '/') power)*
      power    := factor ('^' factor)?
      factor   := unary_op factor | atom
      unary_op := '-' | 'abs' | 'sqrt' | 'log' | 'sign' | 'rank' | 'scale'
      atom     := NUMBER | VARIABLE | func_call | '(' expr ')'
      func_call:= IDENT '(' args ')'
      args     := expr (',' expr)*
    """

    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0

    # ── Tokenizer ──
    def tokenize(self, expression: str) -> List[Token]:
        """Convert expression string to token list"""
        tokens = []
        i = 0
        n = len(expression)

        while i < n:
            ch = expression[i]

            # Whitespace
            if ch.isspace():
                i += 1
                continue

            # Number (including decimals and negatives only in context)
            if ch.isdigit() or (ch == '.' and i + 1 < n and expression[i + 1].isdigit()):
                start = i
                while i < n and (expression[i].isdigit() or expression[i] == '.'):
                    i += 1
                tokens.append(Token(TokenType.NUMBER, float(expression[start:i]), start))
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == '_':
                start = i
                while i < n and (expression[i].isalnum() or expression[i] == '_'):
                    i += 1
                tokens.append(Token(TokenType.IDENT, expression[start:i], start))
                continue

            # Operators and punctuation
            if ch == '(':
                tokens.append(Token(TokenType.LPAREN, '(', i))
            elif ch == ')':
                tokens.append(Token(TokenType.RPAREN, ')', i))
            elif ch == ',':
                tokens.append(Token(TokenType.COMMA, ',', i))
            elif ch == '+':
                tokens.append(Token(TokenType.PLUS, '+', i))
            elif ch == '-':
                tokens.append(Token(TokenType.MINUS, '-', i))
            elif ch == '*':
                tokens.append(Token(TokenType.STAR, '*', i))
            elif ch == '/':
                tokens.append(Token(TokenType.SLASH, '/', i))
            elif ch == '^':
                tokens.append(Token(TokenType.CARET, '^', i))
            else:
                raise ParseError(f"Unexpected character: '{ch}'", i)

            i += 1

        tokens.append(Token(TokenType.EOF, '', n))
        return tokens

    # ── Parser ──
    def parse(self, expression: str) -> ASTNode:
        """Parse an expression string into an AST"""
        self.tokens = self.tokenize(expression)
        self.pos = 0
        ast = self._expr()
        if self._current().type != TokenType.EOF:
            t = self._current()
            raise ParseError(f"Unexpected token: {t.value}", t.pos)
        return ast

    def _current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, '', 0)

    def _peek(self) -> Token:
        return self._current()

    def _advance(self) -> Token:
        t = self._current()
        self.pos += 1
        return t

    def _expect(self, ttype: str) -> Token:
        t = self._advance()
        if t.type != ttype:
            raise ParseError(f"Expected {ttype}, got {t.type} ('{t.value}')", t.pos)
        return t

    def _expr(self) -> ASTNode:
        """expr := term (('+' | '-') term)*"""
        left = self._term()
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance().value
            right = self._term()
            left = BinOpNode(op=op, left=left, right=right)
        return left

    def _term(self) -> ASTNode:
        """term := power (('*' | '/') power)*"""
        left = self._power()
        while self._peek().type in (TokenType.STAR, TokenType.SLASH):
            op = self._advance().value
            right = self._power()
            left = BinOpNode(op=op, left=left, right=right)
        return left

    def _power(self) -> ASTNode:
        """power := factor ('^' factor)?"""
        left = self._factor()
        if self._peek().type == TokenType.CARET:
            self._advance()
            right = self._factor()
            left = BinOpNode(op='^', left=left, right=right)
        return left

    def _factor(self) -> ASTNode:
        """factor := unary_op factor | atom"""
        t = self._peek()

        # Unary minus
        if t.type == TokenType.MINUS:
            self._advance()
            return UnaryOpNode(op='neg', operand=self._factor())

        # Named unary operators: abs, sqrt, log, sign, rank, scale
        if t.type == TokenType.IDENT and t.value in UNARY_OPS:
            op_name = self._advance().value
            # These can be followed by a factor without parens: abs(x) or abs x
            # For simplicity and clarity, we require parenthesized argument
            operand = self._factor()
            return UnaryOpNode(op=op_name, operand=operand)

        return self._atom()

    def _atom(self) -> ASTNode:
        """atom := NUMBER | VARIABLE | func_call | '(' expr ')'"""
        t = self._peek()

        if t.type == TokenType.NUMBER:
            self._advance()
            return ConstNode(value=float(t.value))

        if t.type == TokenType.LPAREN:
            self._advance()
            node = self._expr()
            self._expect(TokenType.RPAREN)
            return node

        if t.type == TokenType.IDENT:
            name = t.value
            self._advance()
            # Function call?
            if self._peek().type == TokenType.LPAREN:
                return self._finish_func_call(name)
            # Variable
            if name in VARIABLES:
                return VarNode(name=name)
            raise ParseError(f"Unknown variable: '{name}'. Known: {sorted(VARIABLES)}", t.pos)

        raise ParseError(f"Unexpected token: {t.type} ('{t.value}')", t.pos)

    def _finish_func_call(self, name: str) -> FuncCallNode:
        """Parse function arguments: name '(' args ')'"""
        self._expect(TokenType.LPAREN)
        args = []
        if self._peek().type != TokenType.RPAREN:
            args.append(self._expr())
            while self._peek().type == TokenType.COMMA:
                self._advance()
                args.append(self._expr())
        self._expect(TokenType.RPAREN)

        # Validate function
        if name not in TS_FUNCTIONS:
            raise ParseError(
                f"Unknown function: '{name}'. Known: {sorted(TS_FUNCTIONS.keys())}",
                self._peek().pos)
        min_args, max_args = TS_FUNCTIONS[name]
        if not (min_args <= len(args) <= max_args):
            raise ParseError(
                f"Function '{name}' expects {min_args}-{max_args} args, got {len(args)}",
                self._peek().pos)

        return FuncCallNode(name=name, args=args)


# ═══════════════════════════════════════════════════════════
# Expression Evaluator — AST → np.ndarray
# ═══════════════════════════════════════════════════════════

def _roll_sum(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).sum().values

def _roll_mean(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).mean().values

def _roll_std(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).std().values

def _roll_min(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).min().values

def _roll_max(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).max().values

def _roll_skew(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(20, d)).skew().values

def _roll_kurt(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(20, d)).kurt().values

def _roll_med(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(3, d//2)).median().values

def _roll_rank(x: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(x).rolling(d, min_periods=max(10, d//2)).rank(pct=True).values

def _roll_corr(a: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
    return pd.Series(a).rolling(d, min_periods=max(10, d//2)).corr(pd.Series(b)).values

def _roll_beta(a: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
    """Rolling beta of a vs b (slope of a ~ b)"""
    n = len(a)
    result = np.full(n, np.nan)
    for i in range(d, n):
        x = b[i-d:i]
        y = a[i-d:i]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            continue
        xm, ym = x[mask], y[mask]
        beta = np.cov(xm, ym)[0, 1] / (np.var(xm) + 1e-9)
        result[i] = beta
    return result


def evaluate_ast(ast: ASTNode, data: Dict[str, np.ndarray],
                 n_assets: int = 1, asset_idx: int = 0) -> np.ndarray:
    """
    Evaluate an AST node on the given data dictionary.

    Args:
        ast: The parsed AST
        data: Dict mapping variable names → np.ndarray (n_timesteps,)
              OR (n_timesteps, n_assets) for multi-asset
        n_assets: Number of assets (for cross-sectional ops)
        asset_idx: Which asset index to extract single-asset data from
    """
    if isinstance(ast, ConstNode):
        n = len(next(iter(data.values())))
        return np.full(n if n_assets == 1 else n_assets, ast.value)

    if isinstance(ast, VarNode):
        val = data.get(ast.name)
        if val is None:
            # Try to derive
            if ast.name == "returns":
                close = data["close"]
                ret = np.zeros_like(close)
                ret[1:] = close[1:] / close[:-1] - 1
                return ret
            if ast.name == "log_returns":
                close = data["close"]
                lret = np.zeros_like(close)
                lret[1:] = np.log(close[1:] / close[:-1])
                return lret
            if ast.name == "vwap":
                high, low, close, vol = data["high"], data["low"], data["close"], data["volume"]
                typical = (high + low + close) / 3
                return (typical * vol).cumsum() / (vol.cumsum() + 1e-9)
            raise ValueError(f"Unknown variable: {ast.name}")
        return val

    if isinstance(ast, UnaryOpNode):
        operand = evaluate_ast(ast.operand, data, n_assets, asset_idx)
        if ast.op == 'neg':
            return -operand
        if ast.op == 'abs':
            return np.abs(operand)
        if ast.op == 'sqrt':
            return np.sqrt(np.maximum(operand, 0))
        if ast.op == 'log':
            return np.log(np.maximum(operand, 1e-9))
        if ast.op == 'sign':
            return np.sign(operand)
        if ast.op == 'rank':
            valid = ~np.isnan(operand)
            ranked = np.full(len(operand), np.nan)
            ranked[valid] = stats.rankdata(operand[valid]) / valid.sum()
            return ranked
        if ast.op == 'scale':
            denom = np.nansum(np.abs(operand)) + 1e-9
            return operand / denom
        raise ValueError(f"Unknown unary op: {ast.op}")

    if isinstance(ast, BinOpNode):
        left = evaluate_ast(ast.left, data, n_assets, asset_idx)
        right = evaluate_ast(ast.right, data, n_assets, asset_idx)
        # Broadcasting: handle scalar right with array left and vice versa
        if isinstance(right, (int, float)) or (isinstance(right, np.ndarray) and right.ndim == 0):
            right = float(right)
        if isinstance(left, (int, float)) or (isinstance(left, np.ndarray) and left.ndim == 0):
            left = float(left)

        if ast.op == '+':
            return left + right
        if ast.op == '-':
            return left - right
        if ast.op == '*':
            return left * right
        if ast.op == '/':
            denom = np.where(np.abs(right) < 1e-9, np.nan, right)
            return left / denom
        if ast.op == '^':
            return np.power(np.maximum(left, 0), right)
        raise ValueError(f"Unknown binop: {ast.op}")

    if isinstance(ast, FuncCallNode):
        return _eval_func_call(ast, data, n_assets, asset_idx)

    raise ValueError(f"Unknown AST node type: {type(ast)}")


def _eval_func_call(ast: FuncCallNode, data: Dict[str, np.ndarray],
                    n_assets: int, asset_idx: int) -> np.ndarray:
    """Evaluate a function call node"""
    name = ast.name
    args = [evaluate_ast(a, data, n_assets, asset_idx) for a in ast.args]

    # For multi-asset data, extract single asset if needed
    def _ensure_1d(arr):
        if isinstance(arr, np.ndarray) and arr.ndim > 1:
            if arr.shape[1] > asset_idx:
                return arr[:, asset_idx]
            return arr[:, 0]
        return arr

    # Extract scalar from constant array for window/int parameters
    def _as_scalar(arr, default=1):
        if isinstance(arr, np.ndarray):
            return int(arr.flat[0]) if arr.size > 0 else default
        return int(arr) if arr is not None else default

    try:
        if name == "ts_sum":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_sum(x, d)
        if name == "ts_mean":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_mean(x, d)
        if name == "ts_std":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_std(x, d)
        if name == "ts_min":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_min(x, d)
        if name == "ts_max":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_max(x, d)
        if name == "ts_delta":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            result = np.full(len(x), np.nan)
            result[d:] = x[d:] - x[:-d]
            return result
        if name == "ts_delay":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            result = np.full(len(x), np.nan)
            result[d:] = x[:-d]
            return result
        if name == "ts_rank":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_rank(x, d)
        if name == "ts_zscore":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            mu = _roll_mean(x, d)
            sigma = _roll_std(x, d)
            return (x - mu) / (sigma + 1e-9)
        if name == "ts_roc":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            result = np.full(len(x), np.nan)
            result[d:] = x[d:] / np.where(x[:-d] == 0, np.nan, x[:-d]) - 1
            return result
        if name == "ts_ema":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return pd.Series(x).ewm(span=d, min_periods=d//2).mean().values
        if name == "ts_skew":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_skew(x, d)
        if name == "ts_kurt":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_kurt(x, d)
        if name == "ts_med":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return _roll_med(x, d)
        if name == "ts_prod":
            x, d = _ensure_1d(args[0]), _as_scalar(args[1])
            return pd.Series(x).rolling(d, min_periods=max(3, d//2)).apply(
                lambda s: np.prod(s), raw=True).values
        if name == "ts_corr":
            x, y, d = _ensure_1d(args[0]), _ensure_1d(args[1]), _as_scalar(args[2])
            return _roll_corr(x, y, d)
        if name == "ts_cov":
            x, y, d = _ensure_1d(args[0]), _ensure_1d(args[1]), _as_scalar(args[2])
            rho = _roll_corr(x, y, d)
            sx = _roll_std(x, d)
            sy = _roll_std(y, d)
            return rho * sx * sy
        if name == "ts_beta":
            x, y, d = _ensure_1d(args[0]), _ensure_1d(args[1]), _as_scalar(args[2])
            return _roll_beta(x, y, d)

        # Cross-sectional ops
        if name == "cs_rank":
            x = args[0]
            if x.ndim == 1:
                return (stats.rankdata(x, nan_policy='omit') - 1) / (np.sum(~np.isnan(x)) - 1 + 1e-9)
            # 2D case: rank across assets at each timestep
            result = np.full_like(x, np.nan)
            for t in range(x.shape[0]):
                row = x[t]
                valid = ~np.isnan(row)
                if valid.sum() > 1:
                    result[t, valid] = (stats.rankdata(row[valid]) - 1) / (valid.sum() - 1)
            return result
        if name == "cs_zscore":
            x = args[0]
            if x.ndim == 1:
                mu = np.nanmean(x)
                sigma = np.nanstd(x)
                return (x - mu) / (sigma + 1e-9)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sigma = np.nanstd(x, axis=1, keepdims=True)
            return (x - mu) / (sigma + 1e-9)
        if name == "cs_scale":
            x = args[0]
            scale = float(args[1]) if len(args) > 1 else 1.0
            if x.ndim == 1:
                denom = np.nansum(np.abs(x)) + 1e-9
                return x * scale / denom
            denom = np.nansum(np.abs(x), axis=1, keepdims=True) + 1e-9
            return x * scale / denom

    except Exception as e:
        # Return NaN array on evaluation error
        n = len(args[0]) if isinstance(args[0], np.ndarray) else 100
        return np.full(n, np.nan)

    raise ValueError(f"Unknown function: {name}")


def evaluate_expression(expr: str, df: pd.DataFrame = None,
                        data: Dict[str, np.ndarray] = None) -> np.ndarray:
    """
    Parse and evaluate an alpha expression.

    Args:
        expr: Alpha expression string
        df: OHLCV DataFrame (used to build data dict if data is None)
        data: Dict of variable name → np.ndarray

    Returns:
        np.ndarray of alpha values
    """
    parser = AlphaExpressionParser()
    ast = parser.parse(expr)

    if data is None and df is not None:
        data = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data[col] = df[col].values.astype(float)

    if data is None:
        raise ValueError("Either df or data must be provided")

    return evaluate_ast(ast, data)


def quick_ic(expr: str, df: pd.DataFrame, fwd: int = 5) -> float:
    """Quick rank IC computation for a single expression"""
    alpha = evaluate_expression(expr, df)
    close = df["close"].values
    n = len(close)
    fwd_ret = np.zeros(n)
    fwd_ret[:n-fwd] = close[fwd:] / close[:n-fwd] - 1

    valid = ~(np.isnan(alpha) | np.isnan(fwd_ret))
    if valid.sum() < 30:
        return 0.0
    try:
        ic, _ = stats.spearmanr(alpha[valid], fwd_ret[valid])
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
# Alpha Result & Store
# ═══════════════════════════════════════════════════════════

@dataclass
class AlphaResult:
    """Evaluated alpha factor result"""
    expression: str
    name: str = ""
    category: str = "custom"
    description: str = ""

    # Rank IC metrics
    rank_ic: float = 0.0
    icir: float = 0.0
    ic_std: float = 0.0
    ic_decay: Dict[int, float] = field(default_factory=dict)

    # Performance
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    hit_rate: float = 0.0

    # Stats
    fdr_p_value: float = 1.0
    correlation_with_existing: float = 0.0
    passed: bool = False
    n_obs: int = 0
    fwd_window: int = 5

    # Metadata
    generation: str = "manual"   # grid | genetic | random | manual
    params: Dict[str, any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def summary(self) -> str:
        check = "✅" if self.passed else "❌"
        return (
            f"{check} {self.name:30s} | "
            f"IC={self.rank_ic:+.3f} | ICIR={self.icir:+.2f} | "
            f"Sh={self.sharpe:+.2f} | TO={self.turnover:.3f} | "
            f"[{self.category}] {self.generation}"
        )

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "rank_ic": self.rank_ic,
            "icir": self.icir,
            "ic_std": self.ic_std,
            "ic_decay": {str(k): v for k, v in self.ic_decay.items()},
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "hit_rate": self.hit_rate,
            "fdr_p_value": self.fdr_p_value,
            "correlation_with_existing": self.correlation_with_existing,
            "passed": self.passed,
            "n_obs": self.n_obs,
            "fwd_window": self.fwd_window,
            "generation": self.generation,
            "params": self.params,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlphaResult":
        ic_decay = {int(k): v for k, v in d.get("ic_decay", {}).items()}
        return cls(
            expression=d["expression"],
            name=d.get("name", ""),
            category=d.get("category", "custom"),
            description=d.get("description", ""),
            rank_ic=d.get("rank_ic", 0.0),
            icir=d.get("icir", 0.0),
            ic_std=d.get("ic_std", 0.0),
            ic_decay=ic_decay,
            sharpe=d.get("sharpe", 0.0),
            max_drawdown=d.get("max_drawdown", 0.0),
            turnover=d.get("turnover", 0.0),
            hit_rate=d.get("hit_rate", 0.0),
            fdr_p_value=d.get("fdr_p_value", 1.0),
            correlation_with_existing=d.get("correlation_with_existing", 0.0),
            passed=d.get("passed", False),
            n_obs=d.get("n_obs", 0),
            fwd_window=d.get("fwd_window", 5),
            generation=d.get("generation", "manual"),
            params=d.get("params", {}),
            created_at=d.get("created_at", ""),
        )


class AlphaStore:
    """Persist discovered alphas to disk"""

    def __init__(self, store_dir: Path = None):
        self.store_dir = store_dir or ALPHA_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: List[AlphaResult], name: str = None) -> Path:
        """Save alpha results to JSON"""
        if name is None:
            name = f"alphas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self.store_dir / f"{name}.json"
        data = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "n_alphas": len(results),
            "alphas": [r.to_dict() for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Also save as latest
        latest_path = self.store_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def load(self, name: str = "latest") -> List[AlphaResult]:
        """Load alpha results from JSON"""
        path = self.store_dir / f"{name}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [AlphaResult.from_dict(d) for d in data.get("alphas", [])]

    def list_saved(self) -> List[Dict]:
        """List all saved alpha files"""
        files = []
        for f in sorted(self.store_dir.glob("*.json"), reverse=True):
            if f.stem == "latest":
                continue
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                files.append({
                    "name": data.get("name", f.stem),
                    "saved_at": data.get("saved_at", ""),
                    "n_alphas": data.get("n_alphas", 0),
                    "path": str(f),
                })
            except Exception:
                pass
        return files

    def get_top(self, n: int = 20, min_icir: float = 0.0) -> List[AlphaResult]:
        """Get top N alphas by ICIR"""
        alphas = self.load("latest")
        if min_icir > 0:
            alphas = [a for a in alphas if abs(a.icir) >= min_icir]
        alphas.sort(key=lambda a: abs(a.icir), reverse=True)
        return alphas[:n]

    def get_best_by_category(self, n: int = 5) -> Dict[str, List[AlphaResult]]:
        """Get best alphas per category"""
        alphas = self.load("latest")
        by_cat = {}
        for a in alphas:
            by_cat.setdefault(a.category, []).append(a)
        result = {}
        for cat, items in by_cat.items():
            items.sort(key=lambda a: abs(a.icir), reverse=True)
            result[cat] = items[:n]
        return result


# ═══════════════════════════════════════════════════════════
# Alpha Template Library
# ═══════════════════════════════════════════════════════════

@dataclass
class AlphaTemplate:
    """Pre-built alpha expression template"""
    name: str
    expression: str        # With {w1}, {w2}, ... placeholders
    category: str          # momentum / mean_reversion / volatility / volume / cross_sectional
    description: str
    param_bounds: Dict[str, Tuple[int, int, int]]  # param → (min, max, step)
    # e.g. {"w1": (5, 60, 5), "w2": (20, 200, 10)}


class AlphaTemplateLibrary:
    """50+ pre-defined alpha expression templates"""

    CATEGORIES = ["momentum", "mean_reversion", "volatility", "volume", "cross_sectional"]

    TEMPLATES: List[AlphaTemplate] = [
        # ═══ Momentum (趋势/动量) ═══
        AlphaTemplate("动量比率", "ts_delta(close, {w1}) / ts_std(close, {w2})",
                      "momentum", "短期价格变化 ÷ 波动率归一化",
                      {"w1": (1, 20, 1), "w2": (10, 60, 5)}),
        AlphaTemplate("ROC", "ts_roc(close, {w1})",
                      "momentum", "N日收益率",
                      {"w1": (1, 60, 1)}),
        AlphaTemplate("多窗口动量差", "ts_roc(close, {w1}) - ts_roc(close, {w2})",
                      "momentum", "短期-长期动量差 (加速度)",
                      {"w1": (3, 20, 1), "w2": (20, 120, 5)}),
        AlphaTemplate("EMA交叉", "(ts_ema(close, {w1}) - ts_ema(close, {w2})) / ts_std(close, {w2})",
                      "momentum", "双EMA交叉信号 (金叉/死叉)",
                      {"w1": (3, 20, 1), "w2": (20, 200, 10)}),
        AlphaTemplate("价格-均线偏离", "(close - ts_mean(close, {w1})) / ts_std(close, {w1})",
                      "momentum", "价格偏离移动均线的标准化距离",
                      {"w1": (5, 120, 5)}),
        AlphaTemplate("趋势强度", "ts_delta(close, {w1}) / (ts_max(close, {w1}) - ts_min(close, {w1}) + 1e-9)",
                      "momentum", "价格变动 ÷ 区间振幅",
                      {"w1": (5, 60, 5)}),
        AlphaTemplate("夏普动量", "ts_mean(returns, {w1}) / (ts_std(returns, {w1}) + 1e-9)",
                      "momentum", "滚动夏普比率",
                      {"w1": (10, 120, 10)}),
        AlphaTemplate("新高比率", "(close - ts_max(close, {w1})) / (ts_max(close, {w1}) - ts_min(close, {w1}) + 1e-9)",
                      "momentum", "当前价 vs 区间高点的相对位置",
                      {"w1": (20, 120, 10)}),
        AlphaTemplate("价格加速度", "(ts_delta(close, {w1}) - ts_delay(ts_delta(close, {w1}), {w2})) / ts_std(close, {w2})",
                      "momentum", "二阶动量变化",
                      {"w1": (3, 10, 1), "w2": (10, 30, 5)}),
        AlphaTemplate("排名动量", "rank(ts_roc(close, {w1}))",
                      "momentum", "原始动量排名化 (消除极端值)",
                      {"w1": (5, 60, 5)}),

        # ═══ Mean Reversion (均值回归) ═══
        AlphaTemplate("布林带位置", "(close - ts_mean(close, {w1})) / (ts_std(close, {w1}) * {w2})",
                      "mean_reversion", "布林带标准化位置",
                      {"w1": (10, 60, 5), "w2": (1, 3, 1)}),
        AlphaTemplate("RSI风格", "ts_sum(sign(ts_delta(close, 1)), {w1}) / {w1}",
                      "mean_reversion", "近期涨跌天数占比",
                      {"w1": (5, 30, 1)}),
        AlphaTemplate("反转预期", "-ts_roc(close, {w1}) / (ts_std(close, {w2}) + 1e-9)",
                      "mean_reversion", "过度涨跌后的反向回归预期",
                      {"w1": (3, 20, 1), "w2": (10, 60, 5)}),
        AlphaTemplate("偏离均值程度", "abs(close - ts_mean(close, {w1})) / ts_std(close, {w1})",
                      "mean_reversion", "当前价格偏离均值的标准化程度",
                      {"w1": (10, 120, 10)}),
        AlphaTemplate("连涨连跌", "sign(ts_delta(close, 1)) * ts_sum(abs(ts_delta(close, 1)), {w1}) / ts_std(close, {w1})",
                      "mean_reversion", "方向 × 累积波动 = 趋势疲劳度",
                      {"w1": (5, 30, 1)}),
        AlphaTemplate("通道突破", "(close - ts_min(close, {w1})) / (ts_max(close, {w1}) - ts_min(close, {w1}) + 1e-9) - 0.5",
                      "mean_reversion", "价格在区间内的相对位置 (0=底部 1=顶部)",
                      {"w1": (10, 60, 5)}),
        AlphaTemplate("Hurst风格反转", "-ts_delta(close, {w1}) * abs(ts_delta(close, {w1})) / (ts_std(close, {w1}) + 1e-9)",
                      "mean_reversion", "趋势越强越倾向于反转",
                      {"w1": (5, 30, 1)}),
        AlphaTemplate("波动率加权反转", "-ts_delta(close, {w1}) / (ts_std(close, {w1}) * ts_std(close, {w2}))",
                      "mean_reversion", "短期vs长期波动率加权反转信号",
                      {"w1": (1, 10, 1), "w2": (20, 60, 10)}),
        AlphaTemplate("Z分数", "-ts_zscore(close, {w1})",
                      "mean_reversion", "负Z分数=超卖反弹预期",
                      {"w1": (10, 60, 5)}),

        # ═══ Volatility (波动率结构) ═══
        AlphaTemplate("波动率扩张", "ts_std(close, {w1}) / ts_std(close, {w2}) - 1",
                      "volatility", "短期/长期波动率比 — 波动率扩张信号",
                      {"w1": (5, 20, 1), "w2": (20, 120, 10)}),
        AlphaTemplate("波动率回归", "-(ts_std(close, {w1}) / ts_std(close, {w2}) - 1)",
                      "volatility", "高波动后回归正常的预期",
                      {"w1": (3, 10, 1), "w2": (20, 60, 10)}),
        AlphaTemplate("量价波动比", "ts_std(close, {w1}) / (ts_std(volume, {w1}) + 1e-9)",
                      "volatility", "价格波动 ÷ 成交量波动",
                      {"w1": (10, 60, 5)}),
        AlphaTemplate("收益率离散度", "ts_std(returns, {w1}) * sqrt({w1})",
                      "volatility", "年化波动率 (近似)",
                      {"w1": (5, 60, 5)}),
        AlphaTemplate("尾部风险", "ts_min(returns, {w1}) / (ts_std(returns, {w1}) + 1e-9)",
                      "volatility", "最差日收益/波动率 — 尾部风险度量",
                      {"w1": (20, 120, 10)}),
        AlphaTemplate("偏度信号", "-ts_skew(returns, {w1})",
                      "volatility", "负偏度→涨, 正偏度→跌 (回归对称)",
                      {"w1": (20, 120, 10)}),
        AlphaTemplate("峰度风险", "ts_kurt(returns, {w1})",
                      "volatility", "高锋度=更多极端事件=风险升水",
                      {"w1": (20, 120, 10)}),
        AlphaTemplate("波动率趋势", "ts_delta(ts_std(close, {w1}), {w2})",
                      "volatility", "波动率一阶变化 (vol-of-vol)",
                      {"w1": (10, 40, 5), "w2": (5, 20, 5)}),
        AlphaTemplate("高低价差比", "(ts_max(high, {w1}) - ts_min(low, {w1})) / (ts_mean(close, {w1}) + 1e-9)",
                      "volatility", "区间振幅 ÷ 均价",
                      {"w1": (5, 60, 5)}),

        # ═══ Volume (成交量画像) ═══
        AlphaTemplate("量价相关", "ts_corr(close, volume, {w1})",
                      "volume", "量价相关性 — 正=放量上涨",
                      {"w1": (10, 60, 5)}),
        AlphaTemplate("成交量比率", "ts_mean(volume, {w1}) / ts_mean(volume, {w2})",
                      "volume", "短期/长期成交量比",
                      {"w1": (3, 10, 1), "w2": (20, 120, 10)}),
        AlphaTemplate("量价背离", "ts_roc(close, {w1}) / (sign(ts_roc(volume, {w1})) + 1e-9)",
                      "volume", "放量下跌=负, 缩量上涨=正",
                      {"w1": (5, 30, 5)}),
        AlphaTemplate("成交量加速度", "ts_delta(ts_mean(volume, {w1}), {w2}) / (ts_std(volume, {w2}) + 1e-9)",
                      "volume", "成交量变化率",
                      {"w1": (3, 10, 1), "w2": (10, 30, 5)}),
        AlphaTemplate("换手率代理", "volume / ts_mean(volume, {w1})",
                      "volume", "当日量 ÷ 均值 = 相对活跃度",
                      {"w1": (10, 120, 10)}),
        AlphaTemplate("量价同步率", "sign(ts_delta(close, {w1})) * sign(ts_delta(volume, {w1}))",
                      "volume", "+1=量价同步, -1=量价背离",
                      {"w1": (3, 20, 1)}),
        AlphaTemplate("放量突破", "(close - ts_max(close, {w1})) * volume / (ts_mean(volume, {w2}) + 1e-9)",
                      "volume", "突破区间高点 + 放量确认",
                      {"w1": (10, 60, 5), "w2": (10, 60, 10)}),
        AlphaTemplate("成交量趋势", "ts_ema(volume, {w1}) - ts_ema(volume, {w2})",
                      "volume", "短长期成交量EMA差",
                      {"w1": (3, 12, 1), "w2": (20, 60, 10)}),

        # ═══ Cross Sectional (截面 — 多资产比较) ═══
        AlphaTemplate("截面动量排名", "cs_rank(ts_roc(close, {w1}))",
                      "cross_sectional", "N日收益在全部资产中的排名",
                      {"w1": (5, 60, 5)}),
        AlphaTemplate("截面波动排名", "cs_rank(ts_std(close, {w1}))",
                      "cross_sectional", "波动率在资产池中的百分位",
                      {"w1": (10, 60, 5)}),
        AlphaTemplate("截面量价综合", "cs_rank(ts_delta(close, {w1}) / (ts_std(close, {w1}) + 1e-9))",
                      "cross_sectional", "夏普比的截面排名",
                      {"w1": (10, 40, 5)}),
        AlphaTemplate("截面Z分数", "cs_zscore(ts_roc(close, {w1}))",
                      "cross_sectional", "收益率截面Z分数",
                      {"w1": (5, 30, 5)}),

        # ═══ Advanced / Hybrid (混合) ═══
        AlphaTemplate("动量+波动", "(ts_delta(close, {w1}) / ts_std(close, {w2})) * abs(ts_corr(close, volume, {w3}))",
                      "momentum", "动量 × 量价相关度 (缩量趋势更可信)",
                      {"w1": (3, 20, 1), "w2": (10, 60, 5), "w3": (10, 40, 5)}),
        AlphaTemplate("反转+量确认", "-(ts_roc(close, {w1})) * (volume / (ts_mean(volume, {w2}) + 1e-9))",
                      "mean_reversion", "反转信号 × 相对成交量",
                      {"w1": (3, 10, 1), "w2": (10, 60, 10)}),
        AlphaTemplate("趋势-反转混合", "ts_roc(close, {w1}) / (ts_std(close, {w1}) + 1e-9) - ts_zscore(close, {w2})",
                      "momentum", "短期动量 - 长期过度偏离",
                      {"w1": (5, 20, 1), "w2": (40, 120, 20)}),
        AlphaTemplate("波动率调整收益", "ts_mean(returns, {w1}) / (ts_std(returns, {w1}) * power(1 + abs(ts_skew(returns, {w2})), 0.5))",
                      "volatility", "经偏度调整的夏普比",
                      {"w1": (10, 60, 10), "w2": (20, 120, 10)}),
        AlphaTemplate("信息比率风格", "(ts_roc(close, {w1}) - ts_roc(close, {w2})) / (ts_std(ts_delta(close, 1), {w3}) + 1e-9)",
                      "momentum", "超额收益 / 跟踪误差",
                      {"w1": (5, 30, 5), "w2": (40, 120, 20), "w3": (10, 60, 10)}),
        AlphaTemplate("自适应动量", "ts_delta(close, {w1}) / (ts_std(close, {w1}) + 1e-9) * (1 - abs(ts_corr(close, ts_ema(close, {w2}), {w3})))",
                      "momentum", "动量 × (1-趋势一致性) — 区分趋势vs震荡",
                      {"w1": (5, 30, 5), "w2": (20, 120, 20), "w3": (10, 40, 10)}),
    ]

    @classmethod
    def get_by_category(cls, category: str) -> List[AlphaTemplate]:
        return [t for t in cls.TEMPLATES if t.category == category]

    @classmethod
    def get_all(cls) -> List[AlphaTemplate]:
        return list(cls.TEMPLATES)

    @classmethod
    def get_categories(cls) -> List[str]:
        return cls.CATEGORIES

    @classmethod
    def instantiate(cls, template: AlphaTemplate, params: Dict[str, int]) -> Tuple[str, str]:
        """Fill template parameters → (expression, name)"""
        expr = template.expression
        name_parts = [template.name]
        for key, val in params.items():
            expr = expr.replace(f"{{{key}}}", str(val))
            name_parts.append(f"{key}={val}")
        return expr, "_".join(name_parts[:3])  # Keep name short

    @classmethod
    def generate_grid_params(cls, template: AlphaTemplate, n: int = 20) -> List[Dict[str, int]]:
        """Generate parameter combinations for grid search"""
        keys = list(template.param_bounds.keys())
        if not keys:
            return [{}]

        # Generate ranges
        ranges = []
        for key in keys:
            lo, hi, step = template.param_bounds[key]
            vals = list(range(lo, hi + 1, step))
            ranges.append(vals)

        # Cartesian product, sample if too many
        import itertools
        all_combos = list(itertools.product(*ranges))
        if len(all_combos) > n:
            # Random sample
            random.shuffle(all_combos)
            all_combos = all_combos[:n]

        return [{k: v for k, v in zip(keys, combo)} for combo in all_combos]


# ═══════════════════════════════════════════════════════════
# Alpha Evaluator
# ═══════════════════════════════════════════════════════════

class AlphaEvaluator:
    """Evaluate alpha expressions on historical data"""

    def __init__(self, fwd_windows: List[int] = None,
                 min_obs: int = 60,
                 icir_threshold: float = 0.2,
                 fdr_alpha: float = 0.1):
        self.fwd_windows = fwd_windows or [1, 3, 5, 10, 20]
        self.min_obs = min_obs
        self.icir_threshold = icir_threshold
        self.fdr_alpha = fdr_alpha
        self.parser = AlphaExpressionParser()

    def evaluate(self, expression: str, df: pd.DataFrame,
                 name: str = "", category: str = "custom",
                 generation: str = "manual",
                 params: Dict[str, any] = None) -> AlphaResult:
        """Evaluate a single alpha expression"""
        close = df["close"].values
        n = len(close)

        # Build data dictionary
        data = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data[col] = df[col].values.astype(float)

        # Evaluate expression
        try:
            alpha_ts = evaluate_expression(expression, data=data)
        except Exception as e:
            return AlphaResult(
                expression=expression, name=name or expression[:40],
                category=category, generation=generation,
                params=params or {}, rank_ic=0.0, passed=False)

        # Compute forward returns for each window
        fwd_rets_all = {}
        for fwd in self.fwd_windows:
            fwd_ret = np.zeros(n)
            fwd_ret[:n-fwd] = close[fwd:] / close[:n-fwd] - 1
            fwd_rets_all[fwd] = fwd_ret

        # Primary fwd = 5d
        primary_fwd = 5
        if primary_fwd not in fwd_rets_all:
            primary_fwd = self.fwd_windows[0]
        fwd_ret = fwd_rets_all[primary_fwd]

        # Valid data mask
        valid = ~(np.isnan(alpha_ts) | np.isnan(fwd_ret) |
                  np.isinf(alpha_ts) | np.isinf(fwd_ret))
        a_valid = alpha_ts[valid]
        r_valid = fwd_ret[valid]

        if len(a_valid) < self.min_obs:
            return AlphaResult(
                expression=expression, name=name or expression[:40],
                category=category, generation=generation,
                params=params or {}, rank_ic=0.0, passed=False,
                n_obs=len(a_valid), fwd_window=primary_fwd)

        # Winsorize
        a_clip = self._winsorize(a_valid, 0.01)
        r_clip = self._winsorize(r_valid, 0.01)

        # Rank IC
        try:
            ic, _ = stats.spearmanr(a_clip, r_clip)
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        # IC stability (split-half)
        half = len(a_clip) // 2
        ic_first = 0.0
        ic_second = 0.0
        try:
            if half > 20:
                ic_first, _ = stats.spearmanr(a_clip[:half], r_clip[:half])
                ic_second, _ = stats.spearmanr(a_clip[half:], r_clip[half:])
        except Exception:
            pass
        ic_std = np.std([ic_first, ic_second]) if ic_first != 0 else 0.1
        icir = ic / (ic_std + 1e-9)

        # IC decay
        ic_decay = {}
        for fwd, fr in fwd_rets_all.items():
            v = ~(np.isnan(alpha_ts) | np.isnan(fr))
            if v.sum() > 30:
                try:
                    ic_d, _ = stats.spearmanr(
                        self._winsorize(alpha_ts[v], 0.01),
                        self._winsorize(fr[v], 0.01))
                    ic_decay[fwd] = round(float(ic_d) if not np.isnan(ic_d) else 0.0, 4)
                except Exception:
                    ic_decay[fwd] = 0.0

        # Long-short Sharpe (top/bottom quintile)
        n_q = max(10, len(a_clip) // 5)
        idx_sorted = np.argsort(a_clip)
        long_idx = idx_sorted[-n_q:]
        short_idx = idx_sorted[:n_q]
        long_ret = r_clip[long_idx]
        short_ret = r_clip[short_idx]

        strat_ret = np.concatenate([long_ret, -short_ret])
        sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-9) * np.sqrt(252 / primary_fwd)

        # Max drawdown (on cumulative strategy returns)
        strat_cum = np.cumsum(strat_ret)
        peak = np.maximum.accumulate(strat_cum)
        dd = (strat_cum - peak) / (np.abs(peak) + 1e-9)
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        # Turnover (fraction of quintile that changes daily)
        turnover = 0.0
        if len(a_clip) > 60:
            n_switches = 0
            n_days = 0
            for t in range(60, len(a_clip) - primary_fwd, 5):
                seg_t = alpha_ts[max(0, t-60):t+1]
                seg_next = alpha_ts[max(0, t-60+primary_fwd):t+primary_fwd+1]
                if len(seg_t) < n_q or len(seg_next) < n_q:
                    continue
                top_t = set(np.argsort(seg_t)[-n_q:])
                top_next = set(np.argsort(seg_next)[-n_q:])
                n_switches += len(top_next - top_t)
                n_days += 1
            turnover = n_switches / (n_q * max(n_days, 1))

        # Hit rate
        a_z = (a_clip - np.mean(a_clip)) / (np.std(a_clip) + 1e-9)
        hit_rate = np.mean(np.sign(a_z) == np.sign(r_clip))

        # Correlation check (placeholder — populated by miner)
        corr_with_existing = 0.0

        return AlphaResult(
            expression=expression,
            name=name or expression[:40],
            category=category,
            generation=generation,
            params=params or {},
            rank_ic=round(ic, 4),
            icir=round(icir, 3),
            ic_std=round(ic_std, 4),
            ic_decay=ic_decay,
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 4),
            turnover=round(turnover, 4),
            hit_rate=round(hit_rate, 4),
            n_obs=len(a_clip),
            fwd_window=primary_fwd,
            passed=False,  # set after FDR
        )

    def batch_evaluate(self, tasks: List[Tuple[str, str, str, Dict]],
                       df: pd.DataFrame, verbose: bool = True) -> List[AlphaResult]:
        """
        Batch evaluate multiple alpha expressions.

        Args:
            tasks: List of (expression, name, category, params) tuples
            df: OHLCV DataFrame
            verbose: Print progress

        Returns:
            List of AlphaResult
        """
        results = []
        n = len(tasks)
        for i, (expr, name, cat, params) in enumerate(tasks):
            if verbose and (i + 1) % 50 == 0:
                passed = sum(1 for r in results if r.passed)
                print(f"  进度: {i+1}/{n} | 通过: {passed} | "
                      f"最新: {name[:30]} IC={results[-1].rank_ic if results else 0:+.3f}")
            result = self.evaluate(expr, df, name=name, category=cat,
                                   generation=params.get("_generation", "unknown"),
                                   params=params)
            results.append(result)

        # FDR correction
        results = self._apply_fdr(results)

        if verbose:
            passed = [r for r in results if r.passed]
            print(f"\n📊 完成: {len(results)} 评估 | {len(passed)} 通过 "
                  f"(FDR<{self.fdr_alpha}, ICIR>{self.icir_threshold})")

        return results

    def _winsorize(self, x: np.ndarray, pct: float) -> np.ndarray:
        lo = np.nanpercentile(x, pct * 100)
        hi = np.nanpercentile(x, (1 - pct) * 100)
        return np.clip(x, lo, hi)

    def _apply_fdr(self, results: List[AlphaResult]) -> List[AlphaResult]:
        """Benjamini-Hochberg FDR correction"""
        if len(results) < 2:
            for r in results:
                r.passed = abs(r.icir) > self.icir_threshold
            return results

        # Sort by |ICIR|
        sorted_idx = np.argsort([-abs(r.icir) for r in results])
        m = len(results)

        for rank, idx in enumerate(sorted_idx):
            # BH critical value: (rank/m) * alpha
            bh_crit = ((rank + 1) / m) * self.fdr_alpha
            r = results[idx]
            # Convert ICIR to approximate p-value via |ICIR| heuristic
            # Higher |ICIR| → lower p-value
            r.fdr_p_value = min(1.0, 2 * (1 - stats.norm.cdf(abs(r.icir))))
            r.passed = (r.fdr_p_value < bh_crit and abs(r.icir) > self.icir_threshold)

        return results

    def compute_correlation_matrix(self, alpha_ts_list: List[np.ndarray]) -> np.ndarray:
        """Compute correlation matrix between multiple alpha time series"""
        n = len(alpha_ts_list)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                valid = ~(np.isnan(alpha_ts_list[i]) | np.isnan(alpha_ts_list[j]))
                if valid.sum() > 30:
                    try:
                        c, _ = stats.spearmanr(
                            alpha_ts_list[i][valid], alpha_ts_list[j][valid])
                        corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0
                    except Exception:
                        pass
        return corr


# ═══════════════════════════════════════════════════════════
# Alpha Miner — 3 Discovery Strategies
# ═══════════════════════════════════════════════════════════

class AlphaMiner:
    """Automatic alpha discovery engine"""

    # Genetic operators — mutation targets
    MUTATION_POOL = [
        "ts_delta(close, {w1})", "ts_roc(close, {w1})", "ts_mean(close, {w1})",
        "ts_std(close, {w1})", "ts_zscore(close, {w1})", "ts_rank(close, {w1})",
        "ts_ema(close, {w1})", "close", "volume", "returns",
        "ts_corr(close, volume, {w1})", "ts_sum(close, {w1})",
    ]
    MUTATION_OPS = ["+", "-", "*", "/"]
    MUTATION_UNARY = ["abs", "rank", "sign", "sqrt", "log", "scale", "-"]

    def __init__(self, evaluator: AlphaEvaluator = None,
                 library: AlphaTemplateLibrary = None,
                 df: pd.DataFrame = None,
                 existing_alphas: List[str] = None):
        self.evaluator = evaluator or AlphaEvaluator()
        self.library = library or AlphaTemplateLibrary()
        self.df = df
        self.existing_alphas = existing_alphas or []
        self.existing_alpha_ts: Dict[str, np.ndarray] = {}

    @property
    def data_available(self) -> bool:
        return self.df is not None and len(self.df) > 0

    # ── Strategy 1: Grid Search ──
    def mine_grid(self, df: pd.DataFrame = None,
                  templates: List[str] = None,
                  n_per_template: int = 20,
                  max_total: int = 500,
                  verbose: bool = True) -> List[AlphaResult]:
        """
        Grid search across template parameter space.

        Args:
            df: OHLCV DataFrame
            templates: List of template names to use (None = all)
            n_per_template: Max parameter combinations per template
            max_total: Max total candidates
        """
        df = df or self.df
        if df is None:
            raise ValueError("No data provided")

        # Select templates
        if templates:
            selected = [t for t in self.library.TEMPLATES if t.name in templates]
        else:
            selected = self.library.TEMPLATES

        # Cap total
        n_each = min(n_per_template, max_total // len(selected)) if selected else 0
        if n_each == 0:
            n_each = n_per_template

        if verbose:
            print(f"🔍 Grid Search: {len(selected)} 模板 × ~{n_each} 参数组合")

        # Generate tasks
        tasks = []
        for tmpl in selected:
            param_sets = self.library.generate_grid_params(tmpl, n_each)
            for params in param_sets:
                expr, name = self.library.instantiate(tmpl, params)
                params["_generation"] = "grid"
                params["_template"] = tmpl.name
                tasks.append((expr, name, tmpl.category, params))

        if verbose:
            print(f"  生成 {len(tasks)} 候选Alpha")

        results = self.evaluator.batch_evaluate(tasks, df, verbose=verbose)

        # Redundancy check
        results = self._check_redundancy(results, df)

        return self.filter_and_rank(results)

    # ── Strategy 2: Genetic Programming ──
    def mine_genetic(self, df: pd.DataFrame = None,
                     population_size: int = 200,
                     generations: int = 30,
                     mutation_rate: float = 0.3,
                     crossover_rate: float = 0.5,
                     elite_fraction: float = 0.1,
                     verbose: bool = True) -> List[AlphaResult]:
        """
        Genetic programming for alpha discovery.

        Steps per generation:
          1. Evaluate all individuals
          2. Select top performers (tournament selection)
          3. Crossover: combine two parents
          4. Mutate: randomly modify
          5. Replace worst with offspring
        """
        df = df or self.df
        if df is None:
            raise ValueError("No data provided")

        if verbose:
            print(f"🧬 Genetic Evolution: pop={population_size}, gen={generations}")

        # Initialize population from templates + random expressions
        population = self._init_population(population_size)

        all_results = []
        best_ever = None

        for gen in range(generations):
            if verbose:
                print(f"\n  Gen {gen+1}/{generations} — 评估 {len(population)} 个体...")

            # Evaluate current population
            tasks = [(expr, f"gen{gen}_{i}", "genetic", {"_generation": "genetic", "_gen": gen})
                     for i, expr in enumerate(population)]
            gen_results = self.evaluator.batch_evaluate(tasks, df, verbose=False)
            all_results.extend(gen_results)

            # Sort by |ICIR|
            gen_results.sort(key=lambda r: abs(r.icir), reverse=True)
            gen_results_filtered = [r for r in gen_results if not np.isnan(r.icir)]

            if not gen_results_filtered:
                continue

            best = gen_results_filtered[0]
            if best_ever is None or abs(best.icir) > abs(best_ever.icir):
                best_ever = best

            if verbose:
                top3_icir = [f"{abs(r.icir):.3f}" for r in gen_results_filtered[:3]]
                print(f"    Best ICIR: {top3_icir} | "
                      f"Mean ICIR: {np.mean([abs(r.icir) for r in gen_results_filtered[:20]]):.3f}")

            # Elitism
            n_elite = max(2, int(elite_fraction * population_size))
            elite_exprs = [r.expression for r in gen_results_filtered[:n_elite]]

            # Selection + Crossover + Mutation
            new_population = list(elite_exprs)
            fitnesses = [max(0.01, abs(r.icir)) for r in gen_results_filtered[:min(50, len(gen_results_filtered))]]
            mating_pool = [r.expression for r in gen_results_filtered[:min(50, len(gen_results_filtered))]]

            while len(new_population) < population_size:
                if random.random() < crossover_rate and len(mating_pool) >= 2:
                    # Tournament selection
                    p1 = self._tournament_select(mating_pool, fitnesses)
                    p2 = self._tournament_select(mating_pool, fitnesses)
                    child = self._crossover(p1, p2)
                else:
                    child = random.choice(mating_pool) if mating_pool else random.choice(self.MUTATION_POOL)

                if random.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population[:population_size]

        if verbose and best_ever:
            print(f"\n  🏆 Best ever: {best_ever.name} ICIR={best_ever.icir:.3f}")

        all_results = self._check_redundancy(all_results, df)
        return self.filter_and_rank(all_results, top_n=100)

    def _init_population(self, size: int) -> List[str]:
        """Initialize genetic population from templates + random"""
        pop = []
        # Fill from template instantiations
        for tmpl in self.library.TEMPLATES:
            if len(pop) >= size // 2:
                break
            param_sets = self.library.generate_grid_params(tmpl, 3)
            for params in param_sets[:3]:
                expr, _ = self.library.instantiate(tmpl, params)
                if expr not in pop:
                    pop.append(expr)

        # Fill remaining with random expressions
        while len(pop) < size:
            expr = self._random_expression(max_depth=3)
            if expr not in pop:
                pop.append(expr)

        random.shuffle(pop)
        return pop[:size]

    def _tournament_select(self, pool: List[str], fitnesses: List[float], k: int = 3) -> str:
        """Tournament selection: pick best out of k random candidates"""
        idxs = random.sample(range(len(pool)), min(k, len(pool)))
        best_idx = max(idxs, key=lambda i: fitnesses[i] if i < len(fitnesses) else 0)
        return pool[best_idx]

    def _crossover(self, expr1: str, expr2: str) -> str:
        """Simple subtree crossover: swap random sub-expressions"""
        # For simplicity: swap random parameter values or combine as weighted sum
        if random.random() < 0.5:
            return f"({expr1}) * 0.5 + ({expr2}) * 0.5"
        else:
            return f"({expr1}) / (abs({expr2}) + 1e-9)"

    def _mutate(self, expr: str) -> str:
        """Apply random mutation to an expression"""
        mutation_type = random.choice(['wrap', 'flip', 'scale', 'replace_arg', 'add_noise'])

        if mutation_type == 'wrap':
            op = random.choice(self.MUTATION_UNARY)
            return f"{op}({expr})"
        elif mutation_type == 'flip':
            return f"(-({expr}))"
        elif mutation_type == 'scale':
            scale = random.choice([2, 5, 10, 0.5, 0.2])
            return f"({expr}) * {scale}"
        elif mutation_type == 'replace_arg':
            # Replace a window parameter with a random value
            w = random.choice([3, 5, 7, 10, 14, 20, 30, 50, 60, 90, 120])
            # Simple regex replacement of numeric windows
            import re as _re
            return _re.sub(r'\b\d+\b', str(w), expr, count=1)
        else:
            # Add small noise: multiply by near-1 factor
            noise = random.uniform(0.9, 1.1)
            return f"({expr}) * {noise:.3f}"

    def _random_expression(self, max_depth: int = 3) -> str:
        """Generate a random valid expression from the grammar"""
        if max_depth <= 1:
            return random.choice(["close", "volume", "returns",
                                  "ts_roc(close, 5)", "ts_delta(close, 10)",
                                  "ts_zscore(close, 20)", "ts_std(close, 20)"])

        pattern = random.choice([
            lambda: f"({self._random_expression(max_depth-1)}) {random.choice(self.MUTATION_OPS)} ({self._random_expression(max_depth-1)})",
            lambda: f"{random.choice(self.MUTATION_UNARY)}({self._random_expression(max_depth-1)})",
            lambda: f"ts_delta({random.choice(['close', 'volume', 'returns'])}, {random.choice([3,5,10,20,50])})",
            lambda: f"ts_roc({random.choice(['close', 'volume'])}, {random.choice([5,10,20,30,60])})",
            lambda: f"ts_zscore({random.choice(['close', 'volume'])}, {random.choice([10,20,30,60])})",
            lambda: f"ts_corr(close, volume, {random.choice([10,20,30,60])})",
            lambda: f"ts_mean(close, {random.choice([10,20,50,100])})",
            lambda: f"ts_std(close, {random.choice([10,20,30,50])})",
        ])
        return pattern()

    # ── Strategy 3: Random Exploration ──
    def mine_random(self, df: pd.DataFrame = None,
                    n: int = 500, max_depth: int = 4,
                    verbose: bool = True) -> List[AlphaResult]:
        """Grammar-based random alpha generation"""
        df = df or self.df
        if df is None:
            raise ValueError("No data provided")

        if verbose:
            print(f"🎲 Random Exploration: {n} candidates (max_depth={max_depth})")

        expressions = []
        while len(expressions) < n:
            expr = self._random_expression(max_depth)
            if expr not in expressions:
                expressions.append(expr)

        tasks = [(expr, f"random_{i}", "random", {"_generation": "random"})
                 for i, expr in enumerate(expressions)]

        results = self.evaluator.batch_evaluate(tasks, df, verbose=verbose)
        results = self._check_redundancy(results, df)
        return self.filter_and_rank(results, top_n=min(100, n // 5))

    # ── Post-processing ──
    def _check_redundancy(self, results: List[AlphaResult],
                          df: pd.DataFrame,
                          max_corr: float = 0.7) -> List[AlphaResult]:
        """Compute correlation with existing alphas; penalize redundant ones"""
        if not self.existing_alpha_ts and not results:
            return results

        # Compute alpha time series for valid results
        data = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data[col] = df[col].values.astype(float)

        alpha_ts_list = []
        valid_indices = []
        for i, r in enumerate(results):
            if abs(r.icir) < 0.05:
                continue
            try:
                ts = evaluate_expression(r.expression, data=data)
                if np.isfinite(ts).sum() > 30:
                    alpha_ts_list.append(ts)
                    valid_indices.append(i)
            except Exception:
                pass

        if len(alpha_ts_list) < 2:
            return results

        # Compute correlation matrix
        corr_matrix = self.evaluator.compute_correlation_matrix(alpha_ts_list)

        # For each result, find max correlation with better-ranked alphas
        for rank_i, orig_i in enumerate(valid_indices):
            max_c = 0.0
            for rank_j, orig_j in enumerate(valid_indices):
                if rank_j >= rank_i:  # Only compare vs better-ranked
                    break
                if abs(corr_matrix[rank_i, rank_j]) > max_c:
                    max_c = abs(corr_matrix[rank_i, rank_j])
            results[orig_i].correlation_with_existing = round(max_c, 4)

            # Penalize redundant alphas
            if max_c > max_corr:
                results[orig_i].icir *= 0.5  # Halve ICIR for highly correlated
                results[orig_i].passed = False

        return results

    def filter_and_rank(self, results: List[AlphaResult],
                        top_n: int = 50) -> List[AlphaResult]:
        """Filter, sort, and return top N alphas"""
        # Remove NaN ICIR
        valid = [r for r in results if not np.isnan(r.icir) and abs(r.rank_ic) > 0.01]
        # Sort by |ICIR| descending
        valid.sort(key=lambda r: abs(r.icir), reverse=True)
        # Deduplicate by expression
        seen = set()
        unique = []
        for r in valid:
            if r.expression not in seen:
                seen.add(r.expression)
                unique.append(r)
        return unique[:top_n]


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="🔬 Alpha Miner — 自动Alpha挖掘表达式引擎")
    parser.add_argument("--evaluate", "-e", type=str,
                        help="Evaluate a single alpha expression")
    parser.add_argument("--mine", "-m", action="store_true",
                        help="Run grid search alpha mining")
    parser.add_argument("--evolve", action="store_true",
                        help="Run genetic evolution alpha mining")
    parser.add_argument("--random", action="store_true",
                        help="Run random exploration alpha mining")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of candidates (default: 500)")
    parser.add_argument("--generations", type=int, default=30,
                        help="Genetic generations (default: 30)")
    parser.add_argument("--population", type=int, default=200,
                        help="Genetic population size (default: 200)")
    parser.add_argument("--symbol", "-s", type=str, default="BTC/USDT",
                        help="Symbol to fetch data for (default: BTC/USDT)")
    parser.add_argument("--lookback", type=int, default=500,
                        help="Lookback days (default: 500)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List saved alphas")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N when listing (default: 20)")
    parser.add_argument("--list-templates", action="store_true",
                        help="List available alpha templates")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results (default: True)")
    parser.add_argument("--fwd", type=int, default=5,
                        help="Forward return window for quick IC (default: 5)")

    args = parser.parse_args()

    # ── List templates ──
    if args.list_templates:
        print(f"\n📚 Alpha Template Library ({len(AlphaTemplateLibrary.TEMPLATES)} templates)\n")
        for cat in AlphaTemplateLibrary.CATEGORIES:
            tmpls = AlphaTemplateLibrary.get_by_category(cat)
            print(f"  [{cat}] ({len(tmpls)} templates)")
            for t in tmpls[:3]:
                print(f"    {t.name:25s} | {t.expression:50s} | {t.description}")
            if len(tmpls) > 3:
                print(f"    ... +{len(tmpls)-3} more")
            print()
        return

    # ── List saved alphas ──
    if args.list:
        store = AlphaStore()
        files = store.list_saved()
        if not files:
            print("📭 No saved alphas found. Run --mine first!")
            return
        print(f"\n📦 Saved Alpha Sets ({len(files)})\n")
        for f in files[:10]:
            print(f"  {f['name']:40s} | {f['n_alphas']:4d} alphas | {f['saved_at'][:19]}")
        # Show top from latest
        alphas = store.get_top(args.top)
        if alphas:
            print(f"\n🏆 Top {len(alphas)} Alphas (by |ICIR|):\n")
            print(f"  {'':3s} {'Expression':45s} {'IC':>7s} {'ICIR':>7s} {'Sh':>7s} {'Cat':15s}")
            print(f"  {'-'*3} {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*15}")
            for i, a in enumerate(alphas):
                check = "✅" if a.passed else "❌"
                print(f"  {check} {a.expression[:43]:45s} {a.rank_ic:+.3f} "
                      f"{a.icir:+.2f}  {a.sharpe:+.2f}  {a.category:15s}")
        return

    # ── Evaluate single expression ──
    if args.evaluate:
        expr = args.evaluate
        print(f"\n🔬 Evaluating: {expr}\n")

        # Fetch data
        df = _fetch_data(args.symbol, args.lookback)
        if df is None or len(df) < 100:
            print(f"❌ Failed to fetch data for {args.symbol}")
            return

        print(f"   Data: {args.symbol} {len(df)} days ({df.index[0]} → {df.index[-1]})")

        # Quick IC
        ic5 = quick_ic(expr, df, 5)
        ic1 = quick_ic(expr, df, 1)
        ic10 = quick_ic(expr, df, 10)
        print(f"   Rank IC: 1d={ic1:+.4f}  5d={ic5:+.4f}  10d={ic10:+.4f}")

        # Full evaluation
        evaluator = AlphaEvaluator()
        result = evaluator.evaluate(expr, df, name="manual", category="custom")
        print(f"\n   📊 Full Evaluation (fwd={result.fwd_window}d):")
        print(f"   IC={result.rank_ic:+.4f}  ICIR={result.icir:+.3f}  "
              f"Sharpe={result.sharpe:+.3f}  Turnover={result.turnover:.3f}")
        print(f"   IC Decay: {result.ic_decay}")
        print(f"   N={result.n_obs}  Hit Rate={result.hit_rate:.2%}  "
              f"Passed={'✅' if result.passed else '❌'}")

        return

    # ── Fetch data for mining ──
    df = None
    if args.mine or args.evolve or args.random:
        print(f"📡 Fetching {args.symbol} data ({args.lookback}d)...")
        df = _fetch_data(args.symbol, args.lookback)
        if df is None or len(df) < 100:
            print(f"❌ Failed to fetch data for {args.symbol}")
            return
        print(f"   Got {len(df)} days ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})\n")

    # ── Grid Search Mining ──
    if args.mine:
        evaluator = AlphaEvaluator()
        miner = AlphaMiner(evaluator=evaluator, df=df)
        results = miner.mine_grid(df=df, n_per_template=15, max_total=args.n, verbose=True)

        print(f"\n🏆 Top Alphas (Grid Search):\n")
        _print_results(results, args.top)

        if args.save and results:
            store = AlphaStore()
            path = store.save(results)
            print(f"\n💾 Saved {len(results)} alphas to {path}")

    # ── Genetic Evolution ──
    if args.evolve:
        evaluator = AlphaEvaluator()
        miner = AlphaMiner(evaluator=evaluator, df=df)
        results = miner.mine_genetic(
            df=df, population_size=args.population,
            generations=args.generations, verbose=True)

        print(f"\n🏆 Top Alphas (Genetic Evolution):\n")
        _print_results(results, args.top)

        if args.save and results:
            store = AlphaStore()
            path = store.save(results)
            print(f"\n💾 Saved {len(results)} alphas to {path}")

    # ── Random Exploration ──
    if args.random:
        evaluator = AlphaEvaluator()
        miner = AlphaMiner(evaluator=evaluator, df=df)
        results = miner.mine_random(df=df, n=args.n, verbose=True)

        print(f"\n🏆 Top Alphas (Random Exploration):\n")
        _print_results(results, args.top)

        if args.save and results:
            store = AlphaStore()
            path = store.save(results)
            print(f"\n💾 Saved {len(results)} alphas to {path}")

    # If no action specified, show help
    if not any([args.evaluate, args.mine, args.evolve, args.random,
                args.list, args.list_templates]):
        parser.print_help()
        print(f"\n📚 Available variables: {sorted(VARIABLES)}")
        print(f"🔧 Available functions: {sorted(TS_FUNCTIONS.keys())}")
        print(f"📐 Available unary ops: {UNARY_OPS}")
        print(f"\nExample: python3 alpha_miner.py -e 'ts_delta(close,5)/ts_std(close,20)'")


def _fetch_data(symbol: str, lookback: int = 500) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from ccxt or yfinance"""
    try:
        import ccxt
        exchange = ccxt.binance()
        since = exchange.parse8601(
            (pd.Timestamp.now() - pd.Timedelta(days=lookback)).strftime('%Y-%m-%dT00:00:00Z'))
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=lookback)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️  ccxt fetch failed: {e}")
        # Fallback: try yfinance
        try:
            import yfinance as yf
            ticker = symbol.replace("/", "-")
            df = yf.download(ticker, period=f"{lookback}d", progress=False)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            return None


def _print_results(results: List[AlphaResult], top_n: int = 20):
    """Pretty print alpha results"""
    print(f"  {'':3s} {'Expression':50s} {'IC':>7s} {'ICIR':>7s} {'Sh':>7s} {'TO':>6s} {'Cat':12s} {'Gen':8s}")
    print(f"  {'-'*3} {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*12} {'-'*8}")
    for i, a in enumerate(results[:top_n]):
        check = "✅" if a.passed else "❌"
        expr_short = a.expression[:48]
        print(f"  {check} {expr_short:50s} {a.rank_ic:+.3f} {a.icir:+.2f}  "
              f"{a.sharpe:+.2f}  {a.turnover:.3f} {a.category:12s} {a.generation:8s}")


if __name__ == "__main__":
    main()
