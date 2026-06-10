"""
answer_extraction.py
====================
Robust answer extraction and normalization for math evaluation.

Aligned with MetaMath official eval protocol:
  - eval_gsm8k.py : split on "The answer is: ", extract number via regex + Fraction
  - eval_math.py  : split on "The answer is: ", symbolic comparison via sympy
  - Gold for GSM8K: #### pattern
  - Gold for MATH : \boxed{} in solution

Source: https://github.com/meta-math/MetaMath
"""

import re
import math
from fractions import Fraction
from typing import Optional

try:
    from sympy import sympify, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


# 
# Patterns
# 

FRAC_PATTERN       = re.compile(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}")
PLAIN_FRAC_PATTERN = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
PCT_PATTERN        = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")
UNITS_PATTERN      = re.compile(
    r"\b(meters?|km|miles?|feet|ft|cm|mm|kg|g|lbs?|hours?|minutes?|seconds?|"
    r"dollars?|\$|euros?|years?|days?|weeks?|people|students?|apples?|oranges?|"
    r"items?|units?|pieces?|times?)\b",
    re.IGNORECASE,
)


# 
# Boxed extraction
# 

def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract content from \boxed{...}, handles nested \boxed{\frac{1}{2}}."""
    if not text:
        return None
    m = re.search(r"\\boxed\s*\{", text)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i].strip()
    return None


def _last_boxed(text: str) -> Optional[str]:
    """
    Extract the LAST \boxed{} in text.
    Avoids capturing intermediate boxed steps — takes the final answer only.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx + len("\\boxed{") : i].strip()
    return None


# 
# Normalization
# 

def _fmt(val: float) -> str:
    if math.isinf(val) or math.isnan(val):
        return str(val)
    if val == int(val) and abs(val) < 1e12:
        return str(int(val))
    return f"{val:.6g}"


def normalize_latex_frac(expr: str) -> str:
    def replace_frac(m):
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den == 0:
                return m.group(0)
            return _fmt(num / den)
        except Exception:
            return m.group(0)
    return FRAC_PATTERN.sub(replace_frac, expr)


def to_canonical_number(expr) -> Optional[str]:
    """
    Convert any math expression to a canonical number string.
    Examples:
        "\\frac{1}{2}" → "0.5"
        "3/4"          → "0.75"
        "75%"          → "0.75"
        "1,000"        → "1000"
        "-7"           → "-7"
    """
    if not expr or not isinstance(expr, str):
        return None

    expr = expr.strip()
    expr = expr.replace("$", "").replace(",", "")

    pct_m = PCT_PATTERN.match(expr.strip())
    if pct_m:
        try:
            return _fmt(float(pct_m.group(1)) / 100)
        except Exception:
            pass

    expr = expr.replace(" ", "")
    expr = normalize_latex_frac(expr)
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = expr.replace("{", "").replace("}", "")
    expr = UNITS_PATTERN.sub("", expr).strip()

    fm = PLAIN_FRAC_PATTERN.match(expr)
    if fm:
        try:
            n, d = int(fm.group(1)), int(fm.group(2))
            if d == 0:
                return None
            return _fmt(n / d)
        except Exception:
            pass

    try:
        val = float(expr)
        if math.isinf(val) or math.isnan(val):
            return None
        return _fmt(val)
    except ValueError:
        pass

    if HAS_SYMPY:
        try:
            val = float(sympify(expr))
            if math.isinf(val) or math.isnan(val):
                return None
            return _fmt(val)
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────
# Dataset-specific answer extractors
# Aligned with MetaMath official eval scripts
# ─────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract answer from model output for GSM8K.

    Aligned with MetaMath eval_gsm8k.py:
      - Priority 1: split on 'The answer is: ', extract number via regex
      - Handles fractions via Python Fraction()
      - Fallback: #### pattern (in case model uses training format)

    Source: github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    """
    if not text:
        return None

    # Priority 1 — "The answer is: " (MetaMath official pattern)
    parts = text.split("The answer is: ")
    if len(parts) > 1:
        extract = parts[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract)
        if match:
            num_str = match.group().replace(",", "")
            if "/" in num_str:
                try:
                    frac = Fraction(num_str)
                    result = float(frac.numerator / frac.denominator)
                    return None if result == float("inf") else str(round(result, 6))
                except Exception:
                    return None
            try:
                val = float(num_str)
                return None if val == float("inf") else str(round(val))
            except Exception:
                return None

    # Fallback — #### pattern
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    return None


def extract_math_answer(text: str) -> Optional[str]:
    """
    Extract answer from model output for MATH dataset.

    Aligned with MetaMath eval_math.py:
      - Priority 1: split on 'The answer is: ', extract boxed if present
      - Fallback: last \boxed{} in full text (avoids intermediate boxed steps)

    Source: github.com/meta-math/MetaMath/blob/main/eval_math.py
    """
    if not text:
        return None

    # Priority 1 — "The answer is: " (MetaMath official pattern)
    parts = text.split("The answer is: ")
    if len(parts) > 1:
        ans = parts[-1]
        # Cut at first newline or ". \n"
        extract = ans.split(".\n")[0].strip().rstrip(".").strip()
        if extract:
            # If \boxed{} present in extract, unwrap it
            boxed = extract_boxed_answer(extract)
            return boxed if boxed else extract

    # Fallback — last \boxed{} in full text
    return _last_boxed(text)


# ─────────────────────────────────────────────
# Legacy extractor — kept for backward compat
# Used by metrics.py if called directly
# ─────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """
    Generic extractor. Used as fallback.

    Priority order aligned with MetaMath:
      1. "The answer is: " pattern
      2. Last \boxed{} in text
      3. to_canonical_number on last 5 lines   ← FIX: was on full text
      4. Last number fallback
    """
    if not text:
        return None

    text = text.strip()

    # 1. "The answer is: " — MetaMath primary pattern
    parts = text.split("The answer is: ")
    if len(parts) > 1:
        candidate = parts[-1].split(".\n")[0].strip().rstrip(".").strip()
        if candidate:
            boxed = extract_boxed_answer(candidate)
            return boxed if boxed else candidate

    # 2. Last \boxed{} — avoids intermediate steps
    boxed = _last_boxed(text)
    if boxed:
        return boxed

    # 3. to_canonical_number on last 5 lines only
    #    FIX: was on full text :caused false positives on long solutions
    tail = "\n".join([l for l in text.strip().split("\n") if l.strip()][-5:])
    canonical = to_canonical_number(tail)
    if canonical is not None:
        return canonical

    # 4. Last number fallback
    clean = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", clean)
    if numbers:
        return numbers[-1]

    return None


# 
# Comparison
# 

def answers_are_equal(
    pred: Optional[str],
    gold: Optional[str],
    tol: float = 1e-3,    # FIX: was 1e-4 — too strict for 0.333 vs 1/3
) -> bool:
    """
    Check if two answers are equal.

    Strategy:
      1. Exact string match
      2. Numeric comparison with tol=1e-3
         (1e-4 was too strict: |0.333 - 0.333333| = 3.3e-4 > 1e-4)
      3. Sympy symbolic equality for \pi, \sqrt{2}, polynomials
      4. Normalized string fallback
    """
    if pred is None or gold is None:
        return False

    pred = pred.strip()
    gold = gold.strip()

    # 1. Exact match
    if pred.lower() == gold.lower():
        return True

    # 2. Numeric comparison
    pred_num = to_canonical_number(pred)
    gold_num = to_canonical_number(gold)

    if pred_num is not None and gold_num is not None:
        try:
            return abs(float(pred_num) - float(gold_num)) <= tol
        except Exception:
            return pred_num == gold_num

    # 3. Sympy symbolic — handles \pi, \sqrt{2}, polynomials
    #    FIX: was missing — MATH non-numeric answers fell through to _clean()
    if HAS_SYMPY:
        try:
            if simplify(sympify(pred) - sympify(gold)) == 0:
                return True
        except Exception:
            pass

    # 4. Normalized string fallback
    return _clean(pred) == _clean(gold)


def _clean(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;!?]+$", "", s)
    return s


# 
# Dataset-specific gold extractors
# 

def extract_gsm8k_gold(answer_text: str) -> Optional[str]:
    """GSM8K gold format: '... #### 42'"""
    match = re.search(r"####\s*(-?[\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_math_gold(answer_text: str) -> Optional[str]:
    """MATH gold format: answer is in \\boxed{} in the solution text."""
    return extract_boxed_answer(answer_text) or answer_text.strip()