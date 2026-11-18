"""配方归因：用原随机森林的 Shapley 值解释“合格概率”的变化。

语义（与页面说明一致）：

* 输入为调整前 / 调整后两份 4 料整数配方，各自总重 1000 克。
* 对 4 种原料的每个子集 S：S 内原料取**调整后**用量；其余原料按**调整前**
  配方的比例补齐剩余克数。
  - 设固定项合计 fixed，剩余量 remaining = 1000 - fixed；
    补集 C = 不在 S 中的原料，参考量 ref_i = 原配方克数。
  - 若 sum(ref_C) > 0：按 ref 比例分配 remaining，先取精确下取整，
    再按“小数余量从大到小、同余量按下标从小到大”的固定规则把余克分给
    参考量 > 0 的原料（最大余数法；参考量为 0 的原料恒得 0 克），
    因此每份中间配方都是非负整数且总重恰为 1000。
  - 若 sum(ref_C) == 0（参考剩余量全为零）：把 remaining 均分给 C，
    不能整除的余克按下标从小到大逐一 +1。
* 用**原随机森林**对每份中间配方的 float32 比例推理，得到 v(S)。
* Shapley 值 φ_i 为 4 料的有符号贡献；理想情况下 sum(φ) = v(N) - v(∅)，
  报告同时给出“贡献和 - 实际概率差”的浮点误差。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

from .model import (
    FEATURE_NAMES,
    N_FEATURES,
    TOTAL_GRAMS,
    predict_proba_qualified,
)

OK = "ok"
INVALID = "invalid"
INTERNAL_ERROR = "internal_error"


@dataclass
class AttributionResult:
    status: str
    reason: str = ""
    original: list[int] | None = None
    adjusted: list[int] | None = None
    original_probability: float | None = None
    adjusted_probability: float | None = None
    difference: float | None = None
    contributions: list[dict] | None = None
    contribution_sum: float | None = None
    error: float | None = None
    priority_index: int | None = None
    priority_name: str | None = None
    intermediates: list[dict] | None = None
    compute_ms: float = 0.0

    def to_dict(self) -> dict:
        # 非法输入不携带任何报告字段（页面据此绝不渲染报告）。
        if self.status != OK:
            return {"status": self.status, "reason": self.reason}
        return {
            "status": self.status,
            "original": self.original,
            "adjusted": self.adjusted,
            "original_probability": self.original_probability,
            "adjusted_probability": self.adjusted_probability,
            "difference": self.difference,
            "contributions": self.contributions,
            "contribution_sum": self.contribution_sum,
            "error": self.error,
            "priority_index": self.priority_index,
            "priority_name": self.priority_name,
            "intermediates": self.intermediates,
            "compute_ms": round(self.compute_ms, 3),
        }


def validate_pair(original, adjusted) -> list[str]:
    errs = []
    for name, arr in (("original", original), ("adjusted", adjusted)):
        if len(arr) != N_FEATURES or any(
            (not isinstance(v, int)) or isinstance(v, bool) for v in arr
        ):
            errs.append(f"{name} 必须是 {N_FEATURES} 个整数")
            continue
        if any(v < 0 or v > TOTAL_GRAMS for v in arr):
            errs.append(f"{name} 每项克数必须在 0–{TOTAL_GRAMS} 之间")
        elif sum(arr) != TOTAL_GRAMS:
            errs.append(f"{name} 总克数必须等于 {TOTAL_GRAMS}（当前 {sum(arr)}）")
    return errs


def fill_intermediate(original: list[int], adjusted: list[int], subset: int) -> list[int]:
    """按固定规则构造中间配方；subset 为 4 位掩码（置位 = 固定调整后用量）。

    调用前须通过 validate_pair（保证两份配方均为非负整数且总重 1000）。
    """
    fixed = [i for i in range(N_FEATURES) if subset & (1 << i)]
    comp = [i for i in range(N_FEATURES) if not (subset & (1 << i))]
    z = [0] * N_FEATURES
    fixed_sum = 0
    for i in fixed:
        z[i] = adjusted[i]
        fixed_sum += adjusted[i]
    remaining = TOTAL_GRAMS - fixed_sum
    if remaining < 0:  # 校验后不可能发生；防御性处理
        raise ValueError("固定项合计超过总重")
    if not comp:
        if remaining != 0:
            raise ValueError("补集为空时剩余量必须为零")
        return z

    ref = [original[i] for i in comp]
    ref_total = sum(ref)
    if ref_total > 0:
        # 精确有理数比例 + 最大余数法；余克只分给参考量 > 0 的原料
        #（其小数余量之和 < 正参考项数，故余克总能分完），参考量为 0 的原料恒得 0。
        raw = [Fraction(remaining * w, ref_total) for w in ref]
        floors = [int(r) for r in raw]  # 非负 Fraction 的 int() 即下取整
        leftover = remaining - sum(floors)
        order = sorted(
            (k for k in range(len(comp)) if ref[k] > 0),
            key=lambda k: (-(raw[k] - floors[k]), k),
        )
        for k in order[:leftover]:
            floors[k] += 1
        for k, i in enumerate(comp):
            z[i] = floors[k]
    else:
        # 参考剩余量全为零：均分剩余量，取整余克按下标从小到大分配。
        base, rem = divmod(remaining, len(comp))
        for k, i in enumerate(comp):
            z[i] = base + (1 if k < rem else 0)
    return z


def shapley_values(values: list[float]) -> list[float]:
    """由全部 2^4 个子集值计算 4 料 Shapley 值（标准组合权重）。"""
    n = N_FEATURES
    denom = factorial(n)
    phi = []
    for i in range(n):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            s = bin(mask).count("1")
            weight = Fraction(factorial(s) * factorial(n - s - 1), denom)
            total += float(weight) * (values[mask | (1 << i)] - values[mask])
        phi.append(total)
    return phi


def compute_attribution(clf, original: list[int], adjusted: list[int]) -> AttributionResult:
    """校验 + 16 份中间配方推理 + Shapley 归因。"""
    t0 = time.perf_counter()
    errs = validate_pair(original, adjusted)
    if errs:
        return AttributionResult(status=INVALID, reason="；".join(errs))
    try:
        masks = list(range(1 << N_FEATURES))
        formulas = [fill_intermediate(original, adjusted, m) for m in masks]
        # 原随机森林、float32 比例推理（与试配求解的概率复核同一路径）。
        values = [predict_proba_qualified(clf, g) for g in formulas]
        phi = shapley_values(values)
    except Exception as exc:  # 计算异常与“非法输入”严格区分
        return AttributionResult(
            status=INTERNAL_ERROR,
            reason=f"归因计算异常：{exc}",
            compute_ms=(time.perf_counter() - t0) * 1000,
        )

    p0, p1 = values[0], values[-1]
    difference = p1 - p0
    contribution_sum = sum(phi)
    priority_index = max(
        range(N_FEATURES), key=lambda i: (abs(phi[i]), -i)
    )
    contributions = [
        {
            "index": i,
            "name": FEATURE_NAMES[i],
            "original": original[i],
            "adjusted": adjusted[i],
            "delta": adjusted[i] - original[i],
            "contribution": phi[i],
        }
        for i in range(N_FEATURES)
    ]
    intermediates = [
        {
            "subset": [i for i in range(N_FEATURES) if m & (1 << i)],
            "grams": formulas[m],
            "probability": values[m],
        }
        for m in masks
    ]
    return AttributionResult(
        status=OK,
        original=list(original),
        adjusted=list(adjusted),
        original_probability=p0,
        adjusted_probability=p1,
        difference=difference,
        contributions=contributions,
        contribution_sum=contribution_sum,
        error=contribution_sum - difference,
        priority_index=priority_index,
        priority_name=FEATURE_NAMES[priority_index],
        intermediates=intermediates,
        compute_ms=(time.perf_counter() - t0) * 1000,
    )
