"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑；
 9. 归因：中间配方构造规则（按比例补齐、最大余数、参考为零时均分）、
    Shapley 值与直接按定义穷举一致、效率性（贡献和 = 概率差）、
    非法输入不出报告、API 分支、归因页含迟到结果丢弃逻辑。
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import z3

from app.attribution import (
    INTERNAL_ERROR,
    INVALID as ATTR_INVALID,
    OK as ATTR_OK,
    compute_attribution,
    fill_intermediate,
    shapley_values,
    validate_pair,
)
from app.model import (
    FEATURE_NAMES,
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性
    opt, g, num_changed, total_change = _build_optimize(forest, inp)
    opt.add(num_changed <= k - 1)
    check("z3 反证：更少种数无解", opt.check() == z3.unsat)
    opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
    opt2.add(num_changed2 == k, total_change2 <= m - 1)
    check("z3 反证：同种数更小改动量无解", opt2.check() == z3.unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 归因 ==")
    from app.attribution import fill_intermediate as _fill

    # 9a. 中间配方构造规则
    orig_f = [250, 250, 350, 150]
    adj_f = [390, 250, 210, 150]
    ok_rule = True
    detail = ""
    for mask in range(1 << 4):
        z = _fill(orig_f, adj_f, mask)
        if any(v < 0 or not isinstance(v, int) for v in z) or sum(z) != TOTAL_GRAMS:
            ok_rule = False; detail = f"mask={mask} z={z}"
            break
        for i in range(4):
            if mask & (1 << i) and z[i] != adj_f[i]:
                ok_rule = False; detail = f"mask={mask} 固定项未取调整后用量"
    check("16 份中间配方非负、总重 1000、固定项取调整后用量", ok_rule, detail)
    check("空集子集 = 调整前配方", _fill(orig_f, adj_f, 0) == orig_f)
    check("全集子集 = 调整后配方", _fill(orig_f, adj_f, 0b1111) == adj_f)

    # 比例补齐：子集 {0} 时其余按 250:350:150 补 610 克 → 203/285/122（最大余数法）
    check("按原配方比例补齐（最大余数法）", _fill(orig_f, adj_f, 0b0001) == [390, 203, 285, 122],
          str(_fill(orig_f, adj_f, 0b0001)))
    # 参考量为 0 的原料不得余克：参考 [0,1,1] 分 1 克 → 精确 0/0.5/0.5，余克给下标 1
    check("参考量为 0 的原料不会分到余克",
          _fill([0, 1, 1, 998], [0, 0, 0, 999], 0b1000) == [0, 1, 0, 999],
          str(_fill([0, 1, 1, 998], [0, 0, 0, 999], 0b1000)))
    # 余量并列时下标靠前者优先：参考 [1,1,2] 分 2 克 → 精确 0.5/0.5/1，余克给下标 0
    check("最大余数并列时下标靠前者优先",
          _fill([1, 1, 2, 996], [0, 0, 0, 998], 0b1000) == [1, 0, 1, 998],
          str(_fill([1, 1, 2, 996], [0, 0, 0, 998], 0b1000)))
    # 参考剩余量全为零：补集均分剩余量（1000/3 → 334,333,333，余克按下标从小到大）
    check("参考剩余量全为零时均分并按下标分配余克",
          _fill([0, 0, 0, 1000], [333, 333, 334, 0], 0b1000) == [334, 333, 333, 0],
          str(_fill([0, 0, 0, 1000], [333, 333, 334, 0], 0b1000)))
    check("参考剩余量全为零且可整除时均分",
          _fill([0, 0, 0, 1000], [100, 100, 100, 700], 0b1000) == [100, 100, 100, 700],
          str(_fill([0, 0, 0, 1000], [100, 100, 100, 700], 0b1000)))

    # 9b. Shapley 值与按定义穷举一致
    vals = [float(forest_proba_rational(forest, _fill(orig_f, adj_f, m)))
            for m in range(16)]
    phi_mod = shapley_values(vals)
    phi_ref = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        acc = 0.0
        for k in range(4):
            for S in itertools.combinations(others, k):
                mask = sum(1 << j for j in S)
                w = math.factorial(k) * math.factorial(4 - k - 1) / math.factorial(4)
                acc += w * (vals[mask | (1 << i)] - vals[mask])
        phi_ref.append(acc)
    check("Shapley 值按定义穷举一致", all(abs(a - b) < 1e-12 for a, b in zip(phi_mod, phi_ref)))

    # 9c. 完整归因：效率性 + 报告字段 + 原分类器复算一致
    attr = compute_attribution(clf, orig_f, adj_f)
    check("归因返回 ok", attr.status == ATTR_OK, attr.reason)
    check("两端概率与原分类器一致",
          abs(attr.original_probability - predict_proba_qualified(clf, orig_f)) < 1e-12
          and abs(attr.adjusted_probability - predict_proba_qualified(clf, adj_f)) < 1e-12)
    check("贡献和 = 实际概率差（效率性，误差 < 1e-12）",
          abs(attr.contribution_sum - attr.difference) < 1e-12,
          f"sum={attr.contribution_sum}, diff={attr.difference}")
    check("误差字段 = 贡献和 − 概率差",
          abs(attr.error - (attr.contribution_sum - attr.difference)) < 1e-15)
    check("贡献项齐全且含名称与正负号",
          [c["name"] for c in attr.contributions] == FEATURE_NAMES
          and any(c["contribution"] > 0 for c in attr.contributions))
    pri = attr.priority_index
    check("优先复核项为 |贡献| 最大者",
          abs(attr.contributions[pri]["contribution"]) ==
          max(abs(c["contribution"]) for c in attr.contributions))
    check("16 份中间配方全部列出", len(attr.intermediates) == 16)

    # 9c-2. 随机 60 组配方：构造规则 + 效率性 + 与原分类器一致
    rng_attr = np.random.default_rng(20241118)
    prop_ok = True
    prop_detail = ""
    for _ in range(60):
        o = rng_attr.multinomial(TOTAL_GRAMS, [0.3, 0.25, 0.3, 0.15]).tolist()
        a = rng_attr.multinomial(TOTAL_GRAMS, [0.35, 0.2, 0.25, 0.2]).tolist()
        for m in range(16):
            z = _fill(o, a, m)
            if sum(z) != TOTAL_GRAMS or any(v < 0 for v in z):
                prop_ok = False; prop_detail = f"o={o} a={a} m={m} z={z}"
        ra = compute_attribution(clf, o, a)
        if (ra.status != ATTR_OK or float(np.sum([c["contribution"] for c in ra.contributions]))
                != ra.contribution_sum or abs(ra.contribution_sum - ra.difference) > 1e-12
                or abs(ra.original_probability - predict_proba_qualified(clf, o)) > 1e-12
                or abs(ra.adjusted_probability - predict_proba_qualified(clf, a)) > 1e-12):
            prop_ok = False; prop_detail = f"o={o} a={a}"
        if not prop_ok:
            break
    check("随机 60 组配方：中间配方合法且效率性/两端概率成立", prop_ok, prop_detail)

    # 9c-3. 任意参考配方（含大量零项）也满足构造规则
    zero_ok = True
    for _ in range(60):
        o = rng_attr.multinomial(TOTAL_GRAMS, [0.7, 0.1, 0.1, 0.1]).tolist()
        a = rng_attr.multinomial(TOTAL_GRAMS, [0.1, 0.1, 0.1, 0.7]).tolist()
        for m in range(16):
            z = _fill(o, a, m)
            if sum(z) != TOTAL_GRAMS or any(v < 0 for v in z):
                zero_ok = False
    check("随机 60 组偏斜参考配方：中间配方非负且总重 1000", zero_ok)

    # 9d. 调整前后互换：补齐基准随之改变，故贡献不要求取反；仍需满足效率性。
    rev = compute_attribution(clf, adj_f, orig_f)
    check("调整前后互换后仍返回 ok 且效率性成立",
          rev.status == ATTR_OK and abs(rev.contribution_sum - rev.difference) < 1e-12)
    check("调整前后互换后优先复核项为 |贡献| 最大者",
          abs(rev.contributions[rev.priority_index]["contribution"]) ==
          max(abs(c["contribution"]) for c in rev.contributions))

    # 9e. 非法输入：不出报告（状态 invalid，且序列化后不含报告字段）
    bad_pairs = [
        ([250, 250, 350, 149], adj_f),          # 总重 999
        (orig_f, [390, 250, 210, 151]),         # 总重 1001
        ([250, 250, 350], adj_f),               # 长度 3
        ([250, 250, 350, 150.0], adj_f),        # 浮点
        ([250, 250, 350, True], adj_f),         # 布尔
        ([250, 250, 350, -150], adj_f),         # 负数
        ([250, 250, 350, 1150], adj_f),         # 单项 > 1000
        (None, adj_f),
    ]
    bad_ok = True
    for o, a in bad_pairs:
        r = compute_attribution(clf, o if o is not None else [], a)
        d = r.to_dict()
        if r.status != ATTR_INVALID or set(d.keys()) != {"status", "reason"}:
            bad_ok = False
            print("   failed pair:", o, a, "->", r.status, d.keys())
    check("非法输入均判 invalid 且不携带报告字段", bad_ok)

    # 9f. 计算异常与非法输入区分（注入异常）
    import app.attribution as at
    orig_predict = at.predict_proba_qualified
    at.predict_proba_qualified = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected"))
    try:
        r_boom = compute_attribution(clf, orig_f, adj_f)
    finally:
        at.predict_proba_qualified = orig_predict
    check("计算异常 → internal_error（非 invalid）", r_boom.status == INTERNAL_ERROR
          and set(r_boom.to_dict().keys()) == {"status", "reason"})

    # 9g. API 分支
    r = client.post("/api/attribution", json={"original": orig_f, "adjusted": adj_f})
    aj = r.json()
    check("POST /api/attribution 正常分支", r.status_code == 200 and aj["status"] == "ok"
          and len(aj["contributions"]) == 4 and abs(aj["error"]) < 1e-12)
    r = client.post("/api/attribution", json={"original": [1, 2, 3, 4], "adjusted": adj_f})
    check("POST /api/attribution 非法分支不出报告",
          r.json()["status"] == "invalid" and "contributions" not in r.json())
    r = client.post("/api/attribution", json={})
    check("POST /api/attribution 缺字段判 invalid", r.json()["status"] == "invalid")
    r = client.post("/api/attribution", json={"original": [250.0, 250.0, 350.0, 150.0],
                                              "adjusted": adj_f})
    check("POST /api/attribution 整数值浮点被接受", r.json()["status"] == "ok")
    r = client.get("/attribution")
    check("GET /attribution 返回归因页", r.status_code == 200 and "计算归因" in r.text)
    ah = client.get("/attribution").text
    check("归因页含迟到结果丢弃逻辑", "my !== epoch" in ah and "abort" in ah)
    check("归因页不渲染非 ok 报告", "d.status !== \"ok\"" in ah)
    r = client.get("/api/attribution/sample")
    check("GET /api/attribution/sample",
          validate_pair(r.json()["original"], r.json()["adjusted"]) == [])
    check("归因页导航回原试配", 'href="/"' in ah)

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
