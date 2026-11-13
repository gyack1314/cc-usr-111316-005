"""参考分布：独立性耦合、共同分位（comonotone）耦合、输入已固定的真实联合表。

三者都只是参照系，函数返回时显式标注其假设；绝不参与全可行域界的计算。
"""
from fractions import Fraction

from .rational import parse_rational

ZERO = Fraction(0)
ONE = Fraction(1)


def independence_joint(model):
    """乘积测度：p(combo) = ∏ 边际概率。"""
    dims = model.dimensions
    atoms = []
    for flat, combo in enumerate(model.grid):
        w = ONE
        for k, st_idx in enumerate(combo):
            w *= dims[k].states[st_idx].probability
        if w > 0:
            atoms.append((flat, w))
    return tuple(atoms)


def comonotone_joint(model):
    """共同分位耦合：各维同一分位水平的状态同现（离散分布的上 comonotone 耦合）。

    用累积概率区间的交集切分 [0,1]，每个区间给出一个原子，权重为区间长度。
    """
    dims = model.dimensions
    # 每维的累积断点与对应状态
    breaks = []
    for d in dims:
        pts, cum = [], ZERO
        for st in d.states:
            lo, hi = cum, cum + st.probability
            pts.append((lo, hi))
            cum = hi
        breaks.append(pts)
    boundaries = sorted({ZERO, ONE} | {p for pts in breaks for pair in pts for p in pair})
    atoms = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        mid = (a + b) / 2
        combo = tuple(next(k for k, (lo, hi) in enumerate(pts) if lo <= mid < hi
                           or (mid == ONE and hi == ONE))
                      for pts in breaks)
        flat = model.flat_index(combo)
        w = b - a
        if w > 0:
            atoms.append((flat, w))
    # 同格点合并
    merged = {}
    for flat, w in atoms:
        merged[flat] = merged.get(flat, ZERO) + w
    return tuple(sorted(merged.items()))


def fixed_joint_check(model, fixed):
    """核验调用方给定的真实联合表：形状、非负、归一化、所有边际与已登记约束。

    fixed 为与合并后格点（itertools.product 顺序）对齐的有理数字符串数组（可嵌套）。
    返回 (概率元组, 违例清单)；违例为空即通过。
    """
    flat = _flatten(fixed, model)
    if len(flat) != model.n_cells:
        raise ValueError("固定联合表格点数与模型不符")
    p = tuple(parse_rational(x, "fixed joint") for x in flat)
    violations = []
    if any(w < 0 for w in p):
        violations.append("存在负概率")
    if sum(p, ZERO) != ONE:
        violations.append(f"概率和为 {sum(p, ZERO)}，不等于 1")
    for dim_idx, d in enumerate(model.dimensions):
        for st_idx, st in enumerate(d.states):
            marginal = sum(p[c] for c in range(model.n_cells)
                           if model.cell_dim_index(c, dim_idx) == st_idx)
            if marginal != st.probability:
                violations.append(f"边际 {d.id}[{st_idx}]={marginal} 与资料 {st.probability} 不符")
    from .constraints import build_constraint_rows
    for r in build_constraint_rows(model):
        if r.kind == "marginal":
            continue
        lhs = sum(r.values[k] * p[j] for k, j in enumerate(r.indices))
        if r.sense == "<=" and lhs > r.rhs:
            violations.append(f"固定表违反 {r.ref}: {lhs} > {r.rhs}")
        if r.sense == ">=" and lhs < r.rhs:
            violations.append(f"固定表违反 {r.ref}: {lhs} < {r.rhs}")
    return p, tuple(violations)


def tail_probability(model, p, threshold, strict):
    total = ZERO
    for j, w in enumerate(p):
        L = model.cell_losses[j]
        if (L > threshold) if strict else (L >= threshold):
            total += w
    return total


def _flatten(node, model, dim=0):
    if dim == len(model.dimensions) - 1:
        return list(node)
    out = []
    for sub in node:
        out.extend(_flatten(sub, model, dim + 1))
    return out
