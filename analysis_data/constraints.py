"""把边际、事件闭区间、线性矩约束组装成 LP 约束行（与格点扁平下标对齐）。"""
from dataclasses import dataclass
from fractions import Fraction

ONE = Fraction(1)


@dataclass
class ConstraintRow:
    kind: str           # "marginal" | "event" | "moment"
    ref: str            # 业务标识，如 "marginal:flood[1]"、"event:high-both"
    sense: str          # "=" | "<=" | ">="
    rhs: Fraction
    indices: tuple      # 非零列（格点扁平下标）
    values: tuple       # 对应系数（通常为 1，矩约束为登记系数）


def build_constraint_rows(model):
    rows = []
    dims = model.dimensions
    n_cells = model.n_cells
    # 边际等式：每维每状态一格点组
    for dim_idx, dim in enumerate(dims):
        stride = model.strides[dim_idx]
        block = dim.n
        for st_idx in range(dim.n):
            idxs = tuple(c for c in range(n_cells)
                         if model.cell_dim_index(c, dim_idx) == st_idx)
            rows.append(ConstraintRow(
                kind="marginal",
                ref=f"marginal:{dim.id}[{st_idx}]",
                sense="=",
                rhs=dim.states[st_idx].probability,
                indices=idxs,
                values=tuple(ONE for _ in idxs),
            ))
    # 事件交集概率闭区间
    for ev in model.events:
        idxs = tuple(sorted(ev.cells))
        rows.append(ConstraintRow(
            kind="event", ref=f"event:{ev.id}:lower", sense=">=", rhs=ev.lower,
            indices=idxs, values=tuple(ONE for _ in idxs)))
        rows.append(ConstraintRow(
            kind="event", ref=f"event:{ev.id}:upper", sense="<=", rhs=ev.upper,
            indices=idxs, values=tuple(ONE for _ in idxs)))
    # 线性矩闭区间
    for mo in model.moments:
        idxs = tuple(j for j, a in enumerate(mo.coefficients) if a != 0)
        vals = tuple(mo.coefficients[j] for j in idxs)
        rows.append(ConstraintRow(
            kind="moment", ref=f"moment:{mo.id}:lower", sense=">=", rhs=mo.lower,
            indices=idxs, values=vals))
        rows.append(ConstraintRow(
            kind="moment", ref=f"moment:{mo.id}:upper", sense="<=", rhs=mo.upper,
            indices=idxs, values=vals))
    return rows


def dense_matrix(rows, n_cells):
    """生成稠密 A、b、senses，供单纯形使用；返回与行对齐的元信息。"""
    A, b, senses, refs = [], [], [], []
    for r in rows:
        line = [Fraction(0)] * n_cells
        for j, v in zip(r.indices, r.values):
            line[j] = v
        A.append(line)
        b.append(r.rhs)
        senses.append(r.sense)
        refs.append(r.ref)
    return A, b, senses, refs
