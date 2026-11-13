"""查看合成资料身份，并在调用方要求时计算未知关联下的联合损失锐界。

inspect 只展示资料形状；bounds 运行全可行域线性规划，输出每个阈值的
最小/最大尾概率、达到界的联合分布、约束残差与可验证对偶/Farkas 证书。
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from .io import load
from .domain import summarize
from .model import build_model
from .bounds import BoundsEngine
from .rational import qfloat, qstr
from . import references as R


def _atom_dict(a, model):
    dims = model.dimensions
    per_dim = []
    for k, st_idx in enumerate(a.combo):
        st = dims[k].states[st_idx]
        per_dim.append({
            "dimension": dims[k].id,
            "state_index": st_idx,
            "loss": qstr(st.loss),
            "source_positions": list(st.sources),
        })
    return {
        "joint_state": list(a.combo),
        "probability": qstr(a.probability),
        "contract_total_loss": qstr(a.cell_loss),
        "dimensions": per_dim,
    }


def _residual_dict(rs):
    out = []
    for r in rs:
        out.append({
            "ref": r["ref"], "sense": r["sense"],
            "rhs": qstr(r["rhs"]), "lhs": qstr(r["lhs"]),
            "slack": qstr(r["slack"]),
        })
    return out


def bounds_report(path, include_references=True, strict_values=(True, False)):
    v, h = load(path)
    model = build_model(v)
    engine = BoundsEngine(model)
    feas = engine.check_feasibility()
    report = {
        "manifest_sha256": h,
        "grid_size": model.n_cells,
        "feasible": feas.feasible,
        "loss_function": model.loss_spec.name,
        "thresholds": [],
    }
    if not feas.feasible:
        report["infeasibility_certificate"] = _jsonable(feas.farkas)
        return report
    for t in model.thresholds:
        entry = {"threshold": qstr(t), "results": []}
        for strict in strict_values:
            for sense in ("min", "max"):
                r = engine.bound(t, strict, sense)
                item = {
                    "comparison": ">" if strict else ">=",
                    "sense": sense,
                    "status": r.status,
                    "bound": qstr(r.bound) if r.bound is not None else None,
                    "bound_decimal": qfloat(r.bound) if r.bound is not None else None,
                    "pivots": r.pivots,
                    "method": r.method,
                    "extremal_distribution": [_atom_dict(a, model) for a in r.atoms],
                    "residuals": _residual_dict(r.residuals),
                    "dual_certificate": _jsonable(r.dual_certificate),
                }
                if r.note:
                    item["note"] = r.note
                entry["results"].append(item)
        report["thresholds"].append(entry)
    if include_references:
        report["references"] = _reference_block(model)
    return report


def _reference_block(model):
    """独立性、共同分位参照（显式标注为假设，不替代可行域界）。"""
    p_ind = [Fraction(0)] * model.n_cells
    for c, w in R.independence_joint(model):
        p_ind[c] = w
    p_com = [Fraction(0)] * model.n_cells
    for c, w in R.comonotone_joint(model):
        p_com[c] = w
    tails = []
    for t in model.thresholds:
        for strict in (True, False):
            tails.append({
                "threshold": qstr(t),
                "comparison": ">" if strict else ">=",
                "independence_assumption": qstr(R.tail_probability(model, p_ind, t, strict)),
                "comonotone_coupling": qstr(R.tail_probability(model, p_com, t, strict)),
            })
    return {
        "warning": "参照值依赖明确假设（独立乘积测度/共同分位耦合），不是全可行域锐界，不可替代上下界",
        "tail_probabilities": tails,
    }


def _jsonable(obj):
    if isinstance(obj, Fraction):
        return qstr(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(x) for k, x in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["inspect", "bounds"])
    p.add_argument("file")
    p.add_argument("--no-references", action="store_true")
    a = p.parse_args()
    if a.command == "inspect":
        v, h = load(a.file)
        print(json.dumps({"manifest_sha256": h,
                          "summary": summarize(v, Path(a.file).parent)},
                         ensure_ascii=False, indent=2))
    else:
        rep = bounds_report(a.file, include_references=not a.no_references)
        print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
