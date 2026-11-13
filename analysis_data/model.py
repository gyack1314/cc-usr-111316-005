"""把冻结 JSON 资料转成内部模型：合并同值状态、保留来源、重映射事件下标。

本模块只做形状/口径核验与数据结构构建，不构造运输表、不做任何优化。
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

from .loss_functions import build_loss_spec
from .rational import parse_nonnegative_rational, parse_rational

ONE = Fraction(1)


@dataclass(frozen=True)
class State:
    loss: Fraction
    probability: Fraction
    sources: tuple            # 合并前的原始状态位置（0 起）
    source_records: tuple     # 每个来源 (原始位置, 原始损失字符串, 原始概率)


@dataclass(frozen=True)
class Dimension:
    id: str
    states: tuple             # tuple[State]

    @property
    def n(self):
        return len(self.states)


@dataclass(frozen=True)
class EventConstraint:
    id: str
    allowed: dict             # dim_index -> frozenset[合并后状态下标]
    lower: Fraction
    upper: Fraction
    cells: frozenset = None   # 命中的格点扁平下标（构建格点后回填）


@dataclass(frozen=True)
class MomentConstraint:
    id: str
    lower: Fraction
    upper: Fraction
    coefficients: tuple = None  # 与格点扁平下标对齐的 Fraction 元组；None 表示登记矩函数
    function: str = None
    parameters: dict = field(default_factory=dict)


@dataclass
class Model:
    dimensions: tuple
    events: tuple
    moments: tuple
    thresholds: tuple
    loss_spec: object
    budget: dict
    raw: dict
    # 构建后填充
    grid: list = None              # list[tuple[state indices]]
    cell_losses: tuple = None      # 合同处理后的格点损失
    strides: tuple = None
    shape: tuple = None

    @property
    def n_cells(self):
        n = 1
        for d in self.dimensions:
            n *= d.n
        return n

    def flat_index(self, combo):
        return sum(i * s for i, s in zip(combo, self.strides))

    def cell_dim_index(self, flat, dim):
        return (flat // self.strides[dim]) % self.dimensions[dim].n


def _merge_dimension(dim_id, raw_states):
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError(f"维度 {dim_id!r} 状态为空")
    parsed = []
    for pos, st in enumerate(raw_states):
        if not isinstance(st, dict):
            raise ValueError(f"维度 {dim_id!r} 状态 {pos} 必须是对象")
        loss = parse_nonnegative_rational(st["loss"], f"{dim_id}.loss")
        prob = parse_nonnegative_rational(st["probability"], f"{dim_id}.probability")
        parsed.append((loss, prob, pos, st.get("loss"), st.get("probability")))
    total = sum((p[1] for p in parsed), Fraction(0))
    if total != ONE:
        raise ValueError(f"维度 {dim_id!r} 边际概率和为 {total}，必须恰为 1")
    # 同值合并：按损失分组，保留首次出现顺序
    groups = {}
    order = []
    for loss, prob, pos, raw_loss, raw_prob in parsed:
        if loss not in groups:
            groups[loss] = {"probs": [], "records": []}
            order.append(loss)
        groups[loss]["probs"].append(prob)
        groups[loss]["records"].append((pos, raw_loss, raw_prob))
    states = tuple(
        State(
            loss=loss,
            probability=sum(groups[loss]["probs"], Fraction(0)),
            sources=tuple(r[0] for r in groups[loss]["records"]),
            source_records=tuple(groups[loss]["records"]),
        )
        for loss in order
    )
    return Dimension(id=dim_id, states=states)


def _build_events(raw_events, dimensions, old_to_new):
    events = []
    for ev in raw_events or []:
        if not isinstance(ev, dict):
            raise ValueError("joint_event_constraints 条目必须是对象")
        ev_id = ev["id"]
        raw_allowed = ev["event"]
        if not isinstance(raw_allowed, dict) or not raw_allowed:
            raise ValueError(f"事件 {ev_id!r} 至少声明一个维度的允许状态")
        allowed = {}
        for dim_id, idxs in raw_allowed.items():
            dim_pos = next((i for i, d in enumerate(dimensions) if d.id == dim_id), None)
            if dim_pos is None:
                raise ValueError(f"事件 {ev_id!r} 引用未知维度 {dim_id!r}")
            if not isinstance(idxs, list) or not idxs:
                raise ValueError(f"事件 {ev_id!r} 的 {dim_id!r} 允许状态为空")
            merged_idxs = set()
            for j in idxs:
                if type(j) is not int or j < 0 or j >= len(old_to_new[dim_pos]):
                    raise ValueError(f"事件 {ev_id!r} 状态下标越界: {dim_id}[{j}]")
                merged_idxs.add(old_to_new[dim_pos][j])
            allowed[dim_pos] = frozenset(merged_idxs)
        lower = parse_rational(ev["lower"], f"事件 {ev_id!r}.lower")
        upper = parse_rational(ev["upper"], f"事件 {ev_id!r}.upper")
        if not (Fraction(0) <= lower <= upper <= ONE):
            raise ValueError(f"事件 {ev_id!r} 闭区间必须落在 [0,1] 且 lower<=upper")
        events.append(EventConstraint(id=ev_id, allowed=allowed, lower=lower, upper=upper))
    return tuple(events)


def _build_moments(raw_moments, model_before_grid):
    moments = []
    dims = model_before_grid.dimensions
    grid_size = 1
    for d in dims:
        grid_size *= d.n
    for m in raw_moments or []:
        if not isinstance(m, dict):
            raise ValueError("moment_constraints 条目必须是对象")
        m_id = m["id"]
        lower = parse_rational(m["lower"], f"矩 {m_id!r}.lower")
        upper = parse_rational(m["upper"], f"矩 {m_id!r}.upper")
        if lower > upper:
            raise ValueError(f"矩 {m_id!r} lower>upper")
        if "grid_coefficients" in m and m["grid_coefficients"] is not None:
            flat = _flatten_coefficients(m["grid_coefficients"], dims, m_id)
            if len(flat) != grid_size:
                raise ValueError(f"矩 {m_id!r} 系数形状与合并后格点不符")
            moments.append(MomentConstraint(id=m_id, lower=lower, upper=upper, coefficients=flat))
        elif "function" in m:
            fn = m["function"]
            if fn not in REGISTERED_MOMENT_FUNCTIONS:
                raise ValueError(f"矩 {m_id!r} 使用未登记矩函数 {fn!r}（不执行输入代码）")
            params = m.get("parameters", {}) or {}
            if not isinstance(params, dict):
                raise ValueError(f"矩 {m_id!r}.parameters 必须是对象")
            moments.append(MomentConstraint(id=m_id, lower=lower, upper=upper,
                                            function=fn, parameters=params))
        else:
            raise ValueError(f"矩 {m_id!r} 须给显式格点系数或登记矩函数")
    return tuple(moments)


def _flatten_coefficients(node, dims, m_id):
    """嵌套数组按维度顺序展平为与 itertools.product 格点一致的 Fraction 元组。"""
    if len(dims) == 1:
        if not isinstance(node, list) or len(node) != dims[0].n:
            raise ValueError(f"矩 {m_id!r} 系数形状错误")
        return tuple(parse_rational(x, f"矩 {m_id!r} 系数") for x in node)
    if not isinstance(node, list) or len(node) != dims[0].n:
        raise ValueError(f"矩 {m_id!r} 系数形状错误")
    out = []
    for sub in node:
        out.extend(_flatten_coefficients(sub, dims[1:], m_id))
    return tuple(out)


def build_model(raw):
    if not isinstance(raw, dict) or raw.get("kind") != "risk-coupling":
        raise ValueError("顶层 kind 必须为 risk-coupling")
    raw_dims = raw["dimensions"]
    if not isinstance(raw_dims, list) or not raw_dims:
        raise ValueError("至少需要一个灾害维度")
    dim_ids = [d["id"] for d in raw_dims]
    if len(set(dim_ids)) != len(dim_ids):
        raise ValueError("维度 id 重复")
    dimensions = tuple(_merge_dimension(d["id"], d["states"]) for d in raw_dims)

    # 原始位置 -> 合并后状态下标（事件引用按契约使用原始位置）
    old_to_new = []
    for d in dimensions:
        mapping = {}
        for new_idx, st in enumerate(d.states):
            for old_idx in st.sources:
                mapping[old_idx] = new_idx
        old_to_new.append(mapping)

    events = _build_events(raw.get("joint_event_constraints", []), dimensions, old_to_new)
    thresholds = tuple(parse_rational(t, "threshold") for t in raw["thresholds"])
    loss_spec = build_loss_spec(raw["loss_function"], len(dimensions))
    budget = raw["solver_budget"]
    if type(budget.get("max_states")) is not int or budget["max_states"] < 1:
        raise ValueError("solver_budget.max_states 必须为正整数")
    if type(budget.get("max_steps")) is not int or budget["max_steps"] < 1:
        raise ValueError("solver_budget.max_steps 必须为正整数")

    model = Model(
        dimensions=dimensions,
        events=events,
        moments=(),
        thresholds=thresholds,
        loss_spec=loss_spec,
        budget=budget,
        raw=raw,
    )
    model.moments = _build_moments(raw.get("moment_constraints", []), model)
    _build_grid(model)
    _fill_event_cells(model)
    _fill_moment_functions(model)
    return model


def _build_grid(model):
    dims = model.dimensions
    ranges = [tuple(range(d.n)) for d in dims]
    strides = []
    s = 1
    for d in reversed(dims):
        strides.append(s)
        s *= d.n
    strides = tuple(reversed(strides))
    grid = list(product(*ranges))
    cell_losses = tuple(
        model.loss_spec.cell_loss(tuple(dims[k].states[combo[k]].loss for k in range(len(dims))))
        for combo in grid
    )
    model.grid = grid
    model.strides = strides
    model.shape = tuple(d.n for d in dims)
    model.cell_losses = cell_losses


def _fill_event_cells(model):
    new_events = []
    for ev in model.events:
        hit = set()
        for flat, combo in enumerate(model.grid):
            if all(combo[dim] in idxs for dim, idxs in ev.allowed.items()):
                hit.add(flat)
        if not hit:
            raise ValueError(f"事件 {ev.id!r} 在合并后格点上为空")
        new_events.append(EventConstraint(id=ev.id, allowed=ev.allowed, lower=ev.lower,
                                          upper=ev.upper, cells=frozenset(hit)))
    model.events = tuple(new_events)


def _moment_total_loss_expectation(model, params):
    if params:
        raise ValueError("total-loss-expectation 不接受参数")
    return model.cell_losses


REGISTERED_MOMENT_FUNCTIONS = {
    "total-loss-expectation": _moment_total_loss_expectation,
}


def _fill_moment_functions(model):
    out = []
    for m in model.moments:
        if m.coefficients is not None:
            out.append(m)
        else:
            coeffs = REGISTERED_MOMENT_FUNCTIONS[m.function](model, m.parameters)
            out.append(MomentConstraint(id=m.id, lower=m.lower, upper=m.upper,
                                        coefficients=tuple(coeffs), function=m.function,
                                        parameters=m.parameters))
    model.moments = tuple(out)
