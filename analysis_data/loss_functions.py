"""可信本地格点损失函数登记表。

只允许调用方登记的函数与显式传值参数；任何代码字符串/可执行输入都不执行。
合同口径（data-contract.md）：
- sum-loss：各维损失直接相加；
- aggregate-cover：先各维扣免赔并截各自上限，再把和截共享上限。
"""
from dataclasses import dataclass
from fractions import Fraction

from .rational import parse_nonnegative_rational

ZERO = Fraction(0)


@dataclass(frozen=True)
class LossSpec:
    name: str
    params: dict

    def cell_loss(self, losses):
        """losses 是一个格点各维原始损失的元组（Fraction），返回合同处理后总损失。"""
        if self.name == "sum-loss":
            total = ZERO
            for x in losses:
                total += x
            return total
        if self.name == "aggregate-cover":
            p = self.params
            deductibles = p.get("deductibles")
            caps = p.get("caps")
            shared_cap = p.get("shared_cap")
            retained = ZERO
            for i, x in enumerate(losses):
                d = deductibles[i] if deductibles is not None else ZERO
                c = caps[i] if caps is not None else None
                y = x - d
                if y < 0:
                    y = ZERO
                if c is not None and y > c:
                    y = c
                retained += y
            if shared_cap is not None and retained > shared_cap:
                retained = shared_cap
            return retained
        raise ValueError(f"未登记的损失函数: {self.name!r}（不执行输入代码）")


def build_loss_spec(raw, n_dim):
    if not isinstance(raw, dict):
        raise ValueError("loss_function 必须是对象")
    name = raw.get("name")
    if name not in ("sum-loss", "aggregate-cover"):
        raise ValueError(f"未登记的损失函数: {name!r}（允许 sum-loss / aggregate-cover）")
    params_in = raw.get("parameters", {}) or {}
    if not isinstance(params_in, dict):
        raise ValueError("loss_function.parameters 必须是对象")
    if name == "sum-loss":
        if params_in:
            raise ValueError("sum-loss 不接受参数")
        return LossSpec(name, {})
    # aggregate-cover：参数长度必须与维度一致，全部显式传值
    params = {}
    for key, allow_none in (("deductibles", True), ("caps", True), ("shared_cap", False)):
        v = params_in.get(key)
        if key == "shared_cap":
            if v is None:
                raise ValueError("aggregate-cover 必须显式提供 shared_cap（可为 0）")
            params[key] = parse_nonnegative_rational(v, "shared_cap")
            continue
        if v is None:
            params[key] = None if allow_none else []
            continue
        if not isinstance(v, list) or len(v) != n_dim:
            raise ValueError(f"{key} 长度必须等于维度数 {n_dim}")
        parsed = [parse_nonnegative_rational(x, key) for x in v]
        params[key] = None if any(x is None for x in v) else parsed
    return LossSpec(name, params)
