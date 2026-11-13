"""格点费用缓存：缓存键绑定合同、合并后状态与阈值口径（严格/>=）。

不同合同参数、不同状态合并结果或不同比较口径一律缓存不命中，
杜绝把 sum-loss 的费用误用于 aggregate-cover。
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ContractFingerprint:
    name: str
    params: tuple          # 排序后的 (键, 有理字符串) 结构
    shape: tuple
    state_losses: tuple    # 每维 (损失字符串...)，标识合并后状态
    strict: bool
    threshold: str

    @classmethod
    def make(cls, model, threshold, strict):
        spec = model.loss_spec
        param_items = []
        for k in sorted(spec.params):
            v = spec.params[k]
            if isinstance(v, list):
                v = tuple(str(x) for x in v)
            param_items.append((k, str(v)))
        state_losses = tuple(tuple(str(st.loss) for st in d.states) for d in model.dimensions)
        return cls(name=spec.name, params=tuple(param_items), shape=model.shape,
                   state_losses=state_losses, strict=strict, threshold=str(threshold))


class CostCache:
    def __init__(self):
        self._cell_loss = {}     # 合同+状态 -> 格点损失元组
        self._tail = {}          # 完整指纹 -> 0/1 费用元组

    def cell_losses(self, model):
        key = (model.loss_spec.name,
               tuple(sorted((k, str(v)) for k, v in model.loss_spec.params.items())),
               model.shape)
        if key not in self._cell_loss:
            self._cell_loss[key] = model.cell_losses
        return self._cell_loss[key]

    def tail_indicator(self, model, threshold, strict):
        fp = ContractFingerprint.make(model, threshold, strict)
        if fp in self._tail:
            return self._tail[fp]
        losses = self.cell_losses(model)
        if strict:
            cost = tuple(Fraction(1) if L > threshold else Fraction(0) for L in losses)
        else:
            cost = tuple(Fraction(1) if L >= threshold else Fraction(0) for L in losses)
        self._tail[fp] = cost
        return cost
