"""有理数解析：输入概率/损失均为字符串，null 不等于 0，绝不使用浮点近似。"""
from fractions import Fraction


def parse_rational(x, name="值"):
    """把 '1/2'、'3'、'-7/4' 等字符串解析为 Fraction；拒绝浮点与 bool。"""
    if isinstance(x, bool) or isinstance(x, float):
        raise ValueError(f"{name} 必须是有理数字符串")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if not isinstance(x, str) or not x.strip():
        raise ValueError(f"{name} 必须是非空有理数字符串")
    s = x.strip()
    if "/" in s:
        num, den = s.split("/")
        if "/" in num or "/" in den:
            raise ValueError(f"{name} 有理数格式非法: {x!r}")
        return Fraction(int(num), int(den))  # 分母为 0 时 ZeroDivisionError
    return Fraction(int(s), 1)


def parse_nonnegative_rational(x, name="值"):
    r = parse_rational(x, name)
    if r < 0:
        raise ValueError(f"{name} 不可为负")
    return r


def qstr(q):
    """Fraction 转回紧凑字符串，便于报告序列化。"""
    if not isinstance(q, Fraction):
        return str(q)
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def qfloat(q):
    """仅用于展示，不参与任何证书计算。"""
    return float(q)
