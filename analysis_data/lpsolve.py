"""纯标准库两阶段单纯形 LP 引擎（稠密表 + Bland 规则）。

数值类型可参数化：Fraction（精确，默认）或 float（数值近似，证书会标注）。
只解最小化问题：min c·x  s.t. 约束行 (>=/<=/=) b，x>=0。
标准化口径（b 先取非负，负 RHS 整行取反并翻转约束方向）：
  "<=" 行：加松弛变量（+1），松弛直接作初始基，无需人工变量；
  "="  行：加人工变量作基；
  ">=" 行：加剩余变量（-1，仅记账、永不进基）与人工变量作基。
返回结构化变量解、原始约束对偶、Farkas 不可行证书与枢轴计数。
"""
from dataclasses import dataclass
from fractions import Fraction


class PivotLimitReached(RuntimeError):
    def __init__(self, phase, pivots):
        super().__init__(f"枢轴预算耗尽（phase={phase}, pivots={pivots}）")
        self.phase = phase
        self.pivots = pivots


@dataclass
class LPResult:
    status: str                 # "optimal" | "infeasible"
    x: list                     # 结构化变量解
    objective: object
    dual_original: list         # 每条原始约束的乘子（最小化口径，按原始行号）
    pivots: int
    farkas: dict = None         # 不可行时的证书
    numeric: bool = False
    tolerance: object = None


def _basis_order(bvar):
    kind, idx = bvar
    return (0, idx) if kind == "x" else (1, idx)


def _gauss_solve(A, b, num, tol=0):
    """解 A y = b（方阵），高斯-约当消元；返回 y。Fraction 精确，float 用容差选主元。"""
    n = len(A)
    M = [list(A[i]) + [num(b[i])] for i in range(n)]
    for col in range(n):
        if tol:
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) <= tol:
                raise ValueError("基矩阵奇异")
        else:
            piv = next((r for r in range(col, n) if M[r][col] != 0), None)
            if piv is None:
                raise ValueError("基矩阵奇异")
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f == 0 or (tol and abs(f) <= tol):
                continue
            M[r] = [a - f * bb for a, bb in zip(M[r], M[col])]
    return [M[i][-1] for i in range(n)]


class Simplex:
    def __init__(self, num=Fraction, tol=0):
        self.num = num
        self.tol = tol
        self.zero = num(0)
        self.one = num(1)

    def _neg(self, v):
        return v < -self.tol if self.tol else v < 0

    def _pos(self, v):
        return v > self.tol if self.tol else v > 0

    def solve(self, A, b, c, senses, max_pivots=None):
        m = len(A)
        n = len(c)
        if m == 0:
            return LPResult("optimal", [self.zero] * n, self.zero, [], [], 0,
                            numeric=self.num is not Fraction, tolerance=self.tol or None)
        flip = {"<=": ">=", ">=": "<=", "=": "="}
        rows, sign, eff = [], [], []
        for i in range(m):
            row = [self.num(v) for v in A[i]]
            rhs = self.num(b[i])
            if self._neg(rhs):
                row, rhs = [-v for v in row], -rhs
                sign.append(-1)
                eff.append(flip[senses[i]])
            else:
                sign.append(1)
                eff.append(senses[i])
            rows.append(row)
            b[i] = rhs

        # 列布局：结构 n | 松弛/剩余 m（每行一个，<= 为 +1，否则 -1）| 人工 k
        n_extra = m
        art_rows = [i for i in range(m) if eff[i] != "<="]
        art_pos = {row_i: k for k, row_i in enumerate(art_rows)}  # 行 -> 人工序号
        k = len(art_rows)
        ncols = 1 + n + n_extra + k
        tableau, basis = [], []
        col_of = {("x", j): 1 + j for j in range(n)}
        for i in range(m):
            col_of[("extra", i)] = 1 + n + i
        for row_i, ak in art_pos.items():
            col_of[("art", ak)] = 1 + n + n_extra + ak

        for i in range(m):
            t = [b[i]] + rows[i] + [self.zero] * n_extra + [self.zero] * k
            t[1 + n + i] = self.one if eff[i] == "<=" else -self.one
            if eff[i] == "<=":
                basis.append(("extra", i))
            else:
                ak = art_pos[i]
                t[1 + n + n_extra + ak] = self.one
                basis.append(("art", ak))
            tableau.append(t)
        inv_col = {v: key for key, v in col_of.items()}
        row_origin = list(range(m))
        # 可进基列：
        #  - 全部结构变量；
        #  - "<=" 行的松弛变量（+1）与 ">=" 不等式行的剩余变量（经枢轴后可为正，
        #    是换出人工变量的合法途径，离开行只取正系数故不破坏可行性）；
        #  - 等式行的剩余变量是冗余记账列，永不进入；人工变量仅在 phase 1 参与。
        enterable = list(range(1, 1 + n)) + [
            1 + n + i for i in range(m) if eff[i] in ("<=", ">=")]

        def current_artificial_cols():
            # 人工变量一旦换出基即永不重新进入（标准两阶段法）；只考虑仍在基中的人工列。
            return [col_of[bv] for bv in basis if bv[0] == "art"]

        def phase1_objective():
            # phase-1 目标 = 当前在基人工变量之和；每次按当前基重新规范化，
            # 避免换出的人工列在后续枢轴中污染目标行。
            g = [self.zero] * ncols
            for ac in current_artificial_cols():
                g[ac] = self.one
            for i in range(m):
                cj = g[col_of[basis[i]]]
                if cj == 0:
                    continue
                for j in range(ncols):
                    g[j] -= cj * tableau[i][j]
            return g

        def normalize_objective(raw_cost):
            obj = [self.zero] * ncols
            for j in range(n):
                obj[1 + j] = self.num(raw_cost[j])
            for i in range(m):
                cj = obj[col_of[basis[i]]]
                if cj == 0:
                    continue
                for j in range(ncols):
                    obj[j] -= cj * tableau[i][j]
            return obj

        def pivot(obj, allow_artificial, pivots, renormalize=None):
            art = current_artificial_cols() if allow_artificial else []
            candidates = enterable + art
            enter = next((j for j in candidates if self._neg(obj[j])), None)
            if enter is None:
                return pivots, False
            best = None
            for i in range(m):
                a = tableau[i][enter]
                if self._pos(a):
                    ratio = tableau[i][0] / a
                    if best is None:
                        best = (ratio, i)
                    elif ratio < best[0] - (self.tol or 0):
                        best = (ratio, i)
                    elif abs(ratio - best[0]) <= (self.tol or 0) and \
                            _basis_order(basis[i]) < _basis_order(basis[best[1]]):
                        best = (ratio, i)
            if best is None:
                raise ValueError("LP 无界（概率问题有界，出现说明建模错误）")
            leave = best[1]
            piv = tableau[leave]
            a = piv[enter]
            for j in range(ncols):
                piv[j] /= a
            for i in range(m):
                if i == leave:
                    continue
                factor = tableau[i][enter]
                if factor == 0:
                    continue
                for j in range(ncols):
                    tableau[i][j] -= factor * piv[j]
            for j in range(ncols):
                obj[j] -= obj[enter] * piv[j]
            basis[leave] = inv_col[enter]
            if renormalize is not None:
                fresh = renormalize()
                obj[:] = fresh
            return pivots + 1, True

        pivots = 0
        # ---------- phase 1（仅当存在人工变量） ----------
        if k > 0:
            obj1 = phase1_objective()
            while True:
                if max_pivots is not None and pivots >= max_pivots:
                    raise PivotLimitReached(1, pivots)
                pivots, moved = pivot(obj1, True, pivots, renormalize=phase1_objective)
                if not moved:
                    break
            if self._pos(-obj1[0]):
                # 不可行：phase-1 等式乘子 y_i = 1 - r_art_i（人工列单位位置固定）
                y_std = [self.zero] * m
                for row_i in art_rows:
                    ak = art_pos[row_i]
                    r_art = obj1[col_of[("art", ak)]]
                    y_std[row_i] = self.one - r_art
                y_orig = [y_std[i] * sign[i] for i in range(m)]
                return LPResult(
                    "infeasible", None, None, y_orig, pivots,
                    {"y_standard": y_std, "y_original": y_orig,
                     "violation": -obj1[0], "phase1_objective": -obj1[0]},
                    numeric=self.num is not Fraction, tolerance=self.tol or None)
            # 退化人工基：能换出的人工变量用结构/松弛列换出；
            # 整行在可进基列上全零的是冗余约束，其人工变量以零值保留在基中（不删行），
            # 保持 tableau 维度与列单位结构恒定，避免删行破坏目标行不变量。
            changed = True
            while changed:
                changed = False
                for ai in range(m):
                    if basis[ai][0] != "art":
                        continue
                    enter = next((j for j in enterable
                                  if self._pos(tableau[ai][j]) or self._neg(tableau[ai][j])), None)
                    if enter is None:
                        continue            # 冗余行：人工变量留基（值为 0）
                    piv = tableau[ai]
                    a = piv[enter]
                    for j in range(ncols):
                        piv[j] /= a
                    for r in range(m):
                        if r == ai:
                            continue
                        factor = tableau[r][enter]
                        if factor == 0:
                            continue
                        for j in range(ncols):
                            tableau[r][j] -= factor * piv[j]
                    basis[ai] = inv_col[enter]
                    pivots += 1
                    changed = True

        # ---------- phase 2 ----------
        # 每次枢轴后从当前基重新规范化目标行，避免退化（残留零值人工基）下
        # 增量更新漂移导致提前停在次优点。
        obj2 = normalize_objective(c)
        while True:
            if max_pivots is not None and pivots >= max_pivots:
                raise PivotLimitReached(2, pivots)
            pivots, moved = pivot(obj2, False, pivots,
                                  renormalize=lambda: normalize_objective(c))
            if not moved:
                break

        x = [self.zero] * n
        for i, bvar in enumerate(basis):
            if bvar[0] == "x":
                x[bvar[1]] = tableau[i][0]
        # 目标值以原始解 x 权威复算，不依赖可能受删列影响的目标行常数项。
        objective = sum(self.num(c[j]) * x[j] for j in range(n))
        try:
            y_orig = self._dual_from_basis(rows, b, eff, c, basis, col_of,
                                           row_origin, sign, art_pos)
        except ValueError:
            y_orig = None
        return LPResult("optimal", x, objective, y_orig, pivots,
                        numeric=self.num is not Fraction, tolerance=self.tol or None)

    def _dual_from_basis(self, rows, b, eff, c, basis, col_of, row_origin, sign, art_pos):
        m_cur = len(row_origin)
        n = len(c)
        num = self.num
        art_owner = {ak: row_i for row_i, ak in art_pos.items()}
        var_of = {v: k for k, v in col_of.items()}

        def std_column(std_j):
            """标准化等式系统在当前存活行上的列（结构/松弛剩余/人工）。"""
            col = [self.zero] * m_cur
            bvar = var_of.get(std_j)
            if bvar is not None and bvar[0] == "art":
                art_row = art_owner[bvar[1]]
                if art_row in row_origin:
                    col[row_origin.index(art_row)] = self.one
                return col
            if std_j < n:
                for r, origin in enumerate(row_origin):
                    col[r] = num(rows[origin][std_j])
            else:
                extra_row = std_j - n
                if extra_row in row_origin:
                    r = row_origin.index(extra_row)
                    col[r] = self.one if eff[extra_row] == "<=" else -self.one
            return col

        basic_std = [col_of[bv] for bv in basis]
        B = [[std_column(std_j)[r] for std_j in basic_std] for r in range(m_cur)]
        cB = [num(c[bv[1]]) if bv[0] == "x" else self.zero for bv in basis]
        BT = [[B[r][i] for r in range(m_cur)] for i in range(m_cur)]
        y_cur = _gauss_solve(BT, cB, num, self.tol)
        y_orig = [self.zero] * len(eff)
        for r, origin in enumerate(row_origin):
            y_orig[origin] = y_cur[r] * sign[origin]
        return y_orig
