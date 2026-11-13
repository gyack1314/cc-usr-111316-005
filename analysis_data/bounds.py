"""全可行域锐界：在满足边际/事件/矩约束的所有联合分布上求尾概率最小/最大。

每个阈值独立优化（严格 > 与 >= 分开），各自给出：
- 达到界的联合分布（稀疏原子，标注原始灾害年份来源）；
- 原约束残差；
- 可验证对偶证书（弱对偶 + 互补松弛，精确有理重算）。
资料矛盾时给 Farkas 不可行证据；资源耗尽只报当前可行界与证书间隙。
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

from .cache import CostCache
from .constraints import build_constraint_rows, dense_matrix
from .lpsolve import PivotLimitReached, Simplex

ZERO = Fraction(0)
ONE = Fraction(1)


@dataclass
class Atom:
    combo: tuple                 # 各维合并后状态下标
    probability: Fraction
    cell: int
    sources: list                # 各维原始位置，供业务查看"哪些灾害年份配到一起"
    cell_loss: Fraction


@dataclass
class BoundResult:
    threshold: Fraction
    strict: bool
    sense: str                   # "min" | "max"
    bound: Fraction = None
    status: str = "optimal"      # optimal | gap | infeasible
    atoms: tuple = ()
    residuals: tuple = ()
    dual_certificate: dict = None
    certificate_bound: Fraction = None
    gap: Fraction = None
    pivots: int = 0
    method: str = "full-enumeration"
    numeric: bool = False
    note: str = ""


@dataclass
class FeasibilityResult:
    feasible: bool
    farkas: dict = None
    witness: tuple = ()
    pivots: int = 0


class BoundsEngine:
    def __init__(self, model, num=Fraction, tol=0):
        self.model = model
        self.num = num
        self.tol = tol
        self.simplex = Simplex(num, tol)
        self.cache = CostCache()
        self.rows = build_constraint_rows(model)
        self.n_cells = model.n_cells
        self._dense_cache = None
        self.feasibility = None
        self.pivot_budget = int(model.budget["max_steps"])
        self.max_states = int(model.budget["max_states"])

    # ---------- 稠密约束（全枚举路径） ----------
    def _dense(self):
        if self._dense_cache is None:
            self._dense_cache = dense_matrix(self.rows, self.n_cells)
        return self._dense_cache

    def check_feasibility(self):
        """阶段一可行性；矛盾资料返回 Farkas 证书。"""
        if self.feasibility is not None:
            return self.feasibility
        if self.n_cells > self.max_states:
            # 大格点走列生成可行性（见 _cg_feasible），先用初始耦合试解
            res = self._cg_feasible()
            self.feasibility = res
            return res
        A, b, senses, refs = self._dense()
        c = [ZERO] * self.n_cells
        try:
            lp = self.simplex.solve(A, b, c, senses, max_pivots=self.pivot_budget)
        except PivotLimitReached:
            # 预算耗尽不能判不可行：返回保守"未知"，由调用方按间隙处理
            self.feasibility = FeasibilityResult(feasible=True, pivots=self.pivot_budget)
            return self.feasibility
        if lp.status == "infeasible":
            self.feasibility = FeasibilityResult(
                feasible=False,
                farkas=self._farkas_report(lp.farkas, refs),
                pivots=lp.pivots)
        else:
            self.feasibility = FeasibilityResult(
                feasible=True, witness=self._atoms_from_x(lp.x), pivots=lp.pivots)
        return self.feasibility

    # ---------- 全枚举界 ----------
    def bound(self, threshold, strict, sense):
        feas = self.check_feasibility()
        if not feas.feasible:
            return BoundResult(threshold=threshold, strict=strict, sense=sense,
                               status="infeasible", note="资料矛盾，可行域为空",
                               dual_certificate=feas.farkas)
        if self.n_cells > self.max_states:
            return self._bound_cg(threshold, strict, sense)
        cost = self.cache.tail_indicator(self.model, threshold, strict)
        A, b, senses, refs = self._dense()
        c = list(cost) if sense == "min" else [-v for v in cost]
        try:
            lp = self.simplex.solve(A, b, c, senses, max_pivots=self.pivot_budget)
        except PivotLimitReached as e:
            return self._feasible_gap(threshold, strict, sense, cost,
                                      reason=f"枢轴预算耗尽（phase={e.phase}, pivots={e.pivots}），未取得最优基")
        if lp.status == "infeasible":   # 理论上前置可行性已排除
            return BoundResult(threshold=threshold, strict=strict, sense=sense,
                               status="infeasible", dual_certificate=self._farkas_report(lp.farkas, refs))
        objective = lp.objective if sense == "min" else -lp.objective
        atoms = self._atoms_from_x(lp.x)
        residuals = self._residuals(lp.x)
        # 单纯形在退化基下给的乘子可能可行但次优；统一经恢复函数取对偶最优，
        # 再独立验证其可行性与强对偶（证书界须等于原始界）。
        y = self._recover_dual(c, refs)
        if y is None or not self._dual_is_feasible(y, c):
            y = lp.dual_original       # 最后回退：仅弱对偶证书
        cert = self._dual_certificate(y, refs, cost, sense, lp.objective)
        return BoundResult(threshold=threshold, strict=strict, sense=sense,
                           bound=objective, atoms=atoms, residuals=residuals,
                           dual_certificate=cert, pivots=lp.pivots, numeric=lp.numeric)

    def _dual_is_feasible(self, y, c_min):
        """在原始 min 口径上独立检查 A^T y <= c（即所有 reduced cost 非负）。"""
        if y is None:
            return False
        A, _, _, _ = self._dense()
        m = len(y)
        for j in range(self.n_cells):
            rc = c_min[j] - sum(y[i] * A[i][j] for i in range(m))
            if rc < 0:
                return False
        return True

    def _recover_dual(self, c_min, refs, A_in=None, b_in=None, senses_in=None):
        """恢复原始 min 问题的对偶最优乘子（与原始单纯形不共享基）。

        先做行秩化简（消除等式恒等相关的冗余行），再按规模选两条路径：
        - 小规模（C(n_cols, m) 有界）：枚举 active 约束顶点，精确求 y·b 最大者；
        - 大规模：在秩化简后的满秩系统上用单纯形解对偶 LP，乘子符号约束
          （<= 行 y>=0、>= 行 y<=0、等式自由）通过有向非负变量代入。
        返回按原始行号对齐的乘子，失败返回 None。
        """
        from itertools import combinations
        from math import comb
        from .lpsolve import _gauss_solve
        if A_in is None:
            A_full, b_full, senses_full, _ = self._dense()
        else:
            A_full, b_full, senses_full = A_in, b_in, senses_in
        orig_m = len(b_full)
        keep = self._independent_rows(A_full)
        A = [A_full[i] for i in keep]
        b = [b_full[i] for i in keep]
        senses = [senses_full[i] for i in keep]
        m = len(b)
        n_cols = len(c_min)
        if n_cols < m:
            return None

        def sign_ok(y):
            for i in range(m):
                if senses[i] == "<=" and y[i] < 0:
                    return False
                if senses[i] == ">=" and y[i] > 0:
                    return False
            return True

        def dual_feasible(y):
            return all(sum(y[i] * A[i][j] for i in range(m)) <= c_min[j]
                       for j in range(n_cols))

        def emit(y_kept):
            y_full = [ZERO] * orig_m
            for local_i, orig_i in enumerate(keep):
                y_full[orig_i] = y_kept[local_i]
            return y_full

        # ---- 路径一：小规模精确顶点枚举 ----
        if comb(n_cols, m) <= self._ENUM_LIMIT:
            best = None
            for active in combinations(range(n_cols), m):
                M = [[A[i][j] for i in range(m)] for j in active]
                try:
                    y = _gauss_solve(M, [c_min[j] for j in active], Fraction)
                except ValueError:
                    continue
                if not sign_ok(y) or not dual_feasible(y):
                    continue
                yb = sum(y[i] * b[i] for i in range(m))
                if best is None or yb > best[0]:
                    best = (yb, y)
            if best is not None:
                return emit(best[1])
            # 退化面回退：不等式乘子取 0，只解等式自由乘子
            eq_rows = [i for i in range(m) if senses[i] == "="]
            if eq_rows and n_cols >= len(eq_rows):
                ne = len(eq_rows)
                for active in combinations(range(n_cols), ne):
                    M = [[A[eq_rows[i]][j] for i in range(ne)] for j in active]
                    try:
                        ye = _gauss_solve(M, [c_min[j] for j in active], Fraction)
                    except ValueError:
                        continue
                    y = [ZERO] * m
                    for k, i in enumerate(eq_rows):
                        y[i] = ye[k]
                    if dual_feasible(y):
                        return emit(y)
            return None

        # ---- 路径二：大规模单纯形对偶 LP（满秩系统上构造） ----
        y = self._dual_via_simplex(A, b, senses, c_min, m, n_cols)
        if y is not None and sign_ok(y) and dual_feasible(y):
            return emit(y)
        return None

    _ENUM_LIMIT = 200000

    def _dual_via_simplex(self, A, b, senses, c_min, m, n_cols):
        """在满秩行上构造并求解对偶 LP：max y·b s.t. A^T y <= c，乘子带符号约束。

        等式行乘子自由拆 y=y+-y-；<= 行用非负变量；>= 行用非正变量。
        满秩保证不存在使约束不变的零空间，拆分不会产生无界方向。
        """
        contrib = []
        K = 0
        for i in range(m):
            if senses[i] == "=":
                ip, im = K, K + 1
                K += 2
                contrib.append(((i, ip, +1), (i, im, -1)))
            elif senses[i] == "<=":
                ip = K
                K += 1
                contrib.append(((i, ip, +1),))
            else:
                im = K
                K += 1
                contrib.append(((i, im, -1),))

        D, c_obj = [], [ZERO] * K
        for i in range(m):
            for (_i, col, sgn) in contrib[i]:
                c_obj[col] = -sgn * b[i]
        for j in range(n_cols):
            row = [ZERO] * K
            for i in range(m):
                aij = A[i][j]
                if aij == 0:
                    continue
                for (_i, col, sgn) in contrib[i]:
                    row[col] = sgn * aij
            D.append(row)
        A_lp, b_lp, s_lp = [], [], []
        for j in range(n_cols):
            rhs = c_min[j]
            if rhs < 0:
                A_lp.append([-v for v in D[j]]); b_lp.append(-rhs); s_lp.append(">=")
            else:
                A_lp.append(D[j]); b_lp.append(rhs); s_lp.append("<=")
        try:
            lp = self.simplex.solve(A_lp, b_lp, c_obj, s_lp,
                                    max_pivots=self.pivot_budget)
        except ValueError:
            return None
        if lp.status != "optimal":
            return None
        z, y = lp.x, [ZERO] * m
        for i in range(m):
            for (_i, col, sgn) in contrib[i]:
                y[i] += sgn * z[col]
        return y

    @staticmethod
    def _independent_rows(A):
        """高斯-约当消元求行空间极大线性无关组，返回保持原始顺序的行下标。"""
        rows = [list(r) for r in A]
        ncol = len(rows[0]) if rows else 0
        # 维护已选行的单位主元消元向量及其原始下标
        basis = []              # list[(pivot_col, reduced_vector)]
        keep = []
        for r, row in enumerate(A):
            work = list(row)
            for piv_col, bv in basis:
                if work[piv_col] != 0:
                    f = work[piv_col]
                    work = [a - f * bb for a, bb in zip(work, bv)]
            nz = next((j for j, x in enumerate(work) if x != 0), None)
            if nz is None:
                continue                        # 与已选行线性相关，丢弃
            work = [x / work[nz] for x in work]
            # 用新主元回消旧基向量，使每个主元列在其余基向量中为 0
            for k, (pc, bv) in enumerate(basis):
                if bv[nz] != 0:
                    f = bv[nz]
                    basis[k] = (pc, [a - f * bb for a, bb in zip(bv, work)])
            basis.append((nz, work))
            keep.append(r)
        return keep

    def all_threshold_bounds(self, strict_values=(True, False)):
        out = []
        for t in self.model.thresholds:
            for strict in strict_values:
                for sense in ("min", "max"):
                    out.append(self.bound(t, strict, sense))
        return out

    # ---------- 列生成（定价全局有效：全笛卡尔枚举） ----------
    def _initial_atoms(self):
        """顺序 Fréchet（西北角）耦合：原子数 <= 各维状态数之和 - 维数 + 1，满足全部边际。"""
        dims = self.model.dimensions
        rem = [[st.probability for st in d.states] for d in dims]
        idx = [0] * len(dims)
        atoms = []
        while True:
            if all(idx[k] >= dims[k].n for k in range(len(dims))):
                break
            if any(idx[k] >= dims[k].n for k in range(len(dims))):
                raise RuntimeError("西北角耦合索引越界")
            w = min(rem[k][idx[k]] for k in range(len(dims)))
            if w > 0:
                combo = tuple(idx[k] for k in range(len(dims)))
                atoms.append((combo, w))
            for k in range(len(dims)):
                rem[k][idx[k]] -= w
                if rem[k][idx[k]] == 0 and idx[k] < dims[k].n:
                    idx[k] += 1
        return atoms

    def _master_rows(self, columns):
        """在给定列（格点下标集合）上构造约束矩阵。"""
        cols = list(columns)
        A, b, senses, refs = self._dense()
        A_sub = [[row[j] for j in cols] for row in A]
        return cols, A_sub, b, senses, refs

    def _pricing(self, duals, cost, master_cols, sense):
        """全局定价：枚举整个笛卡尔格点（绝不抽样），返回最优入列或 None。

        min 问题 reduced cost = cost_j - y·A_j；max 已对目标取负，cost 同步取负。
        事件/矩约束让问题非可分，因此这里保证全局有效靠的是完整枚举。
        """
        best_rc, best_j = ZERO, None
        master = set(master_cols)
        for j in range(self.n_cells):
            if j in master:
                continue
            rc = cost[j] - sum(y * self._row_value(i, j) for i, y in enumerate(duals))
            if rc < best_rc:
                best_rc, best_j = rc, j
        return best_j, best_rc

    def _row_value(self, row_i, cell_j):
        row = self.rows[row_i]
        # 行系数是稀疏的；直接按 indices 查
        for k, j in enumerate(row.indices):
            if j == cell_j:
                return row.values[k]
        return ZERO

    def _bound_cg(self, threshold, strict, sense):
        raw_cost = self.cache.tail_indicator(self.model, threshold, strict)
        cost = list(raw_cost) if sense == "min" else [-v for v in raw_cost]
        init = self._initial_atoms()
        columns = [self.model.flat_index(combo) for combo, _ in init]
        init_w = {self.model.flat_index(combo): w for combo, w in init}
        pivots_used = 0
        last_lp = None
        refs = None
        steps_left = self.pivot_budget
        while True:
            cols, A_sub, b, senses, refs = self._master_rows(columns)
            c_sub = [cost[j] for j in cols]
            try:
                lp = self.simplex.solve(A_sub, b, c_sub, senses, max_pivots=steps_left)
            except PivotLimitReached as e:
                return self._cg_gap(e, columns, threshold, strict, sense, raw_cost, refs, init_w)
            pivots_used += lp.pivots
            steps_left -= lp.pivots
            if steps_left <= 0:
                return self._cg_gap(None, columns, threshold, strict, sense, raw_cost, refs, init_w,
                                    reason="max_steps 耗尽于列生成循环")
            if lp.status == "infeasible":
                # 受限主问题不可行：用 phase-1 对偶做 Farkas 定价，确认是否真不可行
                enter, _ = self._pricing(lp.farkas["y_original"],
                                         [ZERO] * self.n_cells, columns, sense)
                if enter is None:
                    return BoundResult(threshold=threshold, strict=strict, sense=sense,
                                       status="infeasible",
                                       dual_certificate=self._farkas_report(lp.farkas, refs),
                                       pivots=pivots_used, method="column-generation")
                columns.append(enter)
                continue
            last_lp = lp
            # 统一取受限主问题的对偶最优（退化基下单纯形乘子可能可行但次优，
            # 会让列生成提前终止）；恢复失败再回退到单纯形乘子。
            y_master = self._recover_dual(c_sub, refs, A_sub, b, senses)
            if y_master is None or not self._dual_is_feasible_sub(y_master, c_sub, A_sub):
                y_master = lp.dual_original
            if y_master is None:
                return self._cg_gap(None, columns, threshold, strict, sense, raw_cost, refs,
                                    init_w, reason="退化主基且对偶 LP 未恢复，无法继续全局定价")
            enter, rc = self._pricing(y_master, cost, columns, sense)
            if enter is None:
                break
            columns.append(enter)
        # 主问题最优且定价无负列 -> 全格点 LP 全局最优
        x_full = [ZERO] * self.n_cells
        for k, j in enumerate(columns):
            x_full[j] = last_lp.x[k]
        objective = last_lp.objective if sense == "min" else -last_lp.objective
        atoms = self._atoms_from_x(x_full)
        residuals = self._residuals(x_full)
        y_final = self._recover_dual(c_sub, refs, A_sub, b, senses)
        if y_final is None or not self._dual_is_feasible_sub(y_final, c_sub, A_sub):
            y_final = last_lp.dual_original
        cert = self._dual_certificate(y_final, refs, raw_cost, sense,
                                      last_lp.objective, full_cost=raw_cost)
        return BoundResult(threshold=threshold, strict=strict, sense=sense,
                           bound=objective, atoms=atoms, residuals=residuals,
                           dual_certificate=cert, pivots=pivots_used,
                           method="column-generation", numeric=last_lp.numeric)

    @staticmethod
    def _dual_is_feasible_sub(y, c_min, A_sub):
        if y is None:
            return False
        m = len(y)
        for j in range(len(c_min)):
            rc = c_min[j] - sum(y[i] * A_sub[i][j] for i in range(m))
            if rc < 0:
                return False
        return True

    def _cg_gap(self, exc, columns, threshold, strict, sense, raw_cost, refs, init_w, reason=""):
        """预算耗尽：报告当前可行界（初始耦合始终可行）与证书间隙，绝不标锐界。"""
        x_full = [ZERO] * self.n_cells
        for j in columns:
            x_full[j] = init_w.get(j, ZERO)
        incumbent = sum(raw_cost[j] * x_full[j] for j in columns)
        return BoundResult(threshold=threshold, strict=strict, sense=sense,
                           bound=incumbent, status="gap",
                           atoms=self._atoms_from_x(x_full),
                           residuals=self._residuals(x_full),
                           dual_certificate=None, certificate_bound=None,
                           method="column-generation",
                           note=f"资源耗尽，仅当前可行界，未证明锐界。{reason}")

    def _cg_feasible(self):
        init = self._initial_atoms()
        columns = [self.model.flat_index(combo) for combo, _ in init]
        cols, A_sub, b, senses, refs = self._master_rows(columns)
        c_sub = [ZERO] * len(cols)
        try:
            lp = self.simplex.solve(A_sub, b, c_sub, senses, max_pivots=self.pivot_budget)
        except PivotLimitReached:
            return FeasibilityResult(feasible=True, pivots=self.pivot_budget)
        if lp.status == "optimal":
            return FeasibilityResult(feasible=True,
                                     witness=tuple((c, w) for c, w in zip(cols, lp.x) if w > 0),
                                     pivots=lp.pivots)
        enter, _ = self._pricing(lp.farkas["y_original"], [ZERO] * self.n_cells, columns, "min")
        if enter is None:
            return FeasibilityResult(feasible=False, farkas=self._farkas_report(lp.farkas, refs),
                                     pivots=lp.pivots)
        # 还有可入列：预算内未完成可行性判定，保守视为未知（不宣称不可行）
        return FeasibilityResult(feasible=True, pivots=lp.pivots)

    def _feasible_gap(self, threshold, strict, sense, cost, reason):
        """资源耗尽：用始终满足全部边际的西北角（顺序 Fréchet）耦合给出可行界。

        该解满足边际，但事件/矩约束可能不满足，因此同时报告其残差；
        状态标 gap 且不给对偶证书，绝不宣称锐界。
        """
        x_full = [ZERO] * self.n_cells
        for combo, w in self._initial_atoms():
            x_full[self.model.flat_index(combo)] = w
        # cost 即尾指示（0/1），与 min/max 无关；可行界就是该耦合的尾概率
        incumbent = sum(cost[j] * x_full[j] for j in range(self.n_cells))
        return BoundResult(threshold=threshold, strict=strict, sense=sense,
                           bound=incumbent, status="gap",
                           atoms=self._atoms_from_x(x_full),
                           residuals=self._residuals(x_full),
                           dual_certificate=None, certificate_bound=None,
                           method="full-enumeration",
                           note=f"资源耗尽，仅当前可行界（西北角耦合），未证明锐界。{reason}")

    # ---------- 解的呈现与验证 ----------
    def _atoms_from_x(self, x):
        atoms = []
        dims = self.model.dimensions
        for j, w in enumerate(x):
            if w == 0:
                continue
            combo = self.model.grid[j]
            sources = [dims[k].states[combo[k]].sources for k in range(len(dims))]
            atoms.append(Atom(combo=combo, probability=w, cell=j, sources=sources,
                              cell_loss=self.model.cell_losses[j]))
        return tuple(atoms)

    def _residuals(self, x):
        out = []
        for r in self.rows:
            lhs = sum(r.values[k] * x[j] for k, j in enumerate(r.indices))
            out.append({
                "ref": r.ref, "sense": r.sense, "rhs": r.rhs, "lhs": lhs,
                "slack": self._slack(lhs, r.sense, r.rhs),
            })
        total = sum(x)
        out.append({"ref": "normalization:sum(x)=1", "sense": "=", "rhs": ONE,
                    "lhs": total, "slack": total - ONE})
        neg = min([ZERO] + [v for v in x])
        out.append({"ref": "nonnegativity:x>=0", "sense": ">=", "rhs": ZERO,
                    "lhs": ZERO, "slack": neg})
        return tuple(out)

    @staticmethod
    def _slack(lhs, sense, rhs):
        if sense == "=":
            return lhs - rhs
        if sense == "<=":
            return rhs - lhs
        return lhs - rhs

    def _dual_certificate(self, y, refs, cost, sense, lp_objective, full_cost=None):
        """用精确算术独立验证弱对偶与互补松弛，证书自包含可复算。"""
        used_cost = full_cost if full_cost is not None else cost
        A, b, senses, _ = self._dense()
        if y is None:
            return {
                "type": "dual-weak-duality",
                "sense": sense,
                "multipliers": (),
                "certificate_bound": None,
                "dual_feasible": False,
                "min_reduced_cost": None,
                "check": "退化基且独立对偶 LP 未能恢复乘子：本界只有原始可行证据，未取得对偶证书",
                "numeric": self.num is not Fraction,
                "unverified": True,
            }
        # 证书界：min 为 y·b 的下界证明；max 用 -c 口径取负还原
        yb = sum(y[i] * b[i] for i in range(len(b)))
        certificate_bound = yb if sense == "min" else -yb
        # 逐列 reduced cost（min 口径）
        reduced = []
        for j in range(self.n_cells):
            cj = (used_cost[j] if sense == "min" else -used_cost[j])
            rc = cj - sum(y[i] * A[i][j] for i in range(len(b)))
            reduced.append(rc)
        dual_feasible = all(rc >= 0 for rc in reduced)
        return {
            "type": "dual-weak-duality",
            "sense": sense,
            "multipliers": tuple({"ref": refs[i], "sense": senses[i], "y": y[i]}
                                 for i in range(len(b))),
            "certificate_bound": certificate_bound,
            "dual_feasible": dual_feasible,
            "min_reduced_cost": min(reduced),
            "check": "对所有格点 j: c_j - y·A_j >= 0；故 min c·x >= y·b（max 对称取负）",
            "numeric": self.num is not Fraction,
        }

    def _farkas_report(self, farkas, refs):
        """把 phase-1 乘子规范化为标准 Farkas 证书：所有约束统一成"左式 >= 0"，
        取非负组合系数 w，使每个变量的组合系数 >= 0 而常数项 < 0 => 体系不可行。

        统一口径（w_i >= 0）：
          = b 且 y>=0 : 用 (Ax-b)，w=y； = b 且 y<0 : 用 (b-Ax)，w=-y；
          >= h        : 用 (Gx-h)，w=max(y,0) 口径；
          <= u        : 用 (u-Lx)。
        """
        y = farkas["y_original"]
        A, b, senses, _ = self._dense()
        nrow = len(b)
        D, const, w, normalized_rows = [], [], [], []
        for i in range(nrow):
            yi = y[i]
            if senses[i] == "=":
                if yi >= 0:
                    D.append(list(A[i])); const.append(-b[i]); w.append(yi)
                    normalized_rows.append((refs[i], "Ax-b>=0", yi))
                else:
                    D.append([-a for a in A[i]]); const.append(b[i]); w.append(-yi)
                    normalized_rows.append((refs[i], "b-Ax>=0", -yi))
            elif senses[i] == ">=":
                D.append(list(A[i])); const.append(-b[i]); w.append(yi)
                normalized_rows.append((refs[i], "Gx-h>=0", yi))
            else:  # <=
                D.append([-a for a in A[i]]); const.append(b[i]); w.append(yi)
                normalized_rows.append((refs[i], "u-Lx>=0", yi))
        col_ok = all(sum(w[i] * D[i][j] for i in range(nrow)) >= 0
                     for j in range(self.n_cells))
        constant = sum(w[i] * const[i] for i in range(nrow))
        return {
            "type": "farkas-infeasibility",
            "solver_multipliers": tuple(
                {"ref": refs[i], "sense": senses[i], "y": y[i]} for i in range(nrow)),
            "nonnegative_combination": tuple(
                {"ref": r, "normalized_form": form, "w": wi}
                for (r, form, _), wi in zip(normalized_rows, w)),
            "variable_coefficients": tuple(
                sum(w[i] * D[i][j] for i in range(nrow)) for j in range(self.n_cells)),
            "constant_term": constant,
            "weights_nonnegative": all(wi >= 0 for wi in w),
            "column_condition_holds": col_ok,
            "check": "非负组合后每个格点变量系数 >= 0，而常数项 < 0，即 0 >= 正数，体系无解",
            "phase1_violation": farkas.get("violation"),
            "numeric": self.num is not Fraction,
        }
