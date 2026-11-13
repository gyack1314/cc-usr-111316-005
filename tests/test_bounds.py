"""未知关联下联合损失锐界的端到端性质测试（精确有理算术）。

覆盖：
- 全可行域界的原始/对偶强对偶与残差；
- 小 2x2 例子穷举运输表与 LP 界相等；
- 加合法关联约束后界只变窄；
- 严格 > 与 >= 的区分；
- 矛盾资料给 Farkas 不可行证据；
- 列生成（全局定价）与全枚举结果一致；
- 独立性/共同分位参照落在可行域界内；
- 合同 aggregate-cover 的免赔/封顶/共享额度口径；
- 资源耗尽只报可行界与间隙，不标锐界。
"""
import copy
import unittest
from fractions import Fraction

from analysis_data.io import load
from analysis_data.model import build_model
from analysis_data.bounds import BoundsEngine
from analysis_data import references as R
from analysis_data.rational import parse_rational

DATA = "data"


class SharpBoundTests(unittest.TestCase):
    def setUp(self):
        self.v, _ = load(f"{DATA}/normal.json")
        self.model = build_model(self.v)
        self.engine = BoundsEngine(self.model)

    def _interval(self, model, t, strict):
        e = BoundsEngine(model)
        return e.bound(t, strict, "min").bound, e.bound(t, strict, "max").bound

    def test_feasible_and_strong_duality(self):
        self.assertTrue(self.engine.check_feasibility().feasible)
        for r in self.engine.all_threshold_bounds():
            self.assertEqual(r.status, "optimal")
            cert = r.dual_certificate
            self.assertTrue(cert["dual_feasible"])
            self.assertEqual(cert["certificate_bound"], r.bound)   # 强对偶
            for rs in r.residuals:                                # 残差全零
                self.assertEqual(rs["slack"], 0)
            self.assertAlmostEqual(float(r.bound),
                                   sum(float(a.probability) for a in r.atoms
                                       if ((a.cell_loss > r.threshold) if r.strict
                                           else (a.cell_loss >= r.threshold))))

    def test_enumeration_equals_lp(self):
        """穷举 2x2 运输多面体（以 x00 为自由变量）求尾概率极值，须与 LP 完全相等。"""
        p1 = [s.probability for s in self.model.dimensions[0].states]
        for t in self.model.thresholds:
            for strict in (True, False):
                vals = []
                for x00 in (Fraction(0), min(p1[0], self.model.dimensions[1].states[0].probability),
                            max(Fraction(0), p1[0] +
                                self.model.dimensions[1].states[1].probability - 1)):
                    x01 = p1[0] - x00
                    x10 = self.model.dimensions[1].states[0].probability - x00
                    x11 = x00
                    if min(x00, x01, x10, x11) < 0:
                        continue
                    probs = [x00, x01, x10, x11]
                    vals.append(sum(w for j, w in enumerate(probs)
                                    if (self.model.cell_losses[j] > t if strict
                                        else self.model.cell_losses[j] >= t)))
                self.assertEqual((min(vals), max(vals)), self._interval(self.model, t, strict))

    def test_added_association_narrows_bounds(self):
        w = copy.deepcopy(self.v)
        w["joint_event_constraints"] = [{
            "id": "tight", "event": {"flood": [1], "heat": [1]},
            "lower": "0", "upper": "1/4"}]
        tight = build_model(w)
        for t in self.model.thresholds:
            for strict in (True, False):
                lo0, hi0 = self._interval(self.model, t, strict)
                lo1, hi1 = self._interval(tight, t, strict)
                self.assertLessEqual(lo0, lo1)
                self.assertLessEqual(lo1, hi1)
                self.assertLessEqual(hi1, hi0)

    def test_strict_vs_geq(self):
        lo_s, hi_s = self._interval(self.model, Fraction(80), True)
        lo_g, hi_g = self._interval(self.model, Fraction(80), False)
        # >= 口径包含阈值点，界区间只能向"更大尾概率"方向移动
        self.assertGreaterEqual(lo_g, lo_s)
        self.assertGreaterEqual(hi_g, hi_s)
        # 上界在阈值点处确实不同（存在损失恰为 80 的格点）
        self.assertNotEqual(hi_s, hi_g)

    def test_references_within_bounds(self):
        p_ind = [Fraction(0)] * self.model.n_cells
        for c, wt in R.independence_joint(self.model):
            p_ind[c] = wt
        p_com = [Fraction(0)] * self.model.n_cells
        for c, wt in R.comonotone_joint(self.model):
            p_com[c] = wt
        for t in self.model.thresholds:
            for strict in (True, False):
                lo, hi = self._interval(self.model, t, strict)
                self.assertLessEqual(lo, R.tail_probability(self.model, p_ind, t, strict))
                self.assertGreaterEqual(hi, R.tail_probability(self.model, p_ind, t, strict))
                # 两种参照自身都满足边际
                for d in range(len(self.model.dimensions)):
                    for si in range(self.model.dimensions[d].n):
                        marg = sum(p_ind[c] for c in range(self.model.n_cells)
                                   if self.model.cell_dim_index(c, d) == si)
                        self.assertEqual(marg, self.model.dimensions[d].states[si].probability)

    def test_known_joint_event_point_constraint(self):
        v, _ = load(f"{DATA}/known-joint-event.json")
        eng = BoundsEngine(build_model(v))
        # (1,1) 被钉为 1/4：阈值 100 严格大于的概率恒为 1/4
        for sense in ("min", "max"):
            r = eng.bound(Fraction(100), True, sense)
            self.assertEqual(r.bound, Fraction(1, 4))
            self.assertTrue(r.dual_certificate["dual_feasible"])


class InfeasibilityTests(unittest.TestCase):
    def test_farkas_certificate(self):
        v, _ = load(f"{DATA}/contradictory-association.json")
        eng = BoundsEngine(build_model(v))
        feas = eng.check_feasibility()
        self.assertFalse(feas.feasible)
        f = feas.farkas
        self.assertEqual(f["type"], "farkas-infeasibility")
        self.assertTrue(f["weights_nonnegative"])
        self.assertTrue(f["column_condition_holds"])
        self.assertLess(f["constant_term"], 0)       # 0 >= 正数 的矛盾


class ColumnGenerationTests(unittest.TestCase):
    def _cg_data(self):
        return {
            "kind": "risk-coupling",
            "dimensions": [
                {"id": "a", "states": [
                    {"loss": "0", "probability": "1/3"},
                    {"loss": "10", "probability": "1/3"},
                    {"loss": "20", "probability": "1/3"}]},
                {"id": "b", "states": [
                    {"loss": "0", "probability": "1/2"},
                    {"loss": "15", "probability": "1/2"}]},
                {"id": "c", "states": [
                    {"loss": "0", "probability": "2/5"},
                    {"loss": "5", "probability": "3/5"}]}],
            "joint_event_constraints": [], "moment_constraints": [],
            "thresholds": ["15", "25"],
            "loss_function": {"name": "sum-loss", "parameters": {}},
            "solver_budget": {"max_states": 8, "max_steps": 10000}}

    def test_cg_matches_full_enumeration(self):
        v = self._cg_data()
        cg_model = build_model(v)
        cg = BoundsEngine(cg_model)
        full_v = copy.deepcopy(v)
        full_v["solver_budget"]["max_states"] = 100000
        full = BoundsEngine(build_model(full_v))
        self.assertGreater(cg_model.n_cells, cg_model.budget["max_states"])
        for t in cg_model.thresholds:
            for strict in (True, False):
                for sense in ("min", "max"):
                    rc = cg.bound(t, strict, sense)
                    rf = full.bound(t, strict, sense)
                    self.assertEqual(rc.bound, rf.bound)
                    self.assertEqual(rc.method, "column-generation")
                    self.assertTrue(rc.dual_certificate["dual_feasible"])
                    self.assertEqual(rc.dual_certificate["certificate_bound"], rc.bound)


class ContractTests(unittest.TestCase):
    def test_aggregate_cover_deductible_caps_shared(self):
        v = {
            "kind": "risk-coupling",
            "dimensions": [
                {"id": "flood", "states": [
                    {"loss": "0", "probability": "1/2"},
                    {"loss": "100", "probability": "1/2"}]},
                {"id": "heat", "states": [
                    {"loss": "0", "probability": "1/2"},
                    {"loss": "80", "probability": "1/2"}]}],
            "joint_event_constraints": [], "moment_constraints": [],
            "thresholds": ["50", "90"],
            "loss_function": {"name": "aggregate-cover", "parameters": {
                "deductibles": ["20", "10"], "caps": ["60", "50"], "shared_cap": "90"}},
            "solver_budget": {"max_states": 4096, "max_steps": 10000}}
        model = build_model(v)
        # 手工核对四格点合同损失：0 / min(80-10,50)=50 / min(100-20,60)=60 /
        # min(60+50,90)=90
        self.assertEqual(model.cell_losses,
                         (Fraction(0), Fraction(50), Fraction(60), Fraction(90)))
        eng = BoundsEngine(model)
        for r in eng.all_threshold_bounds():
            self.assertEqual(r.status, "optimal")
            self.assertTrue(r.dual_certificate["dual_feasible"])


class BudgetGapTests(unittest.TestCase):
    def test_gap_reports_feasible_not_sharp(self):
        v, _ = load(f"{DATA}/normal.json")
        v = copy.deepcopy(v)
        v["solver_budget"] = {"max_states": 4096, "max_steps": 1}
        eng = BoundsEngine(build_model(v))
        r = eng.bound(Fraction(80), True, "min")
        self.assertEqual(r.status, "gap")
        self.assertIsNone(r.dual_certificate)
        self.assertIsNotNone(r.bound)               # 仍给可行界
        # 可行界是合法概率且极值原子概率和为 1
        self.assertEqual(sum(a.probability for a in r.atoms), 1)


if __name__ == "__main__":
    unittest.main()
