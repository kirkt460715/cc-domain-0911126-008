import copy
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from tariffs.catalog import Catalog
from tariffs.primitives import InvalidInput
from analysis.engine import evaluate
from analysis import calibration as cal
from analysis.calibration import CalibrationError, build_calibration, parse_controls
from analysis.calibrated_report import calibrated_report
from publication.store import Archive, Conflict
from publication.service import Service
from publication import calibration_bundle
from publication import recovery


FIXTURE = Path('fixtures/county_calibration.json')
CONTROLS = Path('fixtures/population_controls.json')
START, END = '2026-11-01', '2026-12-01'


class CalibrationMathTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(json.loads(FIXTURE.read_text()))
        self.result = evaluate(self.catalog, START, END)
        self.controls = json.loads(CONTROLS.read_text())

    def calibrate(self, document=None):
        return build_calibration(self.catalog, self.result, document or self.controls)

    def test_converges_and_satisfies_every_margin(self):
        record = self.calibrate()
        self.assertEqual(record['status'], 'converged')
        self.assertLessEqual(record['convergence']['final_max_abs_residual'], 1e-8)
        for row in record['constraints']:
            self.assertAlmostEqual(row['achieved_weighted'], row['target'], places=6)
            self.assertLess(abs(row['residual']), 1e-6)

    def test_two_dimensional_margins_are_hand_checkable(self):
        record = self.calibrate()
        by_id = {row['id']: row for row in record['constraints']}
        weights = {row['household']: row['calibrated_weight']
                   for row in record['households']}
        # region east = h + missing + nonuser; west = i + zero
        east = sum(weights[h] for h in ('h', 'missing', 'nonuser'))
        west = weights['i'] + weights['zero']
        self.assertAlmostEqual(east, by_id['region-east']['target'], places=6)
        self.assertAlmostEqual(west, by_id['region-west']['target'], places=6)
        # income status partition sums to the same population total
        known = sum(weights[h] for h in ('h', 'i', 'zero', 'nonuser'))
        missing = weights['missing']
        self.assertAlmostEqual(known, by_id['income-known']['target'], places=6)
        self.assertAlmostEqual(missing, by_id['income-missing']['target'], places=6)
        self.assertAlmostEqual(known + missing, east + west, places=6)

    def test_age_band_child_count_constraints(self):
        record = self.calibrate()
        weights = {row['household']: row['calibrated_weight']
                   for row in record['households']}
        # under-12 infants live in missing (1) and zero (1)
        under12 = weights['missing'] + weights['zero']
        band1223 = 2 * weights['h'] + weights['i']
        by_id = {row['id']: row for row in record['constraints']}
        self.assertAlmostEqual(under12, by_id['children-under-12']['target'], places=6)
        self.assertAlmostEqual(band1223, by_id['children-12-to-23']['target'], places=6)
        total = (2 * weights['h'] + weights['i'] + weights['missing'] + weights['zero'])
        self.assertAlmostEqual(total, by_id['total-children']['target'], places=6)
        # the two bands partition the children, so 12-23 is reported redundant
        redundant = {row['id'] for row in record['redundant_controls']}
        self.assertIn('children-12-to-23', redundant)

    def test_baseline_weights_are_not_mutated(self):
        before = {row['id']: row['weight'] for row in self.catalog.households.rows}
        self.calibrate()
        after = {row['id']: row['weight'] for row in self.catalog.households.rows}
        self.assertEqual(before, after)

    def test_multipliers_stay_inside_box(self):
        record = self.calibrate()
        lower, upper = record['multiplier_bounds']['lower'], record['multiplier_bounds']['upper']
        for row in record['households']:
            self.assertGreaterEqual(row['multiplier'], lower - 1e-12)
            self.assertLessEqual(row['multiplier'], upper + 1e-12)
            self.assertAlmostEqual(row['calibrated_weight'],
                                   row['multiplier'] * row['baseline_weight'], places=9)

    def test_entropy_distance_nonnegative_and_zero_at_identity(self):
        record = self.calibrate()
        self.assertGreaterEqual(record['entropy_distance'], 0.0)
        # A target set equal to baseline weighted margins returns multiplier 1.
        doc = copy.deepcopy(self.controls)
        for control in doc['controls']:
            match = next(c for c in record['constraints'] if c['id'] == control['id'])
            control['target'] = match['baseline_weighted']
        identity = self.calibrate(doc)
        self.assertAlmostEqual(identity['entropy_distance'], 0.0, places=8)
        for row in identity['households']:
            self.assertAlmostEqual(row['multiplier'], 1.0, places=6)

    def test_effective_sample_size(self):
        record = self.calibrate()
        weights = [row['calibrated_weight'] for row in record['households']]
        expected = sum(weights) ** 2 / sum(w * w for w in weights)
        self.assertAlmostEqual(record['effective_sample_size'], expected, places=8)
        self.assertLessEqual(record['effective_sample_size'], len(weights))

    def test_zero_support_is_distinct(self):
        doc = json.loads(Path('fixtures/population_controls_zero_support.json').read_text())
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'zero_support')
        self.assertEqual(ctx.exception.diagnostics['unsupported_controls'][0]['id'],
                         'region-north')

    def test_conflicting_targets_are_distinct(self):
        doc = json.loads(Path('fixtures/population_controls_conflict.json').read_text())
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'conflicting_targets')
        conflict = ctx.exception.diagnostics['conflicts'][0]
        self.assertEqual(conflict['control'], 'income-missing')
        self.assertNotAlmostEqual(conflict['target'], conflict['implied_target'])

    def test_capacity_overrun_is_distinct(self):
        doc = json.loads(Path('fixtures/population_controls_capacity.json').read_text())
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'infeasible_bounds')
        self.assertTrue(ctx.exception.diagnostics['capacity_conflicts'])

    def test_bound_blocked_case_does_not_fake_success(self):
        doc = json.loads(Path('fixtures/population_controls_blocked.json').read_text())
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'blocked_by_bounds')
        diagnostics = ctx.exception.diagnostics
        self.assertTrue(diagnostics['unsatisfied_controls'])
        self.assertTrue(diagnostics['bound_pinned_households']['at_lower']
                        or diagnostics['bound_pinned_households']['at_upper'])
        # residual table is retained even on failure
        self.assertTrue(diagnostics['final_constraint_table'])

    def test_roster_mismatch_rejected(self):
        doc = copy.deepcopy(self.controls)
        doc['households'].pop()
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'roster_mismatch')

    def test_classification_change_rejected(self):
        doc = copy.deepcopy(self.controls)
        doc['households'][0]['region'] = 'west'
        with self.assertRaises(CalibrationError) as ctx:
            self.calibrate(doc)
        self.assertEqual(ctx.exception.status, 'classification_changed')

    def test_invalid_control_document(self):
        with self.assertRaises(InvalidInput):
            parse_controls({'reference_date': '2026-11-01'})
        bad_bounds = copy.deepcopy(self.controls)
        bad_bounds['multiplier_bounds'] = {'lower': '2', 'upper': '0.5'}
        with self.assertRaises(InvalidInput):
            parse_controls(bad_bounds)
        unknown_dim = copy.deepcopy(self.controls)
        unknown_dim['controls'][0]['dimension'] = 'income_amount'
        with self.assertRaises(InvalidInput):
            parse_controls(unknown_dim)

    def test_iteration_trace_records_residuals(self):
        record = self.calibrate()
        trace = record['iteration_trace']
        self.assertGreaterEqual(len(trace), 2)
        self.assertEqual(set(trace[0]['residuals']),
                         {c['id'] for c in record['constraints']
                          if c['id'] not in
                          {r['id'] for r in record['redundant_controls']}})
        final = trace[-1]['max_abs_residual']
        self.assertLessEqual(final, record['convergence']['tolerance'])


class CalibratedReportTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(json.loads(FIXTURE.read_text()))
        self.result = evaluate(self.catalog, START, END)
        self.record = build_calibration(self.catalog, self.result,
                                        json.loads(CONTROLS.read_text()))

    def test_one_row_per_distinct_household(self):
        report = calibrated_report(self.catalog, self.result, self.record)
        self.assertEqual(len(report['households']), 5)
        self.assertEqual(report['household_count'], 5)
        self.assertEqual(report['aggregation_unit'], 'distinct_household')
        for row in report['households']:
            self.assertEqual(row['months_in_window'], 1)

    def test_household_spanning_months_counted_once(self):
        # Extend the window to two months; the household weight must not double.
        longer = evaluate(self.catalog, START, '2027-01-01')
        report = calibrated_report(self.catalog, longer, self.record)
        weights = {row['household']: Decimal(row['calibrated_weight'])
                   for row in report['households']}
        source = {row['household']: row['calibrated_weight']
                  for row in self.record['households']}
        for ident, value in weights.items():
            self.assertAlmostEqual(float(value), source[ident], places=9)
        multi = next(row for row in report['households'] if row['household'] == 'h')
        self.assertEqual(multi['months_in_window'], 2)

    def test_special_households_kept_separate(self):
        report = calibrated_report(self.catalog, self.result, self.record)
        special = report['special_households']
        self.assertIn('missing', special['income_missing'])
        self.assertIn('zero', special['positive_cost_zero_income'])
        # missing household contributes no burden observation
        burden_obs = report['calibrated_distribution']['burden']['observations']
        self.assertEqual(burden_obs, 3)

    def test_weighted_totals_and_ess(self):
        report = calibrated_report(self.catalog, self.result, self.record)
        totals = report['totals']
        # calibrated totals equal baseline totals times the uniform multiplier 2
        self.assertAlmostEqual(
            Decimal(totals['calibrated_weighted_net_operating_cents'])
            / Decimal(totals['baseline_weighted_net_operating_cents']),
            Decimal('2'), places=4)
        self.assertEqual(report['effective_sample_size']['baseline'],
                         report['effective_sample_size']['calibrated'])

    def test_extreme_weights_listed(self):
        report = calibrated_report(self.catalog, self.result, self.record)
        self.assertTrue(report['extreme_weights'])
        self.assertLessEqual(len(report['extreme_weights']), 5)

    def test_binding_rejects_stale_result(self):
        # A different input document (changed household weight) cannot reuse
        # the weight version silently because the input digest differs.
        changed_doc = copy.deepcopy(self.catalog.document)
        changed_doc['households'][0]['weight'] = 9
        changed = Catalog(changed_doc)
        other = evaluate(changed, START, END)
        with self.assertRaises(CalibrationError):
            calibrated_report(changed, other, self.record)

    def test_binding_rejects_classification_change(self):
        changed_doc = copy.deepcopy(self.catalog.document)
        changed_doc['households'][0]['region'] = 'west'
        changed = Catalog(changed_doc)
        other = evaluate(changed, START, END)
        with self.assertRaises(CalibrationError):
            calibrated_report(changed, other, self.record)

    def test_report_digest_present(self):
        report = calibrated_report(self.catalog, self.result, self.record)
        self.assertTrue(report['report_digest'])
        json.dumps(report)  # fully serialisable, no Decimal leakage


class CalibrationArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = Archive(Path(self.temp.name) / 'archive.sqlite3')
        self.service = Service(self.archive)
        self.data = json.loads(FIXTURE.read_text())
        self.controls = json.loads(CONTROLS.read_text())
        self.archive.add_input('county', self.data)
        self.archive.submit('run', 'county', START, END)
        self.archive.execute('run')

    def tearDown(self):
        self.archive.close()
        self.temp.cleanup()

    def _calibrate(self, ident='cal', controls=None):
        return self.archive.add_calibration(ident, 'run', controls or self.controls)

    def test_successful_calibration_persisted(self):
        stored = self._calibrate()
        self.assertEqual(stored['state'], 'complete')
        fetched = self.archive.calibration('cal')
        self.assertEqual(fetched['record']['calibration_digest'],
                         stored['record']['calibration_digest'])
        self.assertTrue(recovery.audit(self.archive)['valid'])

    def test_failed_calibration_is_recorded_not_raised_as_success(self):
        blocked = json.loads(Path('fixtures/population_controls_blocked.json').read_text())
        with self.assertRaises(CalibrationError):
            self._calibrate('bad', blocked)
        stored = self.archive.calibration('bad')
        self.assertEqual(stored['state'], 'failed')
        self.assertIsNone(stored['record'])
        self.assertEqual(stored['error']['status'], 'blocked_by_bounds')
        with self.assertRaises(CalibrationError):
            self.archive.calibrated_record('bad')

    def test_immutable_calibration_request(self):
        self._calibrate()
        altered = copy.deepcopy(self.controls)
        altered['controls'][0]['target'] = 13
        with self.assertRaises(Conflict):
            self.archive.add_calibration('cal', 'run', altered)

    def test_rebinding_rejects_changed_households(self):
        self._calibrate()
        # Swap the underlying input to a different household composition.
        changed = copy.deepcopy(self.data)
        changed['households'][0]['region'] = 'west'
        self.archive.add_input('county2', changed)
        self.archive.submit('run2', 'county2', START, END)
        self.archive.execute('run2')
        with self.assertRaises(CalibrationError):
            self.archive.add_calibration('reuse', 'run2', self.controls)

    def test_service_report_and_recovery_audit(self):
        self._calibrate()
        report = self.service.report('run', 'calibrated', {'calibration': 'cal'})
        self.assertEqual(report['calibration_digest'],
                         self.archive.calibration('cal')['record']['calibration_digest'])
        audit = recovery.audit(self.archive)
        self.assertTrue(audit['valid'])
        self.assertEqual(audit['calibrations'], 1)

    def test_bundle_export_roundtrip(self):
        self._calibrate()
        record = self.archive.calibrated_record('cal')
        result = self.service.result('run')
        catalog = self.service.source('run')
        target = Path(self.temp.name) / 'bundle'
        outcome = calibration_bundle.write(catalog, result, record, target)
        self.assertTrue(outcome['verified'])
        checked = calibration_bundle.verify_directory(target)
        self.assertEqual(checked['calibration_digest'], record['calibration_digest'])
        names = {p.name for p in target.iterdir()}
        self.assertIn('constraints.json', names)
        self.assertIn('household-weights.json', names)
        self.assertIn('iteration-trace.json', names)
        constraints = json.loads((target / 'constraints.json').read_text())
        for row in constraints['rows']:
            self.assertIn('target', row)
            self.assertIn('achieved_weighted', row)
            self.assertIn('residual', row)


if __name__ == '__main__':
    unittest.main()
