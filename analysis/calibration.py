"""Entropy-distance weight calibration against frozen population margins.

The calibration unit is the household.  Baseline expansion weights
(``weight`` on each household row) are never mutated; instead an independent
calibrated weight version is produced so that registered categorical margins
(region, income status) and per-household child-count constraints (total
children and children inside age bands) are met as closely as the chosen
multiplier bounds allow.

Money and income values are deliberately not part of any control vector: they
must never be used as free variables to move the burden conclusions.

The solver minimises the relative-entropy (generalised raking) distance to the
baseline weights subject to linear controls and box bounds on the multiplier,
using a dual Newton/raking iteration.  It distinguishes structural zero
support, linearly dependent or conflicting controls, and box-infeasible
targets instead of ever labelling an unconverged run as successful.
"""
import copy
import math

from tariffs.primitives import (
    digest,
    fields,
    iso,
    sequence,
    word,
    InvalidInput,
)
from analysis.engine import verify
from households.bands import child_age_band


# Registered, non-monetary household classification dimensions.  Income
# amounts are intentionally absent: only the reporting *status* may be used.
CATEGORICAL_DIMENSIONS = ('region', 'income_status')
AGE_BANDS = ('under_12_months', '12_to_23_months', '24_to_35_months', '36_months_and_over')
COUNT_DIMENSIONS = ('child_count',) + tuple('children_' + band for band in AGE_BANDS)
SUPPORTED_DIMENSIONS = CATEGORICAL_DIMENSIONS + COUNT_DIMENSIONS

# Solver safety rails.  Entropy raking can run away numerically when the
# problem is infeasible inside the bounds, so exponents are clipped.
EXPONENT_LIMIT = 60.0
TINY = 1e-12


class CalibrationError(ValueError):
    """Carries a machine-readable failure status plus diagnostics."""

    def __init__(self, status, message, diagnostics=None):
        super().__init__(message)
        self.status = status
        self.diagnostics = diagnostics or {}
        self.diagnostics['status'] = status


def _bounded_number(value, path, low, high):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise InvalidInput(path, 'expected finite number')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(path, 'expected finite number')
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise InvalidInput(path, f'expected number in {low}..{high}')
    return number


def _positive_target(value, path):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise InvalidInput(path, 'target must be a finite nonnegative number')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(path, 'target must be a finite nonnegative number')
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise InvalidInput(path, 'target must be finite and nonnegative')
    return number


def _household_features(household, reference_date):
    """Return every non-monetary feature the control file may reference."""
    income_missing = household['income_low'] is None
    features = {
        'region': household['region'],
        'income_status': 'missing' if income_missing else 'known',
        'child_count': len(household['children']),
    }
    band_counts = {band: 0 for band in AGE_BANDS}
    for child in household['children']:
        band = child_age_band(child, reference_date)
        if band == 'not_born':
            continue
        band_counts[band] += 1
    for band in AGE_BANDS:
        features['children_' + band] = band_counts[band]
    return features


def classification(households, reference_date):
    """Deterministic per-household classification snapshot used for binding.

    Any later change to household membership or to a registered classification
    changes this digest, which forces an explicit rejection rather than silent
    reuse of a stale weight version.
    """
    rows = []
    for household in sorted(households, key=lambda row: row['id']):
        features = _household_features(household, reference_date)
        rows.append({
            'household': household['id'],
            'baseline_weight': household['weight'],
            'region': features['region'],
            'income_status': features['income_status'],
            'child_count': features['child_count'],
            **{'children_' + band: features['children_' + band] for band in AGE_BANDS},
        })
    return {'reference_date': reference_date, 'households': rows, 'digest': digest(rows)}


def parse_controls(document):
    """Validate a frozen control document and build constraint vectors."""
    if not isinstance(document, dict):
        raise InvalidInput('$', 'expected object')
    fields(document, ('reference_date', 'households', 'controls', 'multiplier_bounds'),
           ('tolerance', 'max_iterations'), path='$')
    reference_date = iso(document['reference_date'], 'reference_date')

    roster = sequence(document['households'], 'households', 100000)
    registered = {}
    for index, row in enumerate(roster):
        path = f'households[{index}]'
        fields(row, ('household', 'region', 'income_status', 'child_count'), path=path)
        word(row['household'], path + '.household')
        word(row['region'], path + '.region')
        if row['income_status'] not in ('known', 'missing'):
            raise InvalidInput(path + '.income_status', "must be 'known' or 'missing'")
        if isinstance(row['child_count'], bool) or not isinstance(row['child_count'], int) or not 0 <= row['child_count'] <= 30:
            raise InvalidInput(path + '.child_count', 'expected integer 0..30')
        if row['household'] in registered:
            raise InvalidInput(path, 'duplicate household in control roster')
        registered[row['household']] = row

    bounds = document['multiplier_bounds']
    fields(bounds, ('lower', 'upper'), path='multiplier_bounds')
    lower = _bounded_number(bounds['lower'], 'multiplier_bounds.lower', 0.0, 1.0)
    upper = _bounded_number(bounds['upper'], 'multiplier_bounds.upper', 1.0, 1_000_000.0)
    if lower >= upper:
        raise InvalidInput('multiplier_bounds', 'lower bound must be below upper bound')

    tolerance = 1e-8
    max_iterations = 200
    if 'tolerance' in document:
        tolerance = _bounded_number(document['tolerance'], 'tolerance', 1e-12, 1e-2)
    if 'max_iterations' in document:
        if isinstance(document['max_iterations'], bool) or not isinstance(document['max_iterations'], int):
            raise InvalidInput('max_iterations', 'expected positive integer')
        max_iterations = document['max_iterations']
        if not 1 <= max_iterations <= 100000:
            raise InvalidInput('max_iterations', 'expected integer 1..100000')

    controls = sequence(document['controls'], 'controls', 10000)
    if not controls:
        raise InvalidInput('controls', 'at least one margin is required')
    parsed = []
    seen_ids = set()
    seen_cells = set()
    for index, control in enumerate(controls):
        path = f'controls[{index}]'
        fields(control, ('id', 'dimension', 'target'), ('category',), path=path)
        word(control['id'], path + '.id')
        if control['id'] in seen_ids:
            raise InvalidInput(path, 'duplicate control id')
        seen_ids.add(control['id'])
        dimension = control['dimension']
        if dimension not in SUPPORTED_DIMENSIONS:
            raise InvalidInput(path + '.dimension',
                               f'unsupported dimension; allowed {SUPPORTED_DIMENSIONS}')
        target = _positive_target(control['target'], path + '.target')
        if dimension in CATEGORICAL_DIMENSIONS:
            if 'category' not in control:
                raise InvalidInput(path, 'categorical control requires a category')
            word(control['category'], path + '.category')
            cell = (dimension, control['category'])
        else:
            if 'category' in control:
                raise InvalidInput(path, 'count control must not carry a category')
            cell = (dimension, None)
        if cell in seen_cells:
            raise InvalidInput(path, 'duplicate constraint cell')
        seen_cells.add(cell)
        parsed.append({'id': control['id'], 'dimension': dimension,
                       'category': control.get('category'), 'target': target})

    return {
        'reference_date': reference_date,
        'roster': registered,
        'bounds': (lower, upper),
        'tolerance': tolerance,
        'max_iterations': max_iterations,
        'controls': parsed,
    }


def build_matrix(catalog, parsed):
    """Align controls to catalog households and derive each constraint row.

    Returns one feature vector per control over households in a fixed order.
    Structural zero-support constraints are reported so the caller can fail
    explicitly rather than silently dropping the margin.
    """
    catalog_ids = set(catalog.households.by_id)
    roster_ids = set(parsed['roster'])
    if catalog_ids != roster_ids:
        missing = sorted(roster_ids - catalog_ids)
        extra = sorted(catalog_ids - roster_ids)
        raise CalibrationError(
            'roster_mismatch',
            'control roster does not match the household sample',
            {'households_missing_from_sample': missing,
             'households_missing_from_controls': extra})

    order = sorted(catalog_ids)
    households = [catalog.households.by_id[ident] for ident in order]
    features = [_household_features(row, parsed['reference_date']) for row in households]

    # Reject any drift between the registered classification and the catalog.
    for ident, feature in zip(order, features):
        registered = parsed['roster'][ident]
        if registered['region'] != feature['region']:
            raise CalibrationError('classification_changed',
                                   f'region changed for {ident}',
                                   {'household': ident, 'field': 'region'})
        if registered['income_status'] != feature['income_status']:
            raise CalibrationError('classification_changed',
                                   f'income status changed for {ident}',
                                   {'household': ident, 'field': 'income_status'})
        if registered['child_count'] != feature['child_count']:
            raise CalibrationError('classification_changed',
                                   f'child count changed for {ident}',
                                   {'household': ident, 'field': 'child_count'})

    rows = []
    zero_support = []
    for control in parsed['controls']:
        dimension = control['dimension']
        if dimension in CATEGORICAL_DIMENSIONS:
            vector = [1.0 if feature[dimension] == control['category'] else 0.0
                      for feature in features]
        else:
            vector = [float(feature[dimension]) for feature in features]
        support = sum(vector)
        rows.append({**control, 'vector': vector, 'support': support})
        if support == 0.0:
            zero_support.append({'id': control['id'], 'dimension': dimension,
                                 'category': control['category']})
    return order, households, features, rows, zero_support


# --------------------------------------------------------------------------- #
# Linear algebra helpers (small dense systems, standard library only)
# --------------------------------------------------------------------------- #
def _solve_linear(matrix, vector):
    """Gaussian elimination with partial pivoting.

    Raises ``_Singular`` when the system is singular so callers can apply a
    minimum-norm fallback instead of treating rank loss as a hard failure.
    """
    n = len(vector)
    work = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    pivot_rows = []
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < TINY:
            raise _Singular(col, pivot_rows)
        work[col], work[pivot] = work[pivot], work[col]
        pivot_rows.append(col)
        pivot_value = work[col][col]
        for r in range(col + 1, n):
            factor = work[r][col] / pivot_value
            if factor:
                for c in range(col, n + 1):
                    work[r][c] -= factor * work[col][c]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        remainder = work[row][n] - sum(work[row][c] * solution[c] for c in range(row + 1, n))
        solution[row] = remainder / work[row][row]
    return solution


class _Singular(Exception):
    def __init__(self, column, pivot_rows):
        super().__init__('singular system')
        self.column = column
        self.pivot_rows = pivot_rows


def _minimum_norm_step(hessian, gradient):
    """Solve H x = g with a damped minimum-norm solution when H is singular.

    A singular reduced Hessian means every unit supporting some constraint is
    pinned.  Adding a small ridge keeps the invertible sub-problem exact while
    giving zero contribution along the null directions, so the iteration keeps
    making progress on whichever constraints still have interior support.
    """
    dimension = len(gradient)
    ridge = [[hessian[r][c] + (TINY if r == c else 0.0)
              for c in range(dimension)] for r in range(dimension)]
    try:
        return _solve_linear(ridge, gradient)
    except _Singular:
        return [0.0] * dimension


def _gram_solve(matrix, vector):
    """Solve via the positive-semidefinite Gram matrix, or report singularity."""
    dimension = len(matrix)
    gram = [[sum(matrix[r][k] * matrix[c][k] for k in range(len(matrix[0])))
             for c in range(dimension)] for r in range(dimension)]
    rhs = [sum(matrix[r][k] * vector[k] for k in range(len(vector)))
           for r in range(dimension)]
    return _solve_linear(gram, rhs)


def linear_dependence(rows):
    """Identify linearly dependent control rows using Gram-matrix rank.

    Rows are added one at a time; a row that lies in the span of the accepted
    basis is flagged together with the coefficient combination (in terms of
    basis row indices) that reproduces it.  This works even when the dependence
    involves the implicit intercept (for example two margins that partition the
    same population), which a plain echelon pivot cannot represent.
    """
    independent = []
    dependent = []
    for index, row in enumerate(rows):
        if not independent:
            independent.append(index)
            continue
        basis = [rows[i]['vector'] for i in independent]
        try:
            coefficients = _gram_solve(basis, row['vector'])
        except _Singular:
            independent.append(index)
            continue
        residual = [row['vector'][k] - sum(coefficients[j] * basis[j][k]
                                          for j in range(len(basis)))
                    for k in range(len(row['vector']))]
        if max((abs(v) for v in residual), default=0.0) < 1e-8:
            dependent.append((index, tuple((independent[j], coefficients[j])
                                           for j in range(len(independent))
                                           if abs(coefficients[j]) > 1e-9)))
        else:
            independent.append(index)
    return independent, dependent


def _margin_conflicts(rows, dependent, baseline_weights):
    """Check redundant controls for target consistency with their span."""
    conflicts = []
    for index, combination in dependent:
        redundant = rows[index]
        implied = sum(factor * rows[basis_index]['target']
                      for basis_index, factor in combination)
        baseline_implied = sum(value * weight for value, weight in
                               zip(redundant['vector'], baseline_weights))
        mismatch = abs(implied - redundant['target'])
        if mismatch > max(1e-6, 1e-6 * max(1.0, abs(redundant['target']))):
            conflicts.append({
                'control': redundant['id'],
                'target': redundant['target'],
                'implied_target': implied,
                'difference': redundant['target'] - implied,
                'linear_combo': {rows[basis_index]['id']: factor
                                 for basis_index, factor in combination},
                'baseline_weighted_value': baseline_implied,
            })
    return conflicts


# --------------------------------------------------------------------------- #
# Entropy raking solver
# --------------------------------------------------------------------------- #
def _multipliers(g_values, lower, upper):
    """Box-constrained multipliers: clip exp(g) into [lower, upper]."""
    values = []
    for g in g_values:
        if g <= math.log(lower) - EXPONENT_LIMIT:
            values.append(lower)
        elif g >= math.log(upper) + EXPONENT_LIMIT:
            values.append(upper)
        else:
            values.append(min(upper, max(lower, math.exp(min(EXPONENT_LIMIT, max(-EXPONENT_LIMIT, g))))))
    return values


def _entropy_distance(weights, baseline_weights):
    """Sum of w * log(w / w0) - w + w0, the generalised entropy objective."""
    total = 0.0
    for weight, base in zip(weights, baseline_weights):
        ratio = weight / base
        total += weight * math.log(ratio) - weight + base
    return total


def calibrate(catalog, control_document):
    """Produce a calibrated household weight version.

    The returned object always carries ``status``; a non-converged run is
    reported with diagnostics and never marked ``converged``.
    """
    parsed = parse_controls(control_document)
    lower, upper = parsed['bounds']
    order, households, features, rows, zero_support = build_matrix(catalog, parsed)
    baseline_weights = [float(row['weight']) for row in households]

    if zero_support:
        raise CalibrationError(
            'zero_support',
            'a target margin has no sample household carrying that category',
            {'unsupported_controls': zero_support})

    independent, dependent = linear_dependence(rows)
    conflicts = _margin_conflicts(rows, dependent, baseline_weights)
    if conflicts:
        raise CalibrationError(
            'conflicting_targets',
            'linearly related margins demand inconsistent targets',
            {'conflicts': conflicts})

    active = [rows[i] for i in independent]
    redundant = [rows[i] for i, _ in dependent]
    matrix_active = [row['vector'] for row in active]
    targets = [row['target'] for row in active]
    count = len(order)

    # Capacity envelope of every active constraint inside the multiplier box.
    capacity = []
    for row in active:
        minimum = sum(lower * base * value for value, base in zip(row['vector'], baseline_weights))
        maximum = sum(upper * base * value for value, base in zip(row['vector'], baseline_weights))
        capacity.append({'id': row['id'], 'minimum': minimum, 'maximum': maximum,
                         'target': row['target']})
    overdrawn = [item for item in capacity
                 if item['target'] < item['minimum'] - 1e-6
                 or item['target'] > item['maximum'] + 1e-6]
    if overdrawn:
        raise CalibrationError(
            'infeasible_bounds',
            'at least one target lies outside the capacity given by the multiplier box',
            {'capacity_conflicts': overdrawn, 'multiplier_bounds': {'lower': lower, 'upper': upper}})

    g_values = [0.0] * count
    iterations = []
    status = 'max_iterations'
    weights = list(baseline_weights)
    previous_norm = None
    stall_rounds = 0

    for iteration in range(1, parsed['max_iterations'] + 1):
        multipliers = _multipliers(g_values, lower, upper)
        weights = [multiplier * base for multiplier, base in zip(multipliers, baseline_weights)]
        achieved = [sum(value * weight for value, weight in zip(row['vector'], weights))
                    for row in active]
        residual = [achieved[i] - targets[i] for i in range(len(active))]
        scale = [max(1.0, abs(target)) for target in targets]
        max_abs = max((abs(r) for r in residual), default=0.0)
        max_rel = max((abs(r) / s for r, s in zip(residual, scale)), default=0.0)
        residual_norm = sum(r * r for r in residual)
        pinned = [order[i] for i in range(count)
                  if multipliers[i] <= lower + TINY or multipliers[i] >= upper - TINY]
        iterations.append({
            'iteration': iteration,
            'max_abs_residual': max_abs,
            'max_relative_residual': max_rel,
            'residual_norm': residual_norm,
            'residuals': {row['id']: residual[i] for i, row in enumerate(active)},
            'achieved': {row['id']: achieved[i] for i, row in enumerate(active)},
            'at_lower_bound': [order[i] for i, m in enumerate(multipliers) if abs(m - lower) <= TINY],
            'at_upper_bound': [order[i] for i, m in enumerate(multipliers) if abs(m - upper) <= TINY],
        })
        if max_abs <= parsed['tolerance'] or max_rel <= parsed['tolerance']:
            status = 'converged'
            break

        # Stall detection: when some units are pinned against the multiplier
        # box and the residual norm has stopped improving, the remaining gap is
        # a capacity block rather than a slow-but-healthy iteration.
        if previous_norm is not None:
            if residual_norm >= previous_norm * (1.0 - 1e-10):
                stall_rounds += 1
            else:
                stall_rounds = 0
        if pinned and stall_rounds >= 5:
            status = 'blocked_by_bounds'
            break
        previous_norm = residual_norm

        # Weighted Hessian of the dual, restricted to currently interior units.
        hessian = []
        for r, row in enumerate(active):
            hessian.append([
                sum(weights[k] * row['vector'][k] * active[c]['vector'][k]
                    for k in range(count) if lower < multipliers[k] < upper)
                for c in range(len(active))])
        try:
            step = _solve_linear(hessian, residual)
        except _Singular:
            step = _minimum_norm_step(hessian, residual)
        if not any(abs(value) > TINY for value in step):
            status = 'blocked_by_bounds'
            break

        # Backtracking line search on the residual norm.  Newton moves g by
        # minus the inverse-Hessian-scaled residual; strict descent with an
        # Armijo-style floor keeps a clipped step from pinning units uselessly.
        damping = 1.0
        accepted = False
        current_norm = sum(r * r for r in residual)
        for _ in range(50):
            trial_g = [g_values[k] - damping * sum(step[r] * active[r]['vector'][k]
                                                   for r in range(len(active)))
                       for k in range(count)]
            trial_mult = _multipliers(trial_g, lower, upper)
            trial_weights = [m * base for m, base in zip(trial_mult, baseline_weights)]
            trial_residual = [sum(value * weight for value, weight in zip(row['vector'], trial_weights))
                              - row['target'] for row in active]
            trial_norm = sum(r * r for r in trial_residual)
            if trial_norm <= current_norm * (1.0 - 1e-4 * damping) + 1e-14:
                g_values = trial_g
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            status = 'blocked_by_bounds'
            break

    # Final evaluation at the retained weights.
    multipliers = _multipliers(g_values, lower, upper)
    weights = [multiplier * base for multiplier, base in zip(multipliers, baseline_weights)]
    constraint_report = _constraint_report(rows, weights, baseline_weights)
    final_residual = max((abs(item['residual']) for item in constraint_report), default=0.0)

    if status != 'converged':
        unsatisfied = [item['id'] for item in constraint_report
                       if abs(item['residual']) > max(parsed['tolerance'],
                                                      parsed['tolerance'] * max(1.0, abs(item['target'])))]
        bound_households = {
            'at_lower': [order[i] for i, m in enumerate(multipliers) if abs(m - lower) <= TINY],
            'at_upper': [order[i] for i, m in enumerate(multipliers) if abs(m - upper) <= TINY],
        }
        raise CalibrationError(
            status,
            'raking iteration stopped without satisfying every margin',
            {'final_residual': final_residual,
             'iterations_used': len(iterations),
             'unsatisfied_controls': unsatisfied,
             'bound_pinned_households': bound_households,
             'multiplier_bounds': {'lower': lower, 'upper': upper},
             'final_constraint_table': constraint_report,
             'iteration_trace': iterations,
             'capacity': capacity})

    calibrated = {
        'status': 'converged',
        'reference_date': parsed['reference_date'],
        'convergence': {
            'tolerance': parsed['tolerance'],
            'max_iterations': parsed['max_iterations'],
            'iterations_used': len(iterations),
            'final_max_abs_residual': final_residual,
        },
        'multiplier_bounds': {'lower': lower, 'upper': upper},
        'households': [
            {
                'household': order[i],
                'baseline_weight': baseline_weights[i],
                'multiplier': multipliers[i],
                'calibrated_weight': weights[i],
                'at_lower_bound': abs(multipliers[i] - lower) <= TINY,
                'at_upper_bound': abs(multipliers[i] - upper) <= TINY,
            }
            for i in range(count)
        ],
        'constraints': constraint_report,
        'redundant_controls': [
            {'id': rows[index]['id'],
             'spanned_by': {rows[basis_index]['id']: factor
                            for basis_index, factor in combination}}
            for index, combination in dependent
        ],
        'iteration_trace': iterations,
        'capacity': capacity,
        'entropy_distance': _entropy_distance(weights, baseline_weights),
        'effective_sample_size': _effective_sample_size(weights),
        'classification_digest': classification(
            [catalog.households.by_id[ident] for ident in order],
            parsed['reference_date'])['digest'],
    }
    calibrated['calibration_digest'] = digest(
        {key: value for key, value in calibrated.items() if key != 'calibration_digest'})
    return calibrated


def _effective_sample_size(weights):
    total = sum(weights)
    return total * total / sum(w * w for w in weights) if total else 0.0


def _constraint_report(rows, weights, baseline_weights):
    report = []
    for row in rows:
        target = row['target']
        achieved = sum(value * weight for value, weight in zip(row['vector'], weights))
        baseline = sum(value * weight for value, weight in zip(row['vector'], baseline_weights))
        report.append({
            'id': row['id'],
            'dimension': row['dimension'],
            'category': row['category'],
            'target': target,
            'baseline_weighted': baseline,
            'achieved_weighted': achieved,
            'residual': achieved - target,
            'relative_residual': (achieved - target) / max(1.0, abs(target)),
        })
    return report


def build_calibration(catalog, result, control_document):
    """Run calibration and bind the weight version to its exact inputs."""
    verify(result)
    if result['input_digest'] != catalog.fingerprint:
        raise CalibrationError('binding_mismatch',
                               'result belongs to another catalog document')
    calibrated = calibrate(catalog, control_document)
    calibrated['binding'] = binding_inputs(catalog, result, control_document)
    calibrated['control_document'] = copy.deepcopy(control_document)
    calibrated['calibration_digest'] = digest(
        {key: value for key, value in calibrated.items() if key != 'calibration_digest'})
    return calibrated


def calibrated_weights(calibrated):
    """Map household id to its calibrated weight (float)."""
    if calibrated.get('status') != 'converged':
        raise CalibrationError(calibrated.get('status', 'unknown'),
                               'only a converged calibration supplies weights')
    return {row['household']: row['calibrated_weight'] for row in calibrated['households']}


def binding_inputs(catalog, result, control_document):
    """Digest bundle tying a weight version to its exact inputs."""
    parsed = parse_controls(control_document)
    snapshot = classification(list(catalog.households.by_id.values()), parsed['reference_date'])
    return {
        'input_digest': catalog.fingerprint,
        'result_digest': result['result_digest'],
        'reference_date': parsed['reference_date'],
        'household_keys': sorted(catalog.households.by_id),
        'classification_digest': snapshot['digest'],
        'control_digest': digest(control_document),
    }


def verify_binding(record, catalog, result=None):
    """Reject silent reuse when household keys or classifications changed.

    The input digest, household keys and classification digest must match
    exactly.  The originating ``result_digest`` is retained as provenance but
    is not an exact-match gate, because the same household weights may be
    applied to a fee result over a different reporting window; callers that
    require the identical result pass ``strict_result=True`` data via the
    separate ``verify_result_binding`` helper.
    """
    expected = {
        'input_digest': catalog.fingerprint,
        'household_keys': sorted(catalog.households.by_id),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise CalibrationError('binding_mismatch',
                                   f'calibration record is not reusable: {key} differs',
                                   {'field': key, 'stored': record.get(key), 'current': value})
    if result is not None and result['input_digest'] != catalog.fingerprint:
        raise CalibrationError('binding_mismatch',
                               'result belongs to a different input document',
                               {'field': 'input_digest'})
    snapshot = classification(list(catalog.households.by_id.values()), record['reference_date'])
    if record.get('classification_digest') != snapshot['digest']:
        raise CalibrationError('classification_changed',
                               'household classification changed since the weight version was built',
                               {'stored': record.get('classification_digest'),
                                'current': snapshot['digest']})
    return True


def verify_result_binding(record, result):
    """Exact provenance check that the record originated from this result."""
    if record.get('result_digest') != result['result_digest']:
        raise CalibrationError('binding_mismatch',
                               'calibration originated from a different fee result',
                               {'field': 'result_digest',
                                'stored': record.get('result_digest'),
                                'current': result['result_digest']})
    return True
