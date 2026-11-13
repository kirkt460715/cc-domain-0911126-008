"""Household-level reporting under a calibrated weight version.

The calibration unit is the household, so every statistic here is computed on
one row per distinct household.  A household present in several report months
must never be re-weighted or counted as multiple households: its monetary
values are summed across months while its calibrated weight is attached once.

Zero-income-with-positive-cost and missing-income households stay visible as
their own categories rather than being folded into burden ratios they cannot
have.
"""
from decimal import Decimal

from tariffs.primitives import digest
from analysis.engine import verify
from analysis.weighted import display
from analysis.calibration import CalibrationError, verify_binding


def _q(value):
    return Decimal(str(value))


def _wmean(series):
    total = sum(weight for _, weight in series)
    if not total:
        return None
    return sum(value * weight for value, weight in series) / total


def _wquantile(series, probability):
    if not series:
        return None
    ordered = sorted(series, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    threshold = probability * total
    cumulative = Decimal(0)
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _wgini(series):
    if any(value < 0 for value, _ in series):
        raise ValueError('gini requires nonnegative observations')
    ordered = sorted(series, key=lambda item: item[0])
    total_weight = sum(weight for _, weight in ordered)
    total_value = sum(value * weight for value, weight in ordered)
    if not total_weight or not total_value:
        return None
    cumulative_weight = Decimal(0)
    cumulative_value = Decimal(0)
    difference = Decimal(0)
    for value, weight in ordered:
        difference += weight * (value * cumulative_weight - cumulative_value)
        cumulative_weight += weight
        cumulative_value += value * weight
    return difference / (total_weight * total_value)


def _wthreshold_share(series, threshold):
    total = sum(weight for _, weight in series)
    if not total:
        return None
    selected = sum(weight for value, weight in series if value > threshold)
    return Decimal(selected) / total


def _household_weight_map(result):
    """Baseline weight per household, asserting month rows cannot split it."""
    weights = {}
    for row in result['families']:
        ident = row['household']
        if ident in weights and weights[ident] != row['weight']:
            raise ValueError(f'baseline weight for household {ident} differs across months')
        weights[ident] = row['weight']
    return weights


def _per_household(result, calibrated_map):
    """Collapse family-month rows into one record per distinct household."""
    baseline_map = _household_weight_map(result)
    records = {}
    order = []
    for row in result['families']:
        ident = row['household']
        if ident not in records:
            order.append(ident)
            records[ident] = {
                'household': ident,
                'region': row['region'],
                'income_band': row['income_band'],
                'baseline_weight': baseline_map[ident],
                'calibrated_weight': _q(calibrated_map[ident]),
                'months': 0,
                'operating_cents': 0,
                'subsidy_cents': 0,
                'net_operating_cents': 0,
                'deposit_cents': 0,
                'cash_outlay_cents': 0,
                'income_missing': False,
                'positive_cost_zero_income': False,
                'annual_income_cents': row['annual_income_cents'],
                'children_enrolled_any_month': 0,
                'period_income_cents_sum': 0,
                'known_income_months': 0,
                'burden_month_values': [],
            }
        record = records[ident]
        record['months'] += 1
        record['operating_cents'] += row['operating_cents']
        record['subsidy_cents'] += row['subsidy_cents']
        record['net_operating_cents'] += row['net_operating_cents']
        record['deposit_cents'] += row['deposit_cents']
        record['cash_outlay_cents'] += row['cash_outlay_cents']
        record['income_missing'] = record['income_missing'] or row['income_missing']
        record['positive_cost_zero_income'] = (
            record['positive_cost_zero_income'] or row['positive_cost_zero_income'])
        record['children_enrolled_any_month'] = max(
            record['children_enrolled_any_month'], row['children_enrolled'])
        if row['period_income_cents'] is not None:
            record['period_income_cents_sum'] += row['period_income_cents']
            record['known_income_months'] += 1
            if row['burden'] is not None:
                record['burden_month_values'].append(_q(row['burden']))
    return [records[ident] for ident in order]


def _weighted_total(records, field, weight_field):
    return sum(_q(record[field]) * record[weight_field] for record in records)


def _burden_series(records, weight_field):
    """Household burden values with weights, excluding missing/zero income."""
    series = []
    for record in records:
        if record['income_missing'] or not record['burden_month_values']:
            continue
        household_burden = sum(record['burden_month_values']) / len(record['burden_month_values'])
        series.append((household_burden, record[weight_field]))
    return series


def _cost_series(records, weight_field):
    return [(_q(record['net_operating_cents']), record[weight_field]) for record in records]


def _effective_sample_size(weights):
    total = sum(weights)
    if not total:
        return None
    return total * total / sum(weight * weight for weight in weights)


def _distribution(records, weight_field, burden_threshold):
    burdens = _burden_series(records, weight_field)
    costs = _cost_series(records, weight_field)
    return {
        'distinct_households': len(records),
        'weight_total': display(sum(record[weight_field] for record in records)),
        'burden': {
            'observations': len(burdens),
            'weighted_mean': display(_wmean(burdens)),
            'median': display(_wquantile(burdens, Decimal('.5'))),
            'p10': display(_wquantile(burdens, Decimal('.1'))),
            'p90': display(_wquantile(burdens, Decimal('.9'))),
            'above_threshold_share': display(_wthreshold_share(burdens, burden_threshold)),
            'gini': display(_wgini(burdens)),
        },
        'net_cost': {
            'weighted_mean': display(_wmean(costs)),
            'median': display(_wquantile(costs, Decimal('.5'))),
            'gini': display(_wgini(costs)),
        },
    }


def _extreme_weights(records):
    ratios = []
    for record in records:
        ratio = record['calibrated_weight'] / _q(record['baseline_weight'])
        ratios.append((record['household'], ratio, record['calibrated_weight']))
    ordered = sorted(ratios, key=lambda item: item[1], reverse=True)
    return [{
        'household': ident,
        'multiplier': display(ratio),
        'calibrated_weight': display(weight),
    } for ident, ratio, weight in ordered[:5]]


def calibrated_report(catalog, result, calibrated, burden_threshold='0.1',
                      population='all'):
    """Compare baseline and calibrated household weights side by side."""
    verify(result)
    if population not in ('all', 'enrolled'):
        raise ValueError("population must be 'all' or 'enrolled'")
    threshold = _q(burden_threshold)
    if not threshold.is_finite() or threshold < 0:
        raise ValueError('burden threshold must be finite and nonnegative')
    if calibrated.get('status') != 'converged':
        raise CalibrationError(calibrated.get('status', 'unknown'),
                               'cannot report an unconverged calibration')
    verify_binding({
        'input_digest': calibrated['binding']['input_digest'],
        'result_digest': calibrated['binding']['result_digest'],
        'reference_date': calibrated['reference_date'],
        'household_keys': calibrated['binding']['household_keys'],
        'classification_digest': calibrated['classification_digest'],
    }, catalog, result)
    if result['result_digest'] != calibrated['binding']['result_digest']:
        # A different reporting window is allowed; surface it as provenance.
        cross_window = True
    else:
        cross_window = False

    calibrated_map = {row['household']: row['calibrated_weight']
                      for row in calibrated['households']}
    records = _per_household(result, calibrated_map)
    if population == 'enrolled':
        records = [record for record in records if record['children_enrolled_any_month']]

    baseline_weight_list = [_q(record['baseline_weight']) for record in records]
    calibrated_weight_list = [record['calibrated_weight'] for record in records]

    baseline_distribution = _distribution(records, 'baseline_weight', threshold)
    calibrated_distribution = _distribution(records, 'calibrated_weight', threshold)

    missing = [record['household'] for record in records if record['income_missing']]
    zero_positive = [record['household'] for record in records
                     if record['positive_cost_zero_income']]

    body = {
        'version': 1,
        'calibration_digest': calibrated['calibration_digest'],
        'result_digest': result['result_digest'],
        'originating_result_digest': calibrated['binding']['result_digest'],
        'cross_window_reuse': cross_window,
        'input_digest': catalog.fingerprint,
        'population': population,
        'burden_threshold': str(threshold),
        'reference_date': calibrated['reference_date'],
        'aggregation_unit': 'distinct_household',
        'month_duplication_guard': 'one_weight_per_household_months_sum_costs',
        'household_count': len(records),
        'totals': {
            'baseline_weight': display(sum(baseline_weight_list)),
            'calibrated_weight': display(sum(calibrated_weight_list)),
            'baseline_weighted_net_operating_cents': display(
                _weighted_total(records, 'net_operating_cents', 'baseline_weight')),
            'calibrated_weighted_net_operating_cents': display(
                _weighted_total(records, 'net_operating_cents', 'calibrated_weight')),
            'baseline_weighted_operating_cents': display(
                _weighted_total(records, 'operating_cents', 'baseline_weight')),
            'calibrated_weighted_operating_cents': display(
                _weighted_total(records, 'operating_cents', 'calibrated_weight')),
            'baseline_weighted_subsidy_cents': display(
                _weighted_total(records, 'subsidy_cents', 'baseline_weight')),
            'calibrated_weighted_subsidy_cents': display(
                _weighted_total(records, 'subsidy_cents', 'calibrated_weight')),
        },
        'effective_sample_size': {
            'baseline': display(_effective_sample_size(baseline_weight_list)),
            'calibrated': display(_effective_sample_size(calibrated_weight_list)),
        },
        'extreme_weights': _extreme_weights(records),
        'special_households': {
            'income_missing': missing,
            'positive_cost_zero_income': zero_positive,
            'counts': {
                'income_missing_weight_baseline': display(
                    sum(_q(r['baseline_weight']) for r in records if r['income_missing'])),
                'income_missing_weight_calibrated': display(
                    sum(r['calibrated_weight'] for r in records if r['income_missing'])),
                'zero_income_positive_cost_weight_baseline': display(
                    sum(_q(r['baseline_weight']) for r in records if r['positive_cost_zero_income'])),
                'zero_income_positive_cost_weight_calibrated': display(
                    sum(r['calibrated_weight'] for r in records if r['positive_cost_zero_income'])),
            },
        },
        'baseline_distribution': baseline_distribution,
        'calibrated_distribution': calibrated_distribution,
        'households': [{
            'household': record['household'],
            'region': record['region'],
            'months_in_window': record['months'],
            'baseline_weight': display(_q(record['baseline_weight'])),
            'calibrated_weight': display(record['calibrated_weight']),
            'multiplier': display(record['calibrated_weight'] / _q(record['baseline_weight'])),
            'net_operating_cents': record['net_operating_cents'],
            'operating_cents': record['operating_cents'],
            'subsidy_cents': record['subsidy_cents'],
            'income_missing': record['income_missing'],
            'positive_cost_zero_income': record['positive_cost_zero_income'],
        } for record in records],
    }
    body['report_digest'] = digest(body)
    return body
