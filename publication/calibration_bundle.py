"""Export a calibrated weight version for manual, cell-by-cell review.

Analysts must be able to recompute every margin from the household rows, so
the bundle keeps the full constraint table (target, baseline weighted value,
achieved value, residual), per-household weights and multipliers, the solver
trace, and the binding digests.  Nothing here is a bare "weighted" flag.
"""
import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile

from tariffs.primitives import canonical
from analysis.calibration import verify_binding
from analysis.calibrated_report import calibrated_report


def contents(catalog, result, record, burden_threshold='0.1', population='all'):
    binding = record['binding']
    verify_binding({
        'input_digest': binding['input_digest'],
        'result_digest': binding['result_digest'],
        'reference_date': record['reference_date'],
        'household_keys': binding['household_keys'],
        'classification_digest': binding['classification_digest'],
    }, catalog, result)
    report = calibrated_report(catalog, result, record, burden_threshold, population)
    files = {
        'input.json': canonical(catalog.document),
        'analysis.json': canonical(result),
        'controls.json': canonical(record['control_document']),
        'calibration.json': canonical(record),
        'constraints.json': canonical({
            'reference_date': record['reference_date'],
            'rows': record['constraints'],
            'redundant_controls': record['redundant_controls'],
        }),
        'household-weights.json': canonical({
            'reference_date': record['reference_date'],
            'rows': record['households'],
        }),
        'iteration-trace.json': canonical({
            'convergence': record['convergence'],
            'iterations': record['iteration_trace'],
        }),
        'comparison.json': canonical(report),
    }
    manifest = {
        'version': 1,
        'access': 'restricted_analysis_contains_synthetic_or_authorized_household_data',
        'input_digest': catalog.fingerprint,
        'result_digest': result['result_digest'],
        'calibration_digest': record['calibration_digest'],
        'classification_digest': record['classification_digest'],
        'control_digest': binding['control_digest'],
        'burden_threshold': str(burden_threshold),
        'population': population,
        'files': {name: hashlib.sha256(value.encode()).hexdigest()
                  for name, value in sorted(files.items())},
    }
    files['manifest.json'] = canonical(manifest)
    return files


def write(catalog, result, record, directory, burden_threshold='0.1', population='all'):
    files = contents(catalog, result, record, burden_threshold, population)
    root = Path(directory)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.nursery-cal-', dir=root.parent))
    try:
        for name, content in files.items():
            with (temporary / name).open('w', encoding='utf-8', newline='') as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        os.rename(temporary, root)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return verify_directory(root)


def verify_directory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('calibration bundle must be a regular directory')
    manifest_path = root / 'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('invalid calibration bundle manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('unsupported calibration bundle manifest')
    names = set(manifest['files']) | {'manifest.json'}
    if names != {path.name for path in root.iterdir()}:
        raise ValueError('bundle file set mismatch')
    for name, checksum in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('bundle member must be a regular file')
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('bundle checksum mismatch')
    record = json.loads((root / 'calibration.json').read_text(encoding='utf-8'))
    if record['calibration_digest'] != manifest['calibration_digest']:
        raise ValueError('calibration digest mismatch')
    return {'verified': True,
            'calibration_digest': manifest['calibration_digest'],
            'files': len(names)}
