from contextlib import contextmanager
import json
import sqlite3
from tariffs.catalog import Catalog
from tariffs.primitives import canonical, digest, word, integer
from analysis.engine import evaluate, verify
from analysis.quality import audit_result
from analysis.calibration import (
    CalibrationError,
    build_calibration,
    verify_binding,
)
from publication.disclosure import publication_packet, validate_packet


class Conflict(ValueError):
    pass


class Archive:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY,body TEXT NOT NULL,digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS analyses(
            id TEXT PRIMARY KEY,
            input TEXT NOT NULL REFERENCES inputs(id),
            request TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending','complete','failed')),
            result TEXT,
            error TEXT,
            revision INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS releases(
            label TEXT PRIMARY KEY,
            analysis TEXT NOT NULL REFERENCES analyses(id),
            packet TEXT NOT NULL,
            packet_digest TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS calibrations(
            id TEXT PRIMARY KEY,
            analysis TEXT NOT NULL REFERENCES analyses(id),
            request TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('complete','failed')),
            record TEXT,
            error TEXT
        );
        ''')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def atomic(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        else:
            self.db.execute('COMMIT')

    def add_input(self, ident, document):
        word(ident, 'input.id')
        catalog = Catalog(document)
        with self.atomic():
            existing = self.db.execute('SELECT * FROM inputs WHERE id=?', (ident,)).fetchone()
            if existing:
                if existing['digest'] != catalog.fingerprint:
                    raise Conflict('input id already has another immutable document')
            else:
                self.db.execute('INSERT INTO inputs VALUES(?,?,?)', (ident, canonical(catalog.document), catalog.fingerprint))
        return self.input(ident)

    def input(self, ident):
        row = self.db.execute('SELECT * FROM inputs WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        body = json.loads(row['body'])
        if digest(body) != row['digest']:
            raise ValueError('input integrity failure')
        return {'id': ident, 'document': body, 'digest': row['digest']}

    def submit(self, ident, input_id, start, end, endpoint='low', missing='exclude', assumed_income=None):
        word(ident, 'analysis.id')
        catalog = Catalog(self.input(input_id)['document'])
        from tariffs.primitives import days
        days(start, end)
        if endpoint not in ('low', 'high') or missing not in ('exclude', 'assume'):
            raise ValueError('invalid income analysis policy')
        if missing == 'assume':
            integer(assumed_income, 'assumed_income')
        elif assumed_income is not None:
            raise ValueError('unexpected assumed income for exclude policy')
        request = {'input_digest': catalog.fingerprint, 'start': start, 'end': end, 'endpoint': endpoint, 'missing': missing, 'assumed_income': assumed_income}
        signature = digest(request)
        with self.atomic():
            existing = self.db.execute('SELECT * FROM analyses WHERE id=?', (ident,)).fetchone()
            if existing:
                if existing['request_digest'] != signature or existing['input'] != input_id:
                    raise Conflict('analysis id already binds another request')
            else:
                self.db.execute('INSERT INTO analyses(id,input,request,request_digest,state) VALUES(?,?,?,?,?)', (ident, input_id, canonical(request), signature, 'pending'))
        return self.analysis(ident)

    def analysis(self, ident):
        row = self.db.execute('SELECT * FROM analyses WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        value = dict(row)
        value['request'] = json.loads(value['request'])
        if digest(value['request']) != value['request_digest']:
            raise ValueError('analysis request integrity failure')
        if value['result'] is not None:
            value['result'] = json.loads(value['result'])
            verify(value['result'])
        return value

    def execute(self, ident, expected_revision=1):
        integer(expected_revision, 'expected_revision', 1)
        current = self.analysis(ident)
        if current['state'] != 'pending' or current['revision'] != expected_revision:
            raise Conflict('analysis is stale or not pending')
        request = current['request']
        catalog = Catalog(self.input(current['input'])['document'])
        try:
            result = evaluate(catalog, request['start'], request['end'], request['endpoint'], request['missing'], request['assumed_income'])
            checked = audit_result(catalog, result)
            if not checked['valid']:
                raise ValueError('analysis arithmetic audit failed')
        except ValueError as exc:
            with self.atomic():
                updated = self.db.execute("UPDATE analyses SET state='failed',error=?,revision=revision+1 WHERE id=? AND state='pending' AND revision=?", (str(exc)[:2000], ident, expected_revision)).rowcount
                if not updated:
                    raise Conflict('another worker completed analysis') from exc
            raise
        with self.atomic():
            updated = self.db.execute("UPDATE analyses SET state='complete',result=?,error=NULL,revision=revision+1 WHERE id=? AND state='pending' AND revision=?", (canonical(result), ident, expected_revision)).rowcount
            if not updated:
                raise Conflict('another worker completed analysis')
        return self.analysis(ident)

    def retry(self, ident, expected_revision):
        integer(expected_revision, 'expected_revision', 1)
        with self.atomic():
            current = self.analysis(ident)
            if current['state'] != 'failed' or current['revision'] != expected_revision:
                raise Conflict('only current failed analyses can retry')
            self.db.execute("UPDATE analyses SET state='pending',error=NULL,revision=revision+1 WHERE id=?", (ident,))
        return self.analysis(ident)

    def release(self, label, ident, minimum_households=3):
        word(label, 'release.label')
        with self.atomic():
            current = self.analysis(ident)
            if current['state'] != 'complete':
                raise Conflict('analysis is not complete')
            packet = publication_packet(current['result'], minimum_households)
            validate_packet(packet)
            existing = self.db.execute('SELECT * FROM releases WHERE label=?', (label,)).fetchone()
            if existing:
                if existing['analysis'] != ident or existing['packet_digest'] != packet['packet_digest']:
                    raise Conflict('release label is immutable')
            else:
                self.db.execute('INSERT INTO releases VALUES(?,?,?,?)', (label, ident, canonical(packet), packet['packet_digest']))
        return self.publication(label)

    def publication(self, label):
        row = self.db.execute('SELECT * FROM releases WHERE label=?', (label,)).fetchone()
        if row is None:
            raise KeyError(label)
        packet = json.loads(row['packet'])
        validate_packet(packet)
        if packet['packet_digest'] != row['packet_digest']:
            raise ValueError('release packet integrity failure')
        current = self.analysis(row['analysis'])
        if packet['result_digest'] != current['result']['result_digest']:
            raise ValueError('release points to another analysis result')
        return {'label': label, 'analysis': row['analysis'], 'packet': packet}

    def add_calibration(self, ident, analysis_id, control_document):
        word(ident, 'calibration.id')
        run = self.analysis(analysis_id)
        if run['state'] != 'complete':
            raise Conflict('calibration requires a complete analysis')
        catalog = Catalog(self.input(run['input'])['document'])
        request = {'analysis': analysis_id, 'control': control_document}
        signature = digest(request)
        with self.atomic():
            existing = self.db.execute('SELECT * FROM calibrations WHERE id=?', (ident,)).fetchone()
            if existing and existing['request_digest'] != signature:
                raise Conflict('calibration id already binds another request')
        try:
            record = build_calibration(catalog, run['result'], control_document)
        except CalibrationError as exc:
            with self.atomic():
                self.db.execute(
                    'INSERT OR REPLACE INTO calibrations(id,analysis,request,request_digest,state,record,error) '
                    'VALUES(?,?,?,?,?,?,?)',
                    (ident, analysis_id, canonical(request), signature, 'failed', None,
                     canonical({'status': exc.status, 'message': str(exc), 'diagnostics': exc.diagnostics})))
            raise
        with self.atomic():
            self.db.execute(
                'INSERT OR REPLACE INTO calibrations(id,analysis,request,request_digest,state,record,error) '
                'VALUES(?,?,?,?,?,?,?)',
                (ident, analysis_id, canonical(request), signature, 'complete',
                 canonical(record), None))
        return self.calibration(ident)

    def calibration(self, ident):
        row = self.db.execute('SELECT * FROM calibrations WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        value = dict(row)
        request = json.loads(value['request'])
        if digest(request) != value['request_digest']:
            raise ValueError('calibration request integrity failure')
        value['request'] = request
        if value['record'] is not None:
            value['record'] = json.loads(value['record'])
            record = value['record']
            if record.get('calibration_digest') != digest(
                    {k: v for k, v in record.items() if k != 'calibration_digest'}):
                raise ValueError('calibration record integrity failure')
        if value['error'] is not None:
            value['error'] = json.loads(value['error'])
        return value

    def calibrated_record(self, ident):
        value = self.calibration(ident)
        if value['state'] != 'complete':
            raise CalibrationError(value['error']['status'] if value['error'] else 'failed',
                                   'calibration did not converge')
        run = self.analysis(value['analysis'])
        catalog = Catalog(self.input(run['input'])['document'])
        binding = value['record']['binding']
        verify_binding({
            'input_digest': binding['input_digest'],
            'result_digest': binding['result_digest'],
            'reference_date': value['record']['reference_date'],
            'household_keys': binding['household_keys'],
            'classification_digest': binding['classification_digest'],
        }, catalog, run['result'])
        return value['record']
