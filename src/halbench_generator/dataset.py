"""Dataset orchestration: deterministic task seeds, bounded concurrency, atomic batches."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy
import yaml

from .engine import GenerationError, Settings, simulate
from .recipes import systems_root, digest, get_recipe, rng_for
from halbench_loader._paths import datasets_root
from .signals import Disturbance, make_inputs


def candidate(recipe, seed, split, trace_id, attempt, duration, period,
              process_noise, settings=None):
    c = recipe.cfg
    sid = c.get('random_namespace', c['system_id'])
    mode, y, initial_inputs = recipe.sample_initial(rng_for(seed, sid, split, trace_id, 0, attempt))
    signals = make_inputs(c['inputs'], rng_for(seed, sid, split, trace_id, 1, attempt),
                          duration, initial_inputs)
    p = c['process']
    disturbance = Disturbance(len(y), [c['states'].index(v) for v in p['channels']], p['scales'],
                              p['interval'], duration, process_noise,
                              rng_for(seed, sid, split, trace_id, 2, attempt))
    description = dict(attempt=attempt, initial_state=dict(zip(c['states'], y.tolist())),
                       inputs={name: s.describe() for name, s in zip(c['inputs'], signals)})
    try:
        result = simulate(recipe.model, mode, y, signals, disturbance, duration, period,
                          c['state_scales'], c['defaults']['max_step'], settings)
    except GenerationError as exc:
        exc.context.update(**description)
        raise
    return result, signals, description


def rejection(recipe, result):
    spec = recipe.cfg['acceptance']
    if spec is None:
        return []
    if spec['kind'] != 'node_bounds':
        raise ValueError('Unknown acceptance rule')
    reasons = []
    for var, (lo, hi) in spec['bounds'].items():
        i = recipe.model.states.index(var)
        if lo is not None and (result.node_min[i] <= lo if var in spec['strict_lower'] else result.node_min[i] < lo):
            reasons.append(f'{var}:lower')
        if hi is not None and result.node_max[i] > hi:
            reasons.append(f'{var}:upper')
    return reasons


def generate_trace(sid, *, trace_id=0, split=0, seed=0, duration=None, sample_period=None,
                   process_noise=0., measurement_noise=0., root=None, settings=None):
    """Maintainer interface returning latent states/jumps as well as public columns."""
    recipe = get_recipe(sid, str(root) if root is not None else None)
    c = recipe.cfg
    duration = c['defaults']['duration'] if duration is None else duration
    sample_period = c['defaults']['sample_period'] if sample_period is None else sample_period
    rejected = []
    attempts = c['acceptance']['max_attempts'] if c['acceptance'] else 1
    for attempt in range(attempts):
        try:
            result, signals, description = candidate(recipe, seed, split, trace_id, attempt,
                                                    duration, sample_period, process_noise, settings)
        except GenerationError as exc:
            exc.context.update(system_id=sid, seed=seed, split=split, trace_id=trace_id, attempt=attempt)
            raise
        except (ArithmeticError, ValueError) as exc:
            raise GenerationError('simulation', f'Undefined state, guard or reset: {exc}',
                                  system_id=sid, seed=seed, split=split,
                                  trace_id=trace_id, attempt=attempt) from exc
        reasons = rejection(recipe, result)
        if reasons:
            rejected.append(dict(**description, reasons=reasons))
            continue
        observations = result.states[:, recipe.indices].copy()
        if measurement_noise:
            observations += (rng_for(seed, c.get('random_namespace', sid), split, trace_id, 3, attempt).normal(size=observations.shape)
                             * np.asarray(c['measurement_scales'])*measurement_noise)
        cols = [result.times[:, None], observations]
        if signals:
            cols.append(np.array([[s(t) for s in signals] for t in result.times]))
        if recipe.events:
            cols.append(np.array([c['observation_events'][q] for q in result.modes], dtype=float))
        data = np.column_stack(cols)
        if not np.isfinite(data).all():
            raise GenerationError('output', 'Nonfinite observation', system_id=sid, trace_id=trace_id)
        summary = dict(**description, trace_id=trace_id, split='train' if split == 0 else 'test',
                       samples=len(data), rejected=rejected,
                       rhs_calls=result.rhs_calls, transitions=len(result.jumps),
                       boundary_suppressions=result.boundary_suppressions,
                       node_min=result.node_min.tolist(), node_max=result.node_max.tolist())
        return data, result, summary
    raise GenerationError('acceptance_budget', 'Acceptance attempts exhausted', system_id=sid,
                          seed=seed, split=split, trace_id=trace_id, rejected=rejected)


def write_trace(path, data, columns):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            zipped.write((','.join(columns)+'\n').encode())
            for start in range(0, len(data), 2048):
                text = '\n'.join(','.join(format(float(x), '.17g') for x in row)
                                 for row in data[start:start+2048])+'\n'
                zipped.write(text.encode())


def _worker(task):
    sid, kwargs, folder = task
    try:
        data, _, summary = generate_trace(sid, **kwargs)
        recipe = get_recipe(sid, kwargs.get('root'))
        columns = ['t']+recipe.outputs+list(recipe.cfg['inputs'])+recipe.events
        relative = f'{summary["split"]}/trace_{kwargs["trace_id"]:06d}.csv.gz'
        path = Path(folder)/relative
        write_trace(path, data, columns)
        return dict(ok=True, **summary, path=relative, sha256=digest(path))
    except Exception as exc:
        return dict(ok=False, category=getattr(exc, 'category', 'internal'),
                    error=f'{type(exc).__name__}: {exc}', context=getattr(exc, 'context', {}),
                    trace_id=kwargs['trace_id'], split=kwargs['split'])


def worker_count(workers, tasks):
    if workers == 'auto':
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        return min(available, max(1, tasks))
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be auto or a positive integer')
    return min(workers, max(1, tasks))


def validate_request(n_train, n_test, seed, process_noise, measurement_noise, workers):
    for name, value in dict(n_train=n_train, n_test=n_test, seed=seed).items():
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if n_train+n_test == 0:
        raise ValueError('Request at least one trace')
    for value in (process_noise, measurement_noise):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Noise strengths must be finite and nonnegative')
    return worker_count(workers, n_train+n_test)


def output_path(output_dir):
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f'Output directory already exists: {destination}. Choose a new output_dir (CLI: --output-dir).')
    if any(destination.is_relative_to(p.resolve()) for p in (datasets_root(), systems_root())):
        raise ValueError('Choose an output path outside the bundled datasets and systems directories')
    return destination


def generate_dataset(system_id, *, n_train, n_test=0, duration=None, sample_period=None,
                     seed=0, process_noise=0., measurement_noise=0., output_dir, workers='auto'):
    """Generate a dataset containing manifest.json and train/test trajectories."""
    from .engine import sample_grid
    count = validate_request(n_train, n_test, seed, process_noise, measurement_noise, workers)
    recipe = get_recipe(system_id, str(systems_root()))
    c = recipe.cfg
    duration = c['defaults']['duration'] if duration is None else float(duration)
    sample_period = c['defaults']['sample_period'] if sample_period is None else float(sample_period)
    sample_grid(duration, sample_period)
    destination = output_path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.partial-', dir=destination.parent))
    for split in ('train', 'test'):
        (stage/split).mkdir(parents=True)
    metadata = yaml.safe_load((recipe.folder/'metadata.yaml').read_text())
    source_files = sorted(Path(__file__).parent.glob('*.py'))
    manifest = dict(schema='halbench-generated-v3', status='running', system_id=system_id,
                    data_origin='generated-from-sx', data_license='MIT',
                    time_unit=metadata['time_unit'], sampling_period=sample_period,
                    variables=metadata['variables'],
                    request=dict(n_train=n_train, n_test=n_test, duration=duration,
                                 sample_period=sample_period, seed=seed,
                                 process_noise=process_noise, measurement_noise=measurement_noise),
                    columns=['t']+recipe.outputs+list(c['inputs'])+recipe.events,
                    recipe=c, system_hashes=recipe.hashes,
                    generator_hashes={p.name:digest(p) for p in source_files},
                    execution=dict(method='DOP853', rtol=1e-11, atol_factor=1e-13,
                                   transition_detection='atomic-root-localization',
                                   boundary_rule='transverse-identity-rearming-v1',
                                   sampling_side='right', workers=count),
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     scipy=scipy.__version__, pyyaml=yaml.__version__), traces=[])
    manifest_path = stage/'manifest.json'

    def save():
        manifest['counts'] = {s: sum(r['split'] == s for r in manifest['traces']) for s in ('train', 'test')}
        lengths = {r['samples'] for r in manifest['traces']}
        manifest['samples_per_trace'] = next(iter(lengths)) if len(lengths) == 1 else None
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    save()
    tasks = ((system_id, dict(trace_id=i, split=s, seed=seed, duration=duration,
                            sample_period=sample_period, process_noise=process_noise,
                            measurement_noise=measurement_noise, root=str(systems_root())), str(stage))
             for s, n in enumerate((n_train, n_test)) for i in range(n))

    def accept(row):
        if not row.pop('ok'):
            manifest['failure'] = row
            raise GenerationError(row['category'], row['error'], **row['context'])
        manifest['traces'].append(row)

    try:
        if count == 1:
            for task in tasks:
                accept(_worker(task))
        else:
            with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as pool:
                pending = deque()
                for _ in range(2*count):
                    task = next(tasks, None)
                    if task is not None:
                        pending.append(pool.submit(_worker, task))
                while pending:
                    accept(pending.popleft().result())
                    task = next(tasks, None)
                    if task is not None:
                        pending.append(pool.submit(_worker, task))
        manifest['status'] = 'complete'
        manifest['accepted'] = len(manifest['traces'])
        manifest['rejected'] = sum(len(x['rejected']) for x in manifest['traces'])
        save()
        if destination.exists():
            raise FileExistsError(f'Output appeared during generation: {destination}')
        stage.rename(destination)
        return destination
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest.setdefault('failure', dict(category=getattr(exc, 'category', 'internal'), error=str(exc),
                                            context=getattr(exc, 'context', {})))
        save()
        raise GenerationError(getattr(exc, 'category', 'internal'),
                              f'{exc}; incomplete batch and diagnostics: {stage}', staging_dir=str(stage)) from exc
