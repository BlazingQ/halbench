"""Load benchmark generation settings and sample initial conditions."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .sx import Model, Predicate, function
from halbench_loader._paths import systems_root



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_for(seed, sid, split, trace_id, purpose, attempt=0):
    sid_words = (np.frombuffer(hashlib.sha256(sid.encode()).digest()[:16], dtype='<u4').tolist()
                 if isinstance(sid, str) else list(sid))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [*sid_words, split, trace_id, purpose, attempt, seed])))


class Recipe:
    def __init__(self, sid, root=None):
        root = Path(root) if root is not None else systems_root()
        if Path(sid).name != sid or sid in ('.', '..'):
            raise ValueError('Invalid system_id')
        self.folder = root/sid
        self.path = self.folder/'generation.yaml'
        meta = yaml.safe_load((self.folder/'metadata.yaml').read_text())
        if meta['type'] != 'synthetic':
            raise ValueError(f'{sid} is a real-world dataset; select a synthetic benchmark for trace generation')
        self.cfg = yaml.safe_load(self.path.read_text())
        c = self.cfg
        if (c['schema'] != 'halbench-generation-v1' or c['system_id'] != sid
                or meta['system_id'] != sid or meta['type'] != 'synthetic'):
            raise ValueError('Unsupported task schema/system')
        namespace = c.get('random_namespace')
        if namespace is not None and (not isinstance(namespace, list) or len(namespace) != 4
                or any(type(v) is not int or not 0 <= v < 2**32 for v in namespace)):
            raise ValueError('random_namespace must contain four unsigned 32-bit integers')
        self.model = Model(self.path.with_name('model.xml'), c['states'], list(c['inputs']))
        self.outputs = meta['variables']['states']
        self.events = meta['variables'].get('events', [])
        if (len(self.outputs) != meta['dim']['output'] or
                len(c['inputs']) != meta['dim']['input'] or self.events):
            raise ValueError('Learning columns must exactly match declared input/output dimensions')
        if list(c['inputs']) != meta['variables'].get('inputs', []) or c['outputs'] != self.outputs:
            raise ValueError('Recipe output columns differ from metadata')
        self.indices = [c['states'].index(v) for v in self.outputs]
        if set(c['observation_events']) != set(self.model.modes) and self.events:
            raise ValueError('Complete event observation map required')
        if self.events and any(len(v) != len(self.events) for v in c['observation_events'].values()):
            raise ValueError('Invalid event observation dimensions')
        self.initial = []
        for loc in c['initial']['components']:
            names = list(loc.get('draws', {}))
            state = function([loc['state'][v] for v in c['states']], names)
            predicate = Predicate(loc['mode_by']['predicate'], names) if 'mode_by' in loc else None
            self.initial.append((loc, state, predicate))
        self.weights = np.asarray([x[0].get('weight', 1.) for x in self.initial], dtype=float)
        self.weights /= self.weights.sum()
        self.hashes = {p.name: digest(p) for p in [self.path, self.path.with_name('model.xml'),
                                                  self.folder/'metadata.yaml']}

    def sample_initial(self, rng):
        loc, fn, pred = self.initial[int(rng.choice(len(self.initial), p=self.weights))]
        draws = []
        for spec in loc.get('draws', {}).values():
            if 'uniform' in spec:
                draws.append(float(rng.uniform(*spec['uniform'])))
            else:
                draws.append(float(rng.choice(spec['choice'])))
        mode = loc['mode_by']['true' if pred.holds(draws) else 'false'] if pred else loc['mode']
        y = np.asarray(fn(*draws), dtype=float)
        env = dict(zip(loc.get('draws', {}), draws))
        initial_inputs = {v: float(function([expr], list(env))(*env.values())[0])
                          for v, expr in loc.get('input_start', {}).items()}
        return str(mode), y, initial_inputs


@lru_cache(maxsize=64)
def get_recipe(sid, root=None):
    return Recipe(sid, root)
