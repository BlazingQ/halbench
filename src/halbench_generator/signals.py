"""Sample input signals and process disturbances for evaluation during integration."""
from __future__ import annotations

import math
import numpy as np


class Signal:
    def __init__(self, kind, parameters, times=None, values=None):
        self.kind, self.parameters = kind, parameters
        self.times = np.asarray(times if times is not None else [], dtype=float)
        self.values = np.asarray(values if values is not None else [], dtype=float)

    @property
    def breakpoints(self):
        return self.times[1:] if self.kind in ('zoh', 'linear') else np.empty(0)

    def __call__(self, t):
        if self.kind in ('zoh', 'linear'):
            i = min(len(self.times)-1, max(0, int(np.searchsorted(self.times, t, side='right')-1)))
            if self.kind == 'linear' and i+1 < len(self.times):
                a = (t-self.times[i])/(self.times[i+1]-self.times[i])
                return self.values[i]*(1-a) + self.values[i+1]*a
            return self.values[i]
        p = self.parameters
        return p['offset'] + p['amplitude']*math.sin(p['frequency']*t+p['phase'])

    def describe(self):
        return dict(kind=self.kind, parameters=self.parameters,
                    times=self.times.tolist(), values=self.values.tolist())


def make_inputs(spec, rng, duration, initial_inputs):
    result = []
    for name, cfg in spec.items():
        kind = cfg['kind']
        if kind == 'sine':
            p = {k: float(rng.uniform(*cfg[k])) for k in ('offset', 'amplitude', 'frequency', 'phase')}
            result.append(Signal(kind, p))
            continue
        step = cfg['interval']
        times = np.arange(int(math.ceil(duration/step))+1)*step
        if cfg.get('alternating'):
            start = initial_inputs[name]
            values = (start + np.arange(len(times))) % 2
        elif 'increment' in cfg:
            lo, hi = cfg['bounds']
            values = [float(rng.uniform(lo, hi))]
            for _ in times[1:]:
                # Sample uniformly from the increment interval intersected with the input bounds.
                values.append(float(rng.uniform(max(lo, values[-1]+cfg['increment'][0]),
                                                min(hi, values[-1]+cfg['increment'][1]))))
        else:
            values = rng.uniform(*cfg['bounds'], size=len(times))
        result.append(Signal(kind, dict(interval=step), times, values))
    return result


class Disturbance:
    def __init__(self, dimension, channels, scales, interval, duration, strength, rng):
        self.interval = interval
        self.times = np.arange(int(math.ceil(duration/interval))+1)*interval if strength else np.array([0.])
        self.values = np.zeros((len(self.times), dimension))
        if strength:
            self.values[:, channels] = rng.normal(size=(len(self.times), len(channels))) * np.asarray(scales)*strength

    def __call__(self, t):
        return self.values[min(len(self.values)-1, max(0, int(np.searchsorted(self.times, t, side='right')-1)))]
