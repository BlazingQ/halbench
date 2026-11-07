"""Common continuous execution, boundary rearming, and right-side sampling."""
from __future__ import annotations

from dataclasses import dataclass
import ast
import math
import time

import numpy as np
from scipy.integrate import solve_ivp


class GenerationError(RuntimeError):
    def __init__(self, category, message, **context):
        super().__init__(message)
        self.category, self.context = category, context


@dataclass(frozen=True)
class Settings:
    method: str = 'DOP853'
    rtol: float = 1e-11
    atol_factor: float = 1e-13
    max_step_factor: float = 1.
    boundary_tol: float = 2e-11
    max_jumps: int = 200000
    max_rhs: int = 5000000
    timeout: float = 300.


@dataclass
class Result:
    times: np.ndarray
    states: np.ndarray
    modes: list
    jumps: list
    rhs_calls: int
    node_min: np.ndarray
    node_max: np.ndarray
    boundary_suppressions: int


def sample_grid(duration, period):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(period) or period <= 0:
        raise ValueError('duration and sample_period must be finite and positive')
    n = int(math.floor(duration/period))
    if n > 10000000:
        raise ValueError('More than ten million samples per trace')
    grid = np.arange(n+1, dtype=float)*period
    tol = 32*np.finfo(float).eps*max(1., duration)
    if abs(grid[-1]-duration) <= tol:
        grid[-1] = duration
    else:
        grid = np.append(grid, duration)
    return grid


def simulate(model, mode, initial, signals, disturbance, duration, period, scales,
             max_step, settings=None):
    cfg = settings or Settings()
    grid = sample_grid(duration, period)
    output = np.full((len(grid), len(initial)), np.nan)
    sample_modes = [None]*len(grid)
    y = np.asarray(initial, dtype=float).copy()
    if mode not in model.modes or not np.isfinite(y).all() or len(y) != len(model.states):
        raise GenerationError('configuration', 'Invalid complete initial state/mode')
    atol = np.asarray(scales)*cfg.atol_factor
    if atol.shape != y.shape or np.any(atol <= 0):
        raise ValueError('Positive per-state numerical scales required')
    started = time.monotonic()
    t, count, sample_index, suppressions = 0., 0, 0, 0
    jumps = []
    node_min, node_max = y.copy(), y.copy()
    time_tol = 64*np.finfo(float).eps*max(1., duration)
    breakpoints = sorted(set(float(v) for s in signals for v in s.breakpoints if 0 < v < duration)
                         | set(float(v) for v in disturbance.times[1:] if 0 < v < duration)) + [duration]
    break_index = 0
    tick = 1
    initial_clock = y[model.clock[0]] if model.clock else 0.

    def inputs(tt):
        return np.array([s(tt) for s in signals])

    def fail(category, message):
        raise GenerationError(category, message, time=float(t), mode=mode, state=y.tolist(),
                              rhs_calls=count, jumps=len(jumps))

    def checked_rhs(q, tt, yy, uu, noise):
        nonlocal count
        count += 1
        if count > cfg.max_rhs or (count % 512 == 0 and time.monotonic()-started > cfg.timeout):
            fail('execution_budget', 'RHS or wall-time budget exhausted')
        try:
            result = model.rhs(q, yy, uu) + noise
        except (ArithmeticError, ValueError, TypeError) as exc:
            fail('simulation', f'Undefined dynamics: {exc}')
        if not np.isfinite(result).all():
            fail('simulation', 'Nonfinite dynamics')
        return result

    def tolerance(atom, args):
        return cfg.boundary_tol*atom.scale(*args)[0]

    def holds(predicate, args, *, limit=False, q=None, tt=None, yy=None):
        rates = None
        if limit:
            rates = [rate(a, q, tt, yy) for a in predicate.atoms]
        for i, a in enumerate(predicate.atoms):
            m, tol = a.margin(args), tolerance(a, args)
            if a.op == '==':
                ok = abs(m) <= tol
            elif a.op in ('<', '>'):
                ok = m > tol or (limit and abs(m) <= tol and rates[i] > 0)
            else:
                ok = m >= -tol
            if not ok:
                return False
        return True

    def rate(atom, q, tt, yy):
        h = 1e-7*min(1., max_step)
        f = model.rhs(q, yy, inputs(tt)) + disturbance(tt)
        a = atom.margin(model.args(yy+h*f, inputs(tt+h)))
        b = atom.margin(model.args(yy-h*f, inputs(max(0., tt-h))))
        return (a-b)/(2*h)

    def compatible_return(previous, edge, tt, yy):
        if (previous is None or not previous.identity or not edge.identity
                or edge.target != previous.source
                or len(edge.guard.atoms) != 1 or len(previous.guard.atoms) != 1):
            return False
        a, b = edge.guard.atoms[0], previous.guard.atoms[0]
        if a.op == '==' or b.op == '==' or a.key[:2] != b.key[:2] or a.key[2] == b.key[2]:
            return False
        args = model.args(yy, inputs(tt))
        return (abs(a.margin(args)) <= tolerance(a, args)
                and rate(a, previous.source, tt, yy) < 0
                and rate(a, edge.source, tt, yy) < 0)

    def settle(previous=None, from_event=False):
        nonlocal mode, y, suppressions, node_min, node_max
        seen = set()
        while True:
            args = model.args(y, inputs(t))
            candidates = []
            for edge in model.outgoing[mode]:
                if holds(edge.guard, args):
                    if compatible_return(previous, edge, t, y):
                        suppressions += 1
                    else:
                        candidates.append(edge)
            # Strict physical crossings have no first open-set instant. Resolve
            # their entering boundary only if no exact (closed-priority) edge wins.
            if not candidates and from_event:
                candidates = [e for e in model.outgoing[mode]
                              if holds(e.guard, args, limit=True, q=mode, tt=t, yy=y)
                              and not compatible_return(previous, e, t, y)]
            if len(candidates) > 1:
                fail('simulation', f'Multiple urgent transitions enabled: {[e.index for e in candidates]}')
            if not candidates:
                inv = model.modes[mode][1]
                if not holds(inv, args):
                    fail('simulation', 'Initial/post-reset invariant violation')
                return
            signature = (mode, y.tobytes())
            if signature in seen:
                fail('simulation', 'Unresolved zero-time cycle')
            seen.add(signature)
            if len(jumps) >= cfg.max_jumps:
                fail('execution_budget', 'Transition budget exhausted')
            edge = candidates[0]
            before = y.copy()
            y = model.reset(edge, y, inputs(t))
            if not np.isfinite(y).all():
                fail('simulation', 'Nonfinite reset')
            jumps.append(dict(t=float(t), edge=edge.index, source=mode, target=edge.target,
                              before=before.tolist(), after=y.tolist()))
            node_min, node_max = np.minimum(node_min, y), np.maximum(node_max, y)
            mode, previous = edge.target, edge
            from_event = True

    settle(from_event=True)
    while True:
        # Assign boundary samples only after inputs/reset closure (including T).
        while sample_index < len(grid) and abs(grid[sample_index]-t) <= time_tol:
            output[sample_index], sample_modes[sample_index] = y, mode
            sample_index += 1
        if t >= duration-time_tol:
            break
        if time.monotonic()-started > cfg.timeout:
            fail('execution_budget', 'Wall-time budget exhausted')
        while breakpoints[break_index] <= t+time_tol and break_index+1 < len(breakpoints):
            break_index += 1
        stop = breakpoints[break_index]
        if model.clock:
            period_clock = model.clock[1]
            while tick*period_clock-initial_clock <= t+time_tol:
                tick += 1
            stop = min(stop, tick*period_clock-initial_clock)
        noise = disturbance(t)
        # Use the left input limit for the final RK stage, then update inputs
        # before resolving transitions at the breakpoint.
        def rhs(tt, yy):
            stage_t = min(tt, np.nextafter(stop, -np.inf))
            return checked_rhs(mode, tt, yy, inputs(stage_t), noise)

        events, atom_records = [], []
        if not model.clock:
            unique = {}
            for e in model.outgoing[mode]:
                for a in e.guard.atoms:
                    unique[(a.key, 'eq' if a.op == '==' else 'in')] = (a, 0 if a.op == '==' else 1)
            for a in model.modes[mode][1].atoms:
                unique[(a.key, 'out')] = (a, -1)
            start_args = model.args(y, inputs(t))
            for a, direction in unique.values():
                m = a.margin(start_args)
                scale = tolerance(a, start_args)
                d = rate(a, mode, t, y) if abs(m) <= scale else 0.
                if abs(m) <= scale and abs(d) < 1e-14:
                    # A static equality in a priority conjunction (e.g. y=0
                    # with y'=0) is a condition, not a new time event.
                    # The complete guard is still evaluated at every boundary.
                    dependencies = {n.id for text in (a.left, a.right)
                                    for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)}
                    dependencies -= {'sin', 'cos', 'sqrt', 'abs'}
                    if dependencies <= model.static_states[mode]:
                        continue
                    fail('simulation', 'Unsupported tangential event surface; no continuation policy applied')
                start_value = math.copysign(max(scale, np.finfo(float).eps), d) if abs(m) <= scale else m
                t_start = t
                def event(tt, yy, atom=a, start_value=start_value, t_start=t_start):
                    if tt == t_start:
                        return start_value
                    return atom.margin(model.args(yy, inputs(min(tt, np.nextafter(stop, -np.inf)))))
                event.direction, event.terminal = direction, True
                events.append(event)
                atom_records.append(a)
        try:
            sol = solve_ivp(rhs, (t, stop), y, method=cfg.method, rtol=cfg.rtol,
                            atol=atol, max_step=max_step*cfg.max_step_factor,
                            dense_output=True, events=events or None)
        except GenerationError:
            raise
        except (ArithmeticError, ValueError) as exc:
            fail('simulation', f'Integrator error: {exc}')
        if not sol.success or not np.isfinite(sol.y).all():
            fail('simulation', f'Integrator failed: {sol.message}')
        end = float(sol.t[-1])
        if end <= t:
            fail('simulation', 'Event restart did not advance time')
        for tt, yy in zip(sol.t, sol.y.T):
            if not holds(model.modes[mode][1], model.args(yy, inputs(min(tt, np.nextafter(stop, -np.inf))))):
                fail('simulation', 'Continuous segment left invariant without an enabled transition')
        node_min = np.minimum(node_min, sol.y.min(axis=1))
        node_max = np.maximum(node_max, sol.y.max(axis=1))
        # Samples strictly inside this continuous piece never cross a reset.
        end_index = int(np.searchsorted(grid, end-time_tol, side='left'))
        if end_index > sample_index:
            output[sample_index:end_index] = sol.sol(grid[sample_index:end_index]).T
            sample_modes[sample_index:end_index] = [mode]*(end_index-sample_index)
            sample_index = end_index
        t, y = end, sol.y[:, -1].copy()
        settle(from_event=True)
        # Invariant boundary with an outward vector field and no jump is an
        # undefined continuation, not permission to integrate outside it.
        for a in model.modes[mode][1].atoms:
            args = model.args(y, inputs(t))
            if abs(a.margin(args)) <= tolerance(a, args) and rate(a, mode, t, y) < 0:
                fail('simulation', 'Invariant boundary has no legal continuation')
    if sample_index != len(grid) or not np.isfinite(output).all():
        fail('output', 'Incomplete or nonfinite sampled output')
    return Result(grid, output, sample_modes, jumps, count, node_min, node_max, suppressions)
