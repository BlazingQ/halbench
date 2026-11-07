"""Parse the flat urgent SX models used by the benchmark systems.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


FUNCTIONS = dict(sin=math.sin, cos=math.cos, sqrt=math.sqrt, abs=abs)


def expression(text, names):
    tree = ast.parse(str(text).replace('^', '**'), mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name,
               ast.Load, ast.Constant, ast.Call)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f'Unsupported arithmetic: {text}')
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)):
            raise ValueError(f'Not a numeric literal: {text}')
        if isinstance(node, ast.Name) and node.id not in set(names) | FUNCTIONS.keys():
            raise ValueError(f'Unknown symbol {node.id}')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in FUNCTIONS or len(node.args) != 1 or node.keywords):
            raise ValueError(f'Unsupported function: {text}')
    return ast.unparse(tree.body)


def function(texts, names):
    """Compile validated expressions into a callable for integration."""
    bodies = [expression(t, names) for t in texts]
    body = '(' + ','.join(bodies) + ',)' if bodies else '()'
    return eval(f'lambda {",".join(names)}: {body}', {'__builtins__': {}, **FUNCTIONS})


@dataclass
class Atom:
    left: str
    op: str
    right: str
    fn: object
    key: tuple
    scale: object

    def margin(self, args):
        return self.fn(*args)[0]


class Predicate:
    def __init__(self, text, names):
        self.text = text.strip()
        self.atoms = []
        if self.text == 'true':
            return
        for term in self.text.split('&'):
            match = re.fullmatch(r'\s*(.*?)\s*(<=|>=|==|<|>)\s*(.*?)\s*', term)
            if not match:
                raise ValueError(f'Unsupported predicate: {text}')
            left, op, right = match.groups()
            left, right = expression(left, names), expression(right, names)
            sign = -1 if op in ('<', '<=') else 1
            margin = f'({left})-({right})' if sign == 1 else f'({right})-({left})'
            key = (left, right, sign)
            self.atoms.append(Atom(left, op, right, function([margin], names), key,
                                   function([f'1+abs({left})+abs({right})'], names)))

    def holds(self, args, tol=0., *, strict_boundary=False, rates=None):
        for i, atom in enumerate(self.atoms):
            m = atom.margin(args)
            if not math.isfinite(m):
                raise ArithmeticError('Nonfinite predicate')
            if atom.op == '==':
                ok = abs(m) <= tol
            elif atom.op in ('<', '>'):
                ok = m > tol or (strict_boundary and abs(m) <= tol and rates[i] > 0)
            else:
                ok = m >= -tol
            if not ok:
                return False
        return True


@dataclass
class Edge:
    index: int
    source: str
    target: str
    guard: Predicate
    reset: object
    identity: bool


class Model:
    def __init__(self, path: Path, states, inputs):
        self.states, self.inputs = list(states), list(inputs)
        names = self.states + self.inputs
        if len(set(names)) != len(names) or not all(v.isidentifier() for v in names):
            raise ValueError('Invalid or repeated SX variable names')
        tree = ET.parse(path)
        for node in tree.iter():
            node.tag = node.tag.split('}')[-1]
        root = tree.getroot()
        components = root.findall('component')
        if root.tag != 'sspaceex' or root.get('math') != 'SpaceEx' or len(components) != 1:
            raise ValueError('Only one flat SX component is supported')
        comp = components[0]
        if comp.findall('bind') or set(n.tag for n in comp) - {'param', 'location', 'transition', 'note'}:
            raise ValueError('Unsupported component structure')
        params = comp.findall('param')
        if len(params) != len(names) or {p.get('name') for p in params} != set(names):
            raise ValueError('Recipe/SX variable mismatch')
        for p in params:
            if (p.get('type') != 'real' or p.get('dynamics') != 'any'
                    or p.get('controlled') != ('false' if p.get('name') in inputs else 'true')):
                raise ValueError('Unsupported SX parameter declaration')
        self.modes, self.edges, self.static_states = {}, [], {}

        def assignments(text):
            result = {}
            for term in text.split('&'):
                match = re.fullmatch(r"\s*(\w+)'\s*==\s*(.*?)\s*", term)
                if not match or match[1] in result:
                    raise ValueError(f'Invalid assignment: {text}')
                result[match[1]] = expression(match[2], names)
            if set(result) != set(states):
                raise ValueError('Explicit complete flow/reset required')
            return result

        for node in comp.findall('location'):
            if (set(n.tag for n in node) - {'flow', 'invariant', 'note'}
                    or node.find('flow') is None or node.find('invariant') is None
                    or node.get('id') in self.modes):
                raise ValueError('Unsupported or repeated location')
            flow = assignments(node.findtext('flow'))
            self.static_states[node.get('id')] = {v for v in states if flow[v] in ('0', '0.0')}
            self.modes[node.get('id')] = (
                function([flow[v] for v in states], names),
                Predicate(node.findtext('invariant'), names))
        for i, node in enumerate(comp.findall('transition')):
            if node.get('asap') != 'true' or set(n.tag for n in node) != {'guard', 'assignment'}:
                raise ValueError('Only explicit urgent transitions supported')
            reset = assignments(node.findtext('assignment'))
            edge = Edge(i, node.get('source'), node.get('target'),
                        Predicate(node.findtext('guard'), names),
                        function([reset[v] for v in states], names),
                        all(reset[v] == v for v in states))
            if edge.source not in self.modes or edge.target not in self.modes:
                raise ValueError('Unknown transition location')
            self.edges.append(edge)
        self.outgoing = {q: [e for e in self.edges if e.source == q] for q in self.modes}
        # An exact deadline optimization inferred from SX, not a dataset switch.
        # Every edge is clock-gated and resets the same unit-rate clock to zero.
        self.clock = None
        for v in states:
            candidates = []
            if not self.edges:
                continue
            for edge in self.edges:
                atoms = [a for a in edge.guard.atoms if a.left == v and a.op == '>='
                         and not any(isinstance(n, ast.Name) for n in ast.walk(ast.parse(a.right)))]
                if len(atoms) != 1:
                    break
                candidates.append(float(function([atoms[0].right], [])()[0]))
            if len(candidates) != len(self.edges) or len(set(candidates)) != 1 or candidates[0] <= 0:
                continue
            i = states.index(v)
            # Structural checks on assignments/flows, not probes of dynamics.
            flows = [assignments(n.findtext('flow'))[v] for n in comp.findall('location')]
            resets = [assignments(n.findtext('assignment'))[v] for n in comp.findall('transition')]
            clock_invariants = all(all(a.left == v and a.op == '<='
                and not any(isinstance(n, ast.Name) for n in ast.walk(ast.parse(a.right)))
                and function([a.right], [])()[0] == candidates[0] for a in inv.atoms)
                for _, inv in self.modes.values())
            if (clock_invariants and all(f in ('1', '1.0') for f in flows)
                    and all(r in ('0', '0.0') for r in resets)):
                self.clock = (i, candidates[0])
                break

    def args(self, y, u):
        return (*y, *u)

    def rhs(self, mode, y, u):
        return np.asarray(self.modes[mode][0](*self.args(y, u)), dtype=float)

    def reset(self, edge, y, u):
        return np.asarray(edge.reset(*self.args(y, u)), dtype=float)
