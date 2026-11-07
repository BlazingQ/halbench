"""Locate system definitions and bundled datasets independently."""
import os
from pathlib import Path


def datasets_root():
    configured = os.environ.get('HALBENCH_DATASETS')
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]/'datasets'


def systems_root():
    configured = os.environ.get('HALBENCH_SYSTEMS')
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]/'systems'
