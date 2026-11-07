"""Read bundled or user-generated datasets as pandas DataFrames."""
from pathlib import Path
import json

import pandas as pd
import yaml

from ._paths import datasets_root, systems_root


def list_datasets():
    """Return system identifiers available in the configured datasets directory."""
    return sorted(d.name for d in datasets_root().iterdir()
                  if d.is_dir() and any((d/p).is_file() for p in
                     ('manifest.json', 'traces/manifest.json', 'metadata.yaml')))


def _columns(metadata):
    variables = metadata['variables']
    states, inputs = variables['states'], variables.get('inputs', [])
    if (len(states) != metadata['dim']['output'] or len(inputs) != metadata['dim']['input']
            or variables.get('events')):
        raise ValueError('Observed columns must exactly match input/output dimensions')
    return ['t'] + states + inputs


def _load(folder, split, expected=None):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    folder = Path(folder).expanduser().resolve()
    # Dataset roots and their trace batch directories are both accepted.
    if (folder/'traces/manifest.json').is_file():
        batch, dataset = folder/'traces', folder
    elif (folder/'manifest.json').is_file():
        batch = folder
        dataset = folder.parent if (folder.parent/'metadata.yaml').is_file() else folder
    elif (folder/'collection.json').is_file():
        raise ValueError('Select a system subdirectory within the collection')
    elif (folder/'metadata.yaml').is_file():
        if not (folder/'traces').is_dir():
            raise FileNotFoundError(f'Select a dataset directory containing manifest.json, train/ and test/: {folder}')
        batch, dataset = folder/'traces', folder
    elif any(folder.glob('*/manifest.json')):
        raise ValueError('Select a system subdirectory within the collection')
    else:
        raise FileNotFoundError(f'Dataset manifest not found in: {folder}')
    if (dataset/'metadata.yaml').exists():
        expected = _columns(yaml.safe_load((dataset/'metadata.yaml').read_text()))
    manifest_path = batch/'manifest.json'
    rows = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') not in ('halbench-generated-v1', 'halbench-observed-v1',
                                           'halbench-generated-v2', 'halbench-observed-v2',
                                           'halbench-generated-v3', 'halbench-observed-v3'):
            raise ValueError('Unsupported dataset manifest schema')
        if manifest.get('status') != 'complete':
            raise ValueError('A complete dataset batch is required')
        if expected is not None and expected != manifest['columns']:
            raise ValueError('Manifest columns differ from system metadata')
        if 'variables' in manifest:
            variables = manifest['variables']
            if ['t'] + variables['states'] + variables.get('inputs', []) != manifest['columns']:
                raise ValueError('Manifest columns differ from its variable definitions')
        expected = manifest['columns']
        rows = [r for r in manifest['traces'] if r['split'] == split]
    if rows is None:
        rows = [dict(path=str(p.relative_to(batch))) for p in sorted((batch/split).glob('*.csv*'))]
    traces = {}
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid trace path in manifest')
        frame = pd.read_csv(batch/relative)
        if expected is not None and list(frame.columns) != expected:
            raise ValueError(f'Trace columns differ from declared input/output dimensions: {relative}')
        if 'samples' in row and len(frame) != row['samples']:
            raise ValueError(f'Trace sample count differs from manifest: {relative}')
        name = relative.name.split('.csv')[0]
        if name in traces:
            raise ValueError(f'Duplicate trace name: {name}')
        traces[name] = frame
    return traces


def load(system_id=None, split='train', *, path=None):
    """Load a bundled system by ID or a dataset at an explicit path.

    Examples: load('two_tank'); load(path='generated/tank_42').
    The path identifies the directory containing manifest.json, train/ and test/.
    """
    if path is not None:
        if system_id is not None:
            raise ValueError('Specify a system_id or path, one at a time')
        return _load(path, split)
    if not isinstance(system_id, str) or Path(system_id).name != system_id or system_id in ('.', '..'):
        raise ValueError('Select a bundled system_id or specify path= for your dataset')
    metadata_path = systems_root()/system_id/'metadata.yaml'
    expected = _columns(yaml.safe_load(metadata_path.read_text())) if metadata_path.is_file() else None
    return _load(datasets_root()/system_id, split, expected)


def load_generated(output_dir, split='train'):
    """Read generated data using the same path rules as load(path=...)."""
    return load(path=output_dir, split=split)
