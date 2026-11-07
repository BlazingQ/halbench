"""Generate a collection using each system's default horizon and sampling interval."""
import json
from pathlib import Path
import tempfile

import yaml

from .dataset import generate_dataset, output_path, validate_request
from .engine import GenerationError
from .recipes import systems_root


def generate_collection(*, n_train, n_test=0, seed=0, process_noise=0.,
                        measurement_noise=0., output_dir, workers='auto'):
    """Generate one dataset per synthetic system and return the collection directory."""
    validate_request(n_train, n_test, seed, process_noise, measurement_noise, workers)
    destination = output_path(output_dir)
    systems = [p.parent.name for p in sorted(systems_root().glob('*/metadata.yaml'))
               if yaml.safe_load(p.read_text())['type'] == 'synthetic']
    if not systems:
        raise ValueError('The configured catalog contains no synthetic systems')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.partial-', dir=destination.parent))
    record = dict(status='running', collection_id='all', requested_systems=systems,
                  request=dict(n_train=n_train, n_test=n_test, seed=seed,
                               process_noise=process_noise, measurement_noise=measurement_noise),
                  systems=[])

    try:
        for sid in systems:
            generate_dataset(sid, n_train=n_train, n_test=n_test, seed=seed,
                             process_noise=process_noise, measurement_noise=measurement_noise,
                             output_dir=stage/sid, workers=workers)
            record['systems'].append(sid)
        if destination.exists():
            raise FileExistsError(f'Output appeared during generation: {destination}')
        stage.rename(destination)
        return destination
    except Exception as exc:
        record['status'] = 'failed'
        record['failure'] = dict(error=str(exc), context=getattr(exc, 'context', {}))
        (stage/'diagnostics.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        raise GenerationError(getattr(exc, 'category', 'collection'),
                              f'{exc}; collection diagnostics: {stage}', staging_dir=str(stage)) from exc
