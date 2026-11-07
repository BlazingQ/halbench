"""Check the released dataset contract and exercise every generation recipe."""
import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import halbench_loader as loader
from halbench_generator import generate_dataset, generate_collection
from halbench_generator.recipes import Recipe
from halbench_loader._paths import datasets_root, systems_root


class ReleaseTests(unittest.TestCase):
    def test_catalog_models_and_trace_integrity(self):
        root = datasets_root()
        with (root.parent/'catalog.csv').open() as handle:
            catalog = {row['system_id']: row for row in csv.DictReader(handle)}
        self.assertEqual(set(catalog), set(loader.list_datasets()))
        self.assertEqual(len(catalog), 25)
        synthetic = 0
        for sid in loader.list_datasets():
            with self.subTest(system=sid):
                folder = root/sid
                system = systems_root()/sid
                meta = yaml.safe_load((system/'metadata.yaml').read_text())
                manifest = json.loads((folder/'manifest.json').read_text())
                for field in ('paper_citekey', 'traces', 'duration', 'sampling_period',
                              'initial_origin', 'data_origin', 'data_license',
                              'dynamics_detail', 'guard_detail', 'reset_detail'):
                    self.assertNotIn(field, meta)
                self.assertFalse((folder/'README.md').exists())
                columns = ['t'] + meta['variables']['states'] + meta['variables']['inputs']
                self.assertEqual(len(columns), 1 + sum(meta['dim'].values()))
                self.assertEqual(manifest['columns'], columns)
                self.assertEqual(manifest['status'], 'complete')
                if meta['type'] == 'synthetic':
                    synthetic += 1
                    recipe = Recipe(sid)
                    self.assertEqual(len(recipe.model.modes), meta['num_modes'])
                    self.assertEqual((manifest['counts']['train'], manifest['counts']['test']), (256, 64))
                    self.assertEqual(manifest['request']['process_noise'], 0)
                    self.assertEqual(manifest['request']['measurement_noise'], 0)
                    for name, value in manifest['system_hashes'].items():
                        self.assertEqual(hashlib.sha256((system/name).read_bytes()).hexdigest(), value)
                else:
                    self.assertFalse((system/'model.xml').exists())
                    self.assertFalse((system/'generation.yaml').exists())
                files = set()
                for split in ['train', 'test']:
                    paths = list((folder/split).glob('*.csv*'))
                    self.assertEqual(len(paths), manifest['counts'][split])
                    files.update(path.relative_to(folder).as_posix() for path in paths)
                self.assertEqual(files, {row['path'] for row in manifest['traces']})
                split_hashes = {'train': set(), 'test': set()}
                for row in manifest['traces']:
                    path = folder/row['path']
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'])
                    with gzip.open(path, 'rt', newline='') as handle:
                        self.assertEqual(next(csv.reader(handle)), columns)
                    split_hashes[row['split']].add(row['sha256'])
                if meta['type'] == 'synthetic':
                    self.assertFalse(split_hashes['train'] & split_hashes['test'])
        self.assertEqual(synthetic, 22)

    def test_all_systems_and_four_noise_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            for sid in loader.list_datasets():
                if not (systems_root()/sid/'generation.yaml').exists():
                    continue
                recipe = Recipe(sid)
                for process, measurement in [(0, 0), (.1, 0), (0, .01), (.1, .01)]:
                    with self.subTest(system=sid, process=process, measurement=measurement):
                        output = generate_dataset(sid, n_train=1, n_test=1,
                            duration=min(1., recipe.cfg['defaults']['duration']), seed=20261106,
                            process_noise=process, measurement_noise=measurement, workers=1,
                            output_dir=Path(temp)/sid/f'{process}_{measurement}')
                        manifest = json.loads((output/'manifest.json').read_text())
                        self.assertEqual(manifest['status'], 'complete')
                        for split in ['train', 'test']:
                            traces = loader.load_generated(output, split)
                            self.assertEqual(len(traces), 1)
                            frame = next(iter(traces.values()))
                            self.assertTrue(np.isfinite(frame.to_numpy()).all())
                            self.assertEqual(list(frame.columns), manifest['columns'])

    def test_generated_dataset_is_portable_and_has_the_bundled_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            output = generate_dataset('two_tank', n_train=1, n_test=1, duration=.5,
                                      seed=9, workers=1, output_dir=Path(temp)/'tank')
            self.assertEqual({p.name for p in output.iterdir()}, {'manifest.json', 'train', 'test'})
            original = loader.load(path=output, split='test')
            detached = Path(temp)/'detached'
            shutil.copytree(output, detached)
            with patch.dict(os.environ, {'HALBENCH_DATASETS': str(Path(temp)/'absent_data'),
                                         'HALBENCH_SYSTEMS': str(Path(temp)/'absent_systems')}):
                frames = loader.load(path=detached, split='test')
            np.testing.assert_array_equal(original['trace_000000'], frames['trace_000000'])

    def test_collection_uses_each_system_defaults_and_records_failures(self):
        source = systems_root()
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for sid, duration in [('two_tank', .5), ('simple_heating_system', .7)]:
                folder = temp/'catalog'/sid
                folder.mkdir(parents=True)
                for name in ('metadata.yaml', 'generation.yaml', 'model.xml'):
                    shutil.copy2(source/sid/name, folder/name)
                config = yaml.safe_load((folder/'generation.yaml').read_text())
                config['defaults'].update(duration=duration, sample_period=.1)
                (folder/'generation.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            with patch.dict(os.environ, {'HALBENCH_SYSTEMS': str(temp/'catalog')}):
                output = generate_collection(n_train=1, n_test=1, seed=19, workers=1,
                                             output_dir=temp/'all')
                self.assertEqual({p.name for p in output.iterdir()}, {'two_tank', 'simple_heating_system'})
                for sid, duration in [('two_tank', .5), ('simple_heating_system', .7)]:
                    manifest = json.loads((output/sid/'manifest.json').read_text())
                    self.assertEqual(manifest['request']['duration'], duration)
                    self.assertEqual(manifest['counts'], {'train': 1, 'test': 1})
                    self.assertEqual(len(loader.load(path=output/sid)), 1)
                with self.assertRaises(ValueError):
                    loader.load(path=output)
                with patch('halbench_generator.collection.generate_dataset', side_effect=RuntimeError('fixture failure')):
                    from halbench_generator import GenerationError
                    with self.assertRaises(GenerationError) as caught:
                        generate_collection(n_train=1, output_dir=temp/'failed', workers=1)
                    stage = Path(caught.exception.context['staging_dir'])
                    self.assertEqual(json.loads((stage/'diagnostics.json').read_text())['status'], 'failed')
                    self.assertFalse((temp/'failed').exists())

    def test_collection_rejects_single_system_time_override_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'all'
            command = subprocess.run([sys.executable, '-m', 'halbench_generator', 'all',
                                      '--n-train', '1', '--duration', '1', '--output-dir', str(output)],
                                     capture_output=True, text=True)
            self.assertNotEqual(command.returncode, 0)
            self.assertIn('single system', command.stderr)
            self.assertFalse(output.exists())

    def test_shared_environment_discovery(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'HALBENCH_DATASETS': temp, 'HALBENCH_SYSTEMS': temp}):
                self.assertEqual(datasets_root(), Path(temp).resolve())
                self.assertEqual(loader.list_datasets(), [])
                with self.assertRaises(FileNotFoundError):
                    Recipe('two_tank')

    def test_cli_help_and_observation_only_error(self):
        for flag in ['-h', '--help']:
            command = subprocess.run([sys.executable, '-m', 'halbench_generator', flag],
                                     capture_output=True, text=True, check=True)
            self.assertIn('--output-dir', command.stdout)
            self.assertIn('manifest.json', command.stdout)
        with self.assertRaisesRegex(ValueError, 'real-world'):
            Recipe('ecg_signals')


if __name__ == '__main__':
    unittest.main()
