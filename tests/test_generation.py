"""Behavioral checks for execution semantics and reproducible public batches."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from halbench_generator import generate_dataset, GenerationError
from halbench_generator.dataset import generate_trace
from halbench_generator.engine import simulate, Settings
from halbench_generator.signals import Signal, Disturbance
from halbench_generator.sx import Model


def toy(folder, states, modes, edges, inputs=()):
    root=ET.Element('sspaceex',math='SpaceEx')
    comp=ET.SubElement(root,'component',id='plant')
    for v in [*states,*inputs]:
        ET.SubElement(comp,'param',name=v,type='real',dynamics='any',controlled='false' if v in inputs else 'true')
    for q,(flow,inv) in modes.items():
        loc=ET.SubElement(comp,'location',id=q)
        ET.SubElement(loc,'flow').text=flow
        ET.SubElement(loc,'invariant').text=inv
    for source,target,guard,reset in edges:
        e=ET.SubElement(comp,'transition',source=source,target=target,asap='true')
        ET.SubElement(e,'guard').text=guard
        ET.SubElement(e,'assignment').text=reset
    path=Path(folder)/'model.xml';ET.ElementTree(root).write(path)
    return Model(path,states,list(inputs))


def run(model,initial,*,duration=2,period=.25,signals=(),max_step=2):
    noise=Disturbance(len(initial),[],[],1,duration,0,np.random.default_rng(0))
    return simulate(model,'A',initial,list(signals),noise,duration,period,[1]*len(initial),max_step)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def model(self,states,modes,edges,inputs=()):
        return toy(self.tmp.name,states,modes,edges,inputs)

    def test_narrow_guard_entry_and_final_reset(self):
        m=self.model(['x'],{'A':("x' == 2",'true'),'B':("x' == 0",'true')},
            [('A','B','x >= .99 & x <= 1.01',"x' == 7")])
        r=run(m,[0],duration=.495,period=.1,max_step=100)
        self.assertAlmostEqual(r.jumps[0]['t'],.495,places=12)
        self.assertEqual(r.states[-1,0],7)
        self.assertEqual(r.times[-1],.495)

    def test_simultaneous_reset_and_initial_finite_chain(self):
        m=self.model(['x','y'],{q:("x' == 0 & y' == 0",'true') for q in 'ABC'},
            [('A','B','x >= 1',"x' == y & y' == x"),
             ('B','C','y >= 1',"x' == x+10 & y' == y")])
        r=run(m,[1,2])
        np.testing.assert_array_equal(r.states[0],[12,1])
        self.assertEqual([j['t'] for j in r.jumps],[0,0])

    def test_shared_boundary_no_displacement_and_no_immediate_return(self):
        m=self.model(['x'],{'A':("x' == -1",'true'),'B':("x' == -2",'true')},
            [('A','B','x <= 0',"x' == x"),('B','A','x >= 0',"x' == x")])
        r=run(m,[1])
        self.assertEqual(len(r.jumps),1)
        self.assertAlmostEqual(r.jumps[0]['t'],1,places=12)
        self.assertAlmostEqual(r.jumps[0]['before'][0],0,places=12)
        self.assertAlmostEqual(r.states[-1,0],-2,places=12)

    def test_input_breakpoint_precedes_reset_and_sampling(self):
        m=self.model(['x'],{'A':("x' == u",'true'),'B':("x' == 0",'true')},
            [('A','B','u >= 2',"x' == u")],['u'])
        signal=Signal('zoh',{},[0,.5,2],[1,2,2])
        r=run(m,[0],signals=[signal],duration=1)
        self.assertEqual(r.jumps[0]['t'],.5)
        self.assertAlmostEqual(r.jumps[0]['before'][0],.5,places=12)
        self.assertEqual(r.states[2,0],2)

    def test_ball_contact_uses_post_reset_side(self):
        m=self.model(['h','v'],{'A':("h' == v & v' == -2",'h >= 0')},
            [('A','A','h == 0 & v <= 0',"h' == 0 & v' == -v")])
        r=run(m,[1,0],duration=2,period=.5)
        self.assertAlmostEqual(r.jumps[0]['t'],1,places=12)
        self.assertAlmostEqual(r.states[2,1],2,places=11)
        self.assertAlmostEqual(r.states[-1,0],1,places=11)

    def test_strict_crossing_and_priority_complement(self):
        m=self.model(['x','y'],{q:("x' == 1 & y' == 0",'true') for q in 'ABC'},
            [('A','B','x >= 1 & y <= 0',"x' == x & y' == y"),
             ('A','C','x >= 1 & y > 0',"x' == x & y' == y")])
        r=run(m,[0,0])
        self.assertEqual(r.jumps[0]['target'],'B')
        m=self.model(['x'],{q:("x' == 1",'true') for q in 'AB'},
            [('A','B','x > 1',"x' == x")])
        r=run(m,[0])
        self.assertAlmostEqual(r.jumps[0]['t'],1,places=12)
        m=self.model(['x'],{q:("x' == 1",'true') for q in 'AB'},
            [('A','B','x > 0',"x' == x")])
        r=run(m,[0])
        self.assertEqual(r.jumps[0]['t'],0)

    def test_unresolved_cycle_and_nondeterminism_fail(self):
        m=self.model(['x'],{q:("x' == 0",'true') for q in 'AB'},
            [('A','B','x >= 1',"x' == x"),('B','A','x >= 1',"x' == x")])
        with self.assertRaisesRegex(GenerationError,'zero-time cycle'):
            run(m,[1])
        m=self.model(['x'],{q:("x' == 0",'true') for q in 'ABC'},
            [('A','B','x >= 1',"x' == x"),('A','C','x >= 1',"x' == x")])
        with self.assertRaisesRegex(GenerationError,'Multiple urgent'):
            run(m,[1])

    def test_invariant_without_continuation_fails(self):
        m=self.model(['x'],{'A':("x' == 1",'x <= 1')},[])
        with self.assertRaisesRegex(GenerationError,'no legal continuation'):
            run(m,[0])

    def test_unsafe_expression_fails_at_load(self):
        with self.assertRaises(ValueError):
            self.model(['x'],{'A':("x' == __import__('os').getcwd()",'true')},[])

    def test_tangency_is_reported_instead_of_silently_skipped(self):
        m=self.model(['x','v'],{q:("x' == v & v' == 1",'true') for q in 'AB'},
            [('A','B','x > 0',"x' == x & v' == v")])
        with self.assertRaisesRegex(GenerationError,'tangential'):
            run(m,[0,0])


class BatchTests(unittest.TestCase):
    def test_heating_observations_exclude_controller_state(self):
        data, latent, _ = generate_trace('multi_room_heating', duration=.2, sample_period=.1)
        self.assertEqual(data.shape, (3, 5))  # t, three temperatures, outdoor input
        self.assertEqual(latent.states.shape, (3, 4))  # internal control clock remains

    def test_loader_rejects_undeclared_observations(self):
        from unittest.mock import patch
        import halbench_loader as loader
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'example'
            (folder/'traces/train').mkdir(parents=True)
            (folder/'metadata.yaml').write_text(
                'variables: {states: [x], inputs: []}\ndim: {input: 0, output: 1}\n')
            (folder/'traces/train/extra.csv').write_text('t,x,controller\n0,1,0\n')
            with patch.object(loader, 'datasets_root', return_value=Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'Trace columns differ'):
                    loader.load('example')

    def test_noise_streams_sampling_and_latent_independence(self):
        a,ra,_=generate_trace('two_tank_system',seed=19,duration=2,sample_period=.02,process_noise=.1)
        b,rb,_=generate_trace('two_tank_system',seed=19,duration=2,sample_period=.02,process_noise=.1,measurement_noise=.1)
        c,rc,_=generate_trace('two_tank_system',seed=19,duration=2,sample_period=.01,process_noise=.1)
        np.testing.assert_array_equal(ra.states,rb.states)
        np.testing.assert_allclose(ra.states,rc.states[::2],rtol=0,atol=1e-13)
        np.testing.assert_array_equal(a[:,-1],b[:,-1])
        self.assertGreater(np.max(abs(a[:,1:3]-b[:,1:3])),.01)

    def test_serial_parallel_and_train_count_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=dict(system_id='two_tank_system',n_train=1,n_test=1,duration=.5,sample_period=.1,seed=71)
            first=generate_dataset(**args,workers=1,output_dir=Path(tmp)/'a')
            other=generate_dataset(**{**args,'n_train':2},workers=2,output_dir=Path(tmp)/'b')
            for split in ['train','test']:
                p=Path(split)/'trace_000000.csv.gz'
                self.assertEqual((first/p).read_bytes(),(other/p).read_bytes())
            m=json.loads((first/'manifest.json').read_text())
            self.assertEqual(m['status'],'complete')
            self.assertNotIn('mode',m['columns'])
            from halbench_loader import load_generated
            loaded=load_generated(first,split='test')
            self.assertEqual(len(loaded),1)
            self.assertEqual(list(loaded['trace_000000'].columns),m['columns'])
            with self.assertRaises(FileExistsError):
                generate_dataset(**args,workers=1,output_dir=first)

    def test_invalid_request_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out'
            with self.assertRaises(ValueError):
                generate_dataset('two_tank_system',n_train=1,duration=-1,output_dir=p)
            self.assertFalse(p.exists())

    def test_failed_batch_preserves_diagnostics_and_is_not_loadable(self):
        # An inelastic ball reaches a finite-time contact accumulation for
        # this long request; the generator must not invent a resting mode.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'long_ball'
            with self.assertRaises(GenerationError) as caught:
                generate_dataset('bouncing_ball',n_train=1,duration=20,
                                 sample_period=.1,seed=0,workers=1,output_dir=p)
            self.assertFalse(p.exists())
            stage=Path(caught.exception.context['staging_dir'])
            manifest=json.loads((stage/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertIn(manifest['failure']['category'],('simulation','execution_budget'))
            self.assertEqual(manifest['failure']['context']['seed'],0)
            from halbench_loader import load_generated
            with self.assertRaises(ValueError):
                load_generated(stage)


if __name__=='__main__':
    unittest.main()
