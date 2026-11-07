# halbench — Benchmark Systems and Datasets for Hybrid Automata Learning

A collection of 25 benchmark systems: 22 synthetic systems with models in a
unified SpaceEx (SX) format and a Python trace generator, plus three real-world
datasets. This collection accompanies the survey *Learning hybrid automata from
time-series traces: a comprehensive survey*.

## Installation

Use Python 3.11 or newer. In a Bash terminal, run from the cloned repository root:

~~~sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[generate]'
~~~

`.venv` is the environment directory under the current repository. The first
command creates it using the current Python interpreter; the second activates it
for the terminal. To use another environment directory, replace `.venv` in both
commands. An existing project environment can be activated with the second command.
The editable installation uses the systems and bundled data in this repository.

## Repository layout

~~~text
systems/<system_id>/
  metadata.yaml          # system, variables, source and notes
  model.xml              # synthetic system model
  generation.yaml        # synthetic generation settings

datasets/<system_id>/
  manifest.json          # batch information and file records
  train/*.csv.gz
  test/*.csv.gz
~~~

`systems/` describes all 25 systems. Each synthetic system has an SX model and
settings for initial conditions, inputs, sampling and noise. Real-world system
metadata describe the supplied observations.

`datasets/` provides ready-to-use data: 256 training and 64 test trajectories per
synthetic system, generated with zero noise strengths and seed `20261106`.
The three real-world datasets retain their source train/test splits.

Each dataset manifest records its system ID, variable order, time unit, sample
counts, file hashes and data license. Synthetic manifests also record the actual
generation parameters, configuration snapshot, system-file fingerprints and
software versions. `generation.yaml` contains defaults; the manifest records
the values used for that batch.

## Read data

Read the bundled Two-tank System benchmark (`two_tank`) by system ID:

~~~python
import halbench_loader as hb

systems = hb.list_datasets()
train = hb.load("two_tank", split="train")
test = hb.load("two_tank", split="test")
first_trace = next(iter(train.values()))  # pandas DataFrame
~~~

A system ID resolves to `datasets/<system_id>/` in the installed repository.
For your own data, pass its directory explicitly:

~~~python
train = hb.load(path="generated/tank_42", split="train")
test = hb.load(path="generated/tank_42", split="test")
collection_test = hb.load(path="generated/collection_43/two_tank", split="test")
~~~

The generation commands below create these example paths. Relative paths are
resolved from the current working directory; absolute paths are also supported.
Each call returns a dictionary of trajectory names and pandas DataFrames.
The manifest contains the information required to read a dataset after it is
copied to another directory.

## Trace format

- One trajectory per UTF-8 CSV file with a header, compressed as `.csv.gz`.
  The loader also reads uncompressed `.csv` files listed in a dataset manifest.
- Columns: `t, <outputs...>, <inputs...>`. Names and order are recorded in the
  manifest's `variables.states` and `variables.inputs`.
- `t` is the sample time. `time_unit: unknown` denotes an unspecified physical
  unit, and `sampling_period: null` denotes nonuniform sampling.
- Train and test splits contain complete trajectories.

## Generate data

Generate two training trajectories and one test trajectory for the Two-tank
System benchmark (`two_tank`):

~~~sh
halbench-generate two_tank --n-train 2 --n-test 1 --seed 42 --output-dir generated/tank_42
~~~

Generate one training and one test trajectory for each of the 22 synthetic systems
with the reserved ID `all`:

~~~sh
halbench-generate all --n-train 1 --n-test 1 --seed 43 --output-dir generated/collection_43
~~~

Single-system output, collection members and bundled datasets share this layout:

~~~text
generated/tank_42/
  manifest.json
  train/
  test/

generated/collection_43/
  aircraft/{manifest.json,train/,test/}
  two_tank/{manifest.json,train/,test/}
  ...
~~~

The generator creates the output directory and its parent directories. Use a new
output path for each dataset or collection. Read the results using `hb.load(path=...)`
as shown above.

View the options with `halbench-generate -h` or `halbench-generate --help`.

| Option | Meaning |
|---|---|
| `--n-train`, `--n-test` | Trajectories per split and system; `--n-test` defaults to 0. |
| `--seed` | Nonnegative random seed for initial conditions, inputs and noise; defaults to 0. |
| `--process-noise`, `--measurement-noise` | Nonnegative Gaussian strengths; both default to 0. |
| `--output-dir` | New output path outside the repository's `systems/` and `datasets/`. |
| `--duration`, `--sample-period` | Single-system horizon and sampling interval; defaults come from `generation.yaml`. |

The `all` command uses each system's default horizon and sampling interval.
The single-system command supports overriding these values. Both commands share
the count, seed and noise options.

Generated trajectories include time zero and the requested end time. A final
interval may be shorter when the duration is not divisible by the sampling
interval. Samples at reset times contain the post-reset values.

Process noise adds piecewise-constant Gaussian forcing to configured dynamics
channels. Measurement noise adds independent Gaussian noise to sampled outputs.
For each channel, the standard deviation is the requested noise strength times
the fixed scale in `generation.yaml`.

## Catalog

Select a system name to view its metadata and notes. Closed denotes state-only
systems; Open denotes systems with external inputs. Dim (I+O) gives the input and
output dimensions. [catalog.csv](catalog.csv) provides the same catalog as a CSV.

<!-- CATALOG:BEGIN -->
| Name | System type | System origin | #Modes | Dim (I+O) | Dynamics (Order) | Guard | Reset | Source | Traces (train/test) |
|---|---|---|---|---|---|---|---|---|---|
| [Aircraft](systems/aircraft/metadata.yaml) | Open | synthetic | 1 | 2 + 3 | nonlinear (1st) | none | no | HSCC25 | 256/64 |
| [Biological System](systems/biological_system/metadata.yaml) | Closed | synthetic | 2 | 0 + 9 | polynomial (1st) | linear | no | Dainarx | 256/64 |
| [Bouncing Ball](systems/bouncing_ball/metadata.yaml) | Closed | synthetic | 1 | 0 + 2 | affine (1st) | linear | yes | HSCC25 | 256/64 |
| [Buck Converter](systems/buck_converter/metadata.yaml) | Closed | synthetic | 3 | 0 + 2 | affine (1st) | linear | no | FaMoS | 256/64 |
| [Complex Tank](systems/complex_tank/metadata.yaml) | Closed | synthetic | 8 | 0 + 3 | affine (1st) | linear | no | FaMoS | 256/64 |
| [Controlled Underdamped](systems/controlled_underdamped_system/metadata.yaml) | Open | synthetic | 4 | 1 + 2 | affine (2nd) | linear | yes | Dainarx | 256/64 |
| [Cooperative Vehicles](systems/cooperative_vehicles/metadata.yaml) | Open | synthetic | 2 | 1 + 9 | linear (1st) | fixed-time | no | hautlearn | 256/64 |
| [DC Motor](systems/dc_motor/metadata.yaml) | Closed | synthetic | 2 | 0 + 2 | affine (4th) | linear | no | Dainarx | 256/64 |
| [Duffing Oscillator](systems/duffing_oscillator/metadata.yaml) | Open | synthetic | 2 | 1 + 1 | polynomial (2nd) | nonlinear | yes | Dainarx | 256/64 |
| [ECG Signals](systems/ecg_signals/metadata.yaml) | Closed | real-world | -1 | 0 + 1 | nonlinear (unknown) | implicit | no | HySynth | 3/0 |
| [Engine Timing Model](systems/engine_timing/metadata.yaml) | Open | real-world | -1 | 2 + 1 | nonlinear (unknown) | implicit | no | HybridLearner | 64/40 |
| [Excitable cell](systems/excitable_cell/metadata.yaml) | Closed | synthetic | 4 | 0 + 1 | constant (1st) | constant | no | HybridLearner | 256/64 |
| [F1tenth Car](systems/f1tenth_car/metadata.yaml) | Open | real-world | -1 | 2 + 4 | nonlinear (unknown) | implicit | no | HSCC25 | 36/5 |
| [Heating with Input](systems/heating_system_with_input/metadata.yaml) | Open | synthetic | 2 | 1 + 1 | affine (1st) | linear | no | FaMoS | 256/64 |
| [Jumper Dog](systems/jumper_dog/metadata.yaml) | Closed | synthetic | 2 | 0 + 4 | affine (1st) | linear | no | Dainarx | 256/64 |
| [LotkaVolterra](systems/lotka_volterra/metadata.yaml) | Closed | synthetic | 2 | 0 + 2 | polynomial (1st) | nonlinear | no | Dainarx | 256/64 |
| [Multi-room Heating](systems/multi_room_heating/metadata.yaml) | Open | synthetic | 6 | 1 + 3 | affine (1st) | linear | no | hautlearn | 256/64 |
| [Oscillator 2](systems/oscillator/metadata.yaml) | Closed | synthetic | 4 | 0 + 2 | affine (1st) | linear | no | HybridLearner | 256/64 |
| [Oscillator](systems/oscillator__dainarx/metadata.yaml) | Closed | synthetic | 2 | 0 + 2 | nonlinear (1st) | nonlinear | no | Dainarx | 256/64 |
| [Simple heating system](systems/simple_heating_system/metadata.yaml) | Closed | synthetic | 2 | 0 + 1 | affine (1st) | constant | no | FaMoS | 256/64 |
| [Simple Non-Polynomial](systems/simple_non_polynomial/metadata.yaml) | Closed | synthetic | 2 | 0 + 1 | nonlinear (1st) | linear | yes | Dainarx | 256/64 |
| [Spacecraft](systems/spacecraft/metadata.yaml) | Closed | synthetic | 2 | 0 + 4 | nonlinear (1st) | nonlinear | no | Dainarx | 256/64 |
| [Switched Lorenz](systems/switched_lorenz_attractor/metadata.yaml) | Closed | synthetic | 2 | 0 + 3 | polynomial (1st) | linear | no | InferDynamic | 256/64 |
| [Two-tank System](systems/two_tank/metadata.yaml) | Open | synthetic | 4 | 1 + 2 | affine (1st) | constant | no | HybridLearner | 256/64 |
| [Underdamped System](systems/underdamped_system/metadata.yaml) | Closed | synthetic | 2 | 0 + 2 | affine (2nd) | linear | no | Dainarx | 256/64 |
<!-- CATALOG:END -->

## License and sources

Project contributions and generated datasets use the [MIT License](LICENSE).
The ECG Signals dataset retains HySynth's [MIT License](datasets/ecg_signals/LICENSE),
the Engine Timing Model dataset retains HybridLearner's
[GNU GPL version 3](datasets/engine_timing/LICENSE), and the F1tenth Car dataset
uses [CC0-1.0](datasets/f1tenth_car/LICENSE).
System metadata identify the sources; dataset manifests record the applicable
data license and, for converted source CSV files, their source paths and format changes.
