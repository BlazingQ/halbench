import argparse
from .dataset import generate_dataset
from .collection import generate_collection


def main():
    parser = argparse.ArgumentParser(
        description='Generate a dataset for a synthetic benchmark or a collection for all synthetic systems.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  halbench-generate two_tank --n-train 8 --n-test 2 --seed 42 '
               '--output-dir generated/tank_001\n\n'
               'Collection:\n  halbench-generate all --n-train 8 --n-test 2 --seed 42 '
               '--output-dir generated/collection_42\n\n'
               'The generator creates OUTPUT_DIR and its parent directories.\n'
               'Each dataset contains manifest.json, train/ and test/.\n'
               'The all collection uses each system\'s default horizon and sampling interval.')
    parser.add_argument('system_id', help='synthetic system identifier, or all for the complete synthetic collection')
    parser.add_argument('--n-train', type=int, required=True, help='number of training trajectories')
    parser.add_argument('--n-test', type=int, default=0, help='number of test trajectories (default: 0)')
    parser.add_argument('--duration', type=float, help='single-system time horizon (default: generation.yaml)')
    parser.add_argument('--sample-period', type=float, help='single-system output interval (default: generation.yaml)')
    parser.add_argument('--seed', type=int, default=0, help='nonnegative random seed (default: 0)')
    parser.add_argument('--process-noise', type=float, default=0., help='Gaussian process noise strength (default: 0)')
    parser.add_argument('--measurement-noise', type=float, default=0., help='Gaussian measurement noise strength (default: 0)')
    parser.add_argument('--output-dir', required=True, metavar='OUTPUT_DIR', help='new output path outside systems/ and datasets/')
    parser.add_argument('--workers', default='auto', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers != 'auto':
        try:
            args.workers = int(args.workers)
        except ValueError:
            parser.error('--workers must be auto or a positive integer')
    try:
        options = vars(args).copy()
        if args.system_id == 'all':
            if args.duration is not None or args.sample_period is not None:
                parser.error('--duration and --sample-period apply to a single system; all uses per-system defaults')
            for name in ('system_id', 'duration', 'sample_period'):
                options.pop(name)
            result = generate_collection(**options)
        else:
            result = generate_dataset(**options)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    print(result)


if __name__ == '__main__':
    main()
