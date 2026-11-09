"""Generate catalog.csv and the README catalog from system definitions and datasets."""
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Presentation order follows the survey's Table 3.
SYSTEM_ORDER = (
    'buck_converter', 'simple_heating_system', 'excitable_cell', 'complex_tank',
    'switched_lorenz', 'oscillator', 'oscillator_2', 'underdamped_system',
    'dc_motor', 'jumper_dog', 'lotkavolterra', 'spacecraft', 'biological_system',
    'ecg_signals', 'bouncing_ball', 'simple_non_polynomial', 'heating_with_input',
    'multi_room_heating', 'two_tank_system', 'cooperative_vehicles', 'aircraft',
    'engine_timing_model', 'f1tenth_car', 'controlled_underdamped', 'duffing_oscillator',
)


def dynamics_label(metadata):
    label = metadata['dynamics'].capitalize()
    order = metadata['dynamics_order']
    return f'{label} ({order})' if order not in ('1st', 'unknown') else label

# Catalog columns: display name and metadata accessor.
COLUMNS = [
    ("Name", lambda m: m["name"]),
    ("System type", lambda m: {"state-only": "Closed", "input-output": "Open"}[m["data_format"]]),
    ("System origin", lambda m: m["type"].capitalize() + ('*' if m.get('origin_detail') == 'industrial-simulator' else '')),
    ("#Modes", lambda m: '---' if m["num_modes"] == -1 else m["num_modes"]),
    ("Dim (I+O)", lambda m: f"{m['dim']['input']} + {m['dim']['output']}"),
    ("Dynamics (Order)", dynamics_label),
    ("Guard", lambda m: {'fixed-time': 'Fixed time', 'none': 'No guards',
                         'not-specified': 'Not specified'}.get(m['guard'], m['guard'].capitalize())),
    ("Reset", lambda m: m["reset"].capitalize()),
    ("Source", lambda m: m["source"]["work"]),
    ("Traces (train/test)", lambda m: f"{m['traces']['train']}/{m['traces']['test']}"),
]

BEGIN, END = "<!-- CATALOG:BEGIN -->", "<!-- CATALOG:END -->"


def load_metadata():
    rows = []
    for meta_path in sorted((ROOT / "systems").glob("*/metadata.yaml")):
        with meta_path.open() as f:
            metadata = yaml.safe_load(f)
        manifest = json.loads((ROOT/'datasets'/metadata['system_id']/'manifest.json').read_text())
        metadata['traces'] = {split: sum(r['split'] == split for r in manifest['traces'])
                              for split in ('train', 'test')}
        rows.append(metadata)
    by_id = {m['system_id']: m for m in rows}
    if set(by_id) != set(SYSTEM_ORDER):
        raise ValueError('System definitions and catalog order differ')
    return [by_id[sid] for sid in SYSTEM_ORDER]


def write_catalog_csv(rows):
    out = ROOT / "catalog.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["system_id"] + [c[0] for c in COLUMNS])
        for m in rows:
            w.writerow([m["system_id"]] + [c[1](m) for c in COLUMNS])
    print(f"wrote {out}")


def write_readme_table(rows):
    readme = ROOT / "README.md"
    text = readme.read_text()
    header = "| " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    body = "\n".join(
        "| " + " | ".join(
            f"[{m['name']}](systems/{m['system_id']}/metadata.yaml)" if i == 0 else str(c[1](m))
            for i, c in enumerate(COLUMNS)) + " |" for m in rows
    )
    table = f"{BEGIN}\n{header}\n{sep}\n{body}\n{END}"
    if BEGIN in text and END in text:
        text = text[: text.index(BEGIN)] + table + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + f"\n\n## Catalog\n\n{table}\n"
    readme.write_text(text)
    print(f"updated {readme}")


def main():
    rows = load_metadata()
    write_catalog_csv(rows)
    write_readme_table(rows)


if __name__ == "__main__":
    main()
