"""
Extract the reference values used by ``tests/test_goldens.py`` from the
TerraME ``logo`` goldens of ``terrame-goldens``.

The goldens record, for seeds 1 to 50, every value the ``logo`` models report
after every step. The tests need only a few steps, so this script keeps the
values of the 50 seeds at those steps (``logo_reference.csv``) and records
the SHA-256 of each source file, so the extract can be checked against a
release of ``terrame-goldens``.

    python tests/goldens/make_reference.py /path/to/terrame-goldens
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: case -> steps compared by the tests
CHECKPOINTS = {
    "schelling": (10, 100, 500),
    "predator_prey": (10, 20, 50, 100),
    "life_cycle": (20, 40, 100),
    "ants": (40, 100),
    "disease": (5, 10, 30),
}


def main(root: Path) -> None:
    rows = []
    sources = {}
    for case, steps in CHECKPOINTS.items():
        path = root / "goldens" / "logo" / f"{case}.csv.gz"
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt", newline="") as f:
            for r in csv.DictReader(f):
                if int(r["step"]) in steps:
                    rows.append((case, r["variable"], int(r["step"]), int(r["seed"]), r["value"]))
    rows.sort()
    with open(HERE / "logo_reference.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "variable", "step", "seed", "value"])
        w.writerows(rows)
    manifest = json.loads((root / "manifest.json").read_text())
    meta = {
        "source": "https://github.com/LambdaGeo/terrame-goldens",
        "version": manifest["version"],
        "terrame": manifest["terrame"],
        "packages": [p for p in manifest["packages"] if p["name"] == "logo"],
        "files": sources,
        "checkpoints": {k: list(v) for k, v in CHECKPOINTS.items()},
    }
    (HERE / "logo_reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(rows)} values written")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
