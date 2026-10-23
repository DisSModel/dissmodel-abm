# TerraME reference values for the logo models

`logo_reference.csv` is an extract of the `logo` suite of
[`terrame-goldens`](https://github.com/LambdaGeo/terrame-goldens) v0.1.0
(DOI [10.5281/zenodo.23592970](https://doi.org/10.5281/zenodo.23592970)):
what TerraME 2.0.1 computed running the models of the
[`logo`](https://github.com/TerraME/logo) package (commit `bb57a26`) with
default parameters, for seeds 1 to 50. Only the steps compared by
`tests/test_goldens.py` are kept. Columns: `case, variable, step, seed, value`.

`logo_reference.json` records the source version, the package commit, the
steps kept and the SHA-256 of each source file. To make the extract again
from a copy of `terrame-goldens`:

```bash
python tests/goldens/make_reference.py /path/to/terrame-goldens
```

The values are the output of TerraME and of the `logo` package (INPE and
TerraLAB/UFOP); the extract is offered under CC-BY-4.0, as the goldens it
comes from.
