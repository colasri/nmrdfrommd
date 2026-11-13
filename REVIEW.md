# NMRDfromMD — Review notes

These notes describe the state of `main` at commit `5b9a205`. They are
an outside reader's notes. Some points may be choices made on purpose,
or may come from my misreading the code. Please read them as questions
and suggestions, not as verdicts. I checked each statement against the
code. Where I ran a small experiment, the result is given. Where a point
is still a guess, it says so.

Each point says how confident I am that it is correct:

- **high**: checked with a test or a direct reading of the code.
- **medium**: likely, but it depends on context.
- **low**: a hunch worth looking into.

## What works well

- **Clear structure.** The pipeline has three stages (`initialize`,
  `collect_data`, `finalize`), and the code is split into small,
  single-purpose modules (`geometry`, `correlation`, `fourier`,
  `relaxation`, `selection`).
- **Explicit units.** `units.py` states the internal conventions (Å, ps,
  MHz), and most functions comment the units of their values. This makes
  the physics much easier to follow and check.
- **Careful Fourier conventions.** `fourier_transform` names its
  normalisation convention instead of hiding it.
- **Error estimates.** The online Welford update gives error estimates
  without keeping each atom's data in memory.
- **A real test suite.** It already covers the relaxation formulas with
  closed-form cases, both the isotropic and the anisotropic branches,
  neighbour selection, and input validation.

## 1. Physics and numerical points to check

Points 1.1 to 1.5 each have a small test showing the behaviour, in
`tests/test_points_to_check.py` on the `points-to-check` branch. The
tests are marked `xfail`, so they do not break the suite.

### 1.1 R1 and R2 above f_max/2 rely on extrapolated values (high)

`compute_relaxation_rates` (`relaxation.py`) evaluates J(2f) with
`interp1d(..., fill_value="extrapolate")`. For f above half the Nyquist
frequency, 2f falls outside the computed range, so J(2f) is extrapolated
linearly from the last two points. For a smooth, decaying J this can be
quite inaccurate. With a Lorentzian J, R1 even becomes negative there.

*Suggestions:* report R1 and R2 only up to f_max/2, or flag that region,
or at least document it.

### 1.2 G(0) may get a full weight in the discrete transform (high)

J is computed as `2·dt·rfft(G)`, which approximates the one-sided
integral with the rectangle rule. The trapezoid rule would give the
first sample, G(0), a weight of ½. Here it gets a weight of 1, which adds
G(0)·dt to J at every frequency. For G(t) = exp(−t/τ) with dt = τ/10, we
get J(0) = 2.1τ instead of 2τ, about 5% too high. The relative error grows
at high frequency, where J is small.

*Suggestion:* subtract `dt·G(0)` from the result (equivalently, halve
G(0) before the FFT). This is cheap to do, unless the current weighting
is intended.

### 1.3 `compute_correlation_time` (high)

There are two things to look at in `correlation.py`:

- The non-integral branch with `dim=3` divides by `gij[0][m]`, the m-th
  lag of component 0. `gij[m][0]`, the first value of component m, seems
  to be what is meant. In a test where every component has τ = 1 ps, we
  get 1, 4 and 16 ps.
- In the integral branch, `t` is in ps, so the integral is already in ps.
  Dividing it again by `cst.pico` gives 2e12 instead of 2 for τ = 2 ps.

The function is documented as not being called by `run_analysis()`, so
results from the main pipeline are not affected.

Note that the existing test `test_calculate_tau_oneD_integral` builds its
expected value with the same `/ cst.pico`. It would need to change
together with the code.

### 1.4 Trajectories without a box fail even with `pbc=False` (high)

`_loop_over_trajectory` checks the box angles at every frame,
independently of `pbc`. A trajectory without box information (for
example, an isolated molecule) therefore raises a `TypeError`, even
though `compute_rij` does not use the box when `pbc=False`.

*Suggestion:* skip the check when `pbc=False`. The check could also run
once, unless the box can change during the trajectory.

### 1.5 Heteronuclear neighbours (medium)

`_initialize_physical_constants` uses the ¹H gyromagnetic ratio for every
pair, and `dipolar_prefactor` uses γ⁴. A code comment and a docstring
note acknowledge this ("H is enforced, improve in the future"). However,
nothing stops a user from passing ¹⁹F or ¹³C atoms as `neighbor_group`.
The result would then be off, both through the prefactor (γ_I²γ_S²
instead of γ_I⁴) and through the heteronuclear combination of spectral
densities.

*Suggestion:* until heteronuclear pairs are supported, a short note in the
public `NMRD` docstring, or a warning when the neighbours are not
hydrogens, could prevent surprises.

### 1.6 Error bars may underestimate the uncertainty (medium)

`R1_err` is the standard error of the mean across target atoms, which
assumes the atoms are independent. Atoms of the same molecule, or close
neighbours, are correlated. As an illustration: duplicating every
molecule (adding no information) makes the error bar shrink by a factor of
about 1.6.

*Suggestion:* block averaging over time, or averaging per molecule before
computing the error, would give more conservative values. This is mostly
worth documenting, so users know how to interpret `R1_err`.

### 1.7 Noise in the tail of the correlation function (low)

The autocorrelation is divided by (n − k), so the longest lags are
averaged over very few samples. They go into the FFT as they are, with no
truncation (for example at a chosen `t_max`) and no window. In a small
experiment, adding noise to the tail changed J by about 6%. My guess is
that this contributes to high-frequency noise in J, but I have not shown
that it is the main source.

### 1.8 float32 accumulators (low importance)

`gij` and the per-atom buffers are float32. Comparing with float64 gives
relative differences of up to about 2e-6. This is probably negligible
next to statistical noise, but float64 accumulators would remove the
question at little cost.

### 1.9 Rounding of the timestep (low)

`timestep = np.round(dt, 4)` rounds dt to 0.1 fs. I assume this cleans up
float32 noise in `dt` read from trajectory files: for example, the test
trajectory reports 0.0010000000475 ps. That is reasonable, but it would
change a timestep below about 1 fs. A short comment, or
`np.round(dt, 6)`, would make the intent explicit.

## 2. Performance

This is probably the main practical limitation for large systems.

- **The trajectory is read once per target atom**
  (`collect_data` → `_loop_over_trajectory`). The cost is therefore about
  N_i × the full trajectory I/O. Reading each frame once and processing
  all target atoms (or batches of them) would likely be much faster.
- **The FFT runs in a Python loop over each neighbour j**
  (`_calculate_correlation_ij`). One call to `np.fft.rfft` along the time
  axis on the whole `(n_frames, n_j)` array would replace the loop.
- `self.u.select_atoms(f'index {…}')` builds a selection string for each
  target atom. `self.u.atoms[index_j]` would be simpler and faster.
- The `use_wiener_khinchin` and `use_gpu` options of
  `autocorrelation_function` cannot be reached from `NMRD`. Also,
  `use_wiener_khinchin` and the "legacy" branch use exactly the same FFT
  length: 2·2^⌈log₂ n⌉ equals 2^⌈log₂(2n−1)⌉ for all n ≥ 2. The option
  could therefore be removed, unless something else was intended.
- The `_loop_over_trajectory` docstring mentions `start`, `stop` and
  `step`, but they are not implemented yet. They would be an easy way for
  users to reduce the cost.

## 3. API and code hygiene

- **`logging.basicConfig` runs at import time** (`NMR.py`), and the
  MDAnalysis log level is changed globally. A library usually leaves
  logging configuration to the application. Getting a logger with
  `getLogger(__name__)` is enough.
- **`from .NMR import *` with no `__all__`** also re-exports `np`, `mda`,
  `logging` and the helper functions. An `__all__ = ["NMRD"]`, and maybe
  a `__version__`, would make the public API explicit.
- `INVALID_TYPE_ANALYSIS` contains a `{value}` placeholder that is never
  filled. `EMPTY_J_GROUP` ("Empty atom groups j...") could suggest what to
  change, for example the `type_analysis` value or the neighbour
  selection.
- **Test functions in the library:** three `test_*` functions sit at the
  end of `nmrdfrommd/fourier.py`. They would fit better in `tests/`.
- **Docstrings that could be updated:**
  - The `NMRD` docstring does not describe `spin`, `num_log_points` or
    `seed`.
  - `log_bin` documents a `remove_empty` parameter and a default for
    `num_bins`, but neither exists.
  - There are two small typos: "election functions" in `selection.py`,
    and "time interface" (probably "time interval") in `NMRD`.
- **Results:**
  - `gij_mean` appears to equal `gij` (up to float precision); keeping
    one, next to `gij_err`, may be enough.
  - `results["log"]` bins R1 and R2 but not their errors.
  - Low-frequency log bins can be empty and then contain NaN.
- Minor: the module name `NMR.py` is uppercase (PEP 8 suggests lowercase),
  and `errors.py` and `units.py` have no licence header.

## 4. Tests

- **Two physics validation tests may not test what their names say:**
  - `test_rigid_dipole_analytical_limit` uses the moving two-molecule
    trajectory, not a rigid dipole. It compares R1 and R2 at f = 0, where
    R1 = R2 holds for any J.
  - `test_extreme_narrowing_R1_equals_R2` compares frequencies below
    1e3 MHz. With the test trajectory (dt = 1 fs, 10 frames), the only
    such frequency is f = 0, so it checks the same identity.

  A synthetic `MemoryReader` trajectory with a known correlation function
  would allow a real comparison with analytical results. The simplest
  case is a frozen dipole; another is an exponential correlation compared
  against the BPP expressions.
- **`tests/test_correlation.py` defines three test names twice**
  (`test_autocorrelation_of_constant_signal`,
  `test_autocorrelation_at_zero_lag`, `test_autocorrelation_of_sine_wave`).
  Python keeps only the second definition, so the
  `use_wiener_khinchin=True` variants never run. Renaming them, or using
  `pytest.mark.parametrize`, would fix this. A linter would flag it
  (ruff rule F811).
- `conftest.py` uses relative paths (`datafiles/...`), so the tests only
  pass when run from inside `tests/`, which is what CI does. Resolving
  paths from `__file__` would let `pytest` work from the repository root.
- `type_analysis="inter_molecular"` and `"intra_molecular"` are tested in
  `select_neighbor_indices`, but not through a full `NMRD` run. The
  `dim=3` branch of `compute_correlation_time` is not tested.

## 5. Packaging, CI and docs

Most of these points are addressed on the `points-to-check` branch,
except publishing and versioning, which were left for later on purpose.

- **The PyPI publish job does not seem able to run.** The workflow
  triggers only on pushes and pull requests to `main`, so
  `startsWith(github.ref, 'refs/tags/v')` is never true. Also,
  `environment: pypi` is passed under `with:`. `pypa/gh-action-pypi-publish`
  has no such input, so the environment is probably meant to be a
  job-level `environment:` key.
- **Two sources of metadata.** `setup.py` and `pyproject.toml` both
  describe the package, with different descriptions, readmes, URLs and
  author fields. Because `pyproject.toml` has a `[project]` table,
  setuptools ignores the overlapping fields of `setup.py`, with warnings.
  This includes `install_requires`, so pytest and coverage are, in fact,
  not installed with the package. Keeping only `pyproject.toml` would
  remove the ambiguity. Test tools could go in a dependency group, and
  `requirements.txt` would no longer be needed.
- **Minimum versions.** The code uses `scipy.special.sph_harm_y` (SciPy ≥
  1.15) and `np.trapezoid` (NumPy ≥ 2.0), but the dependencies are
  unpinned. `requires-python = ">=3.12.3"` is very specific. `>=3.12`
  seems sufficient, and an older minimum might also work, though I have
  not checked.
- **CI** uses `actions/checkout@v2` and `actions/setup-python@v2`, which
  run on a deprecated Node version, and tests a single Python version.
- `.gitignore` has `**/.*npz`, which only matches hidden files. `*.npz`
  may have been intended.
- **README:** a short usage example, a link to the documentation, and
  citation information would help new users.
- **`developers/README.md`** describes a manual `setup.py sdist` and
  `twine` release, which differs from the CI publish job. It also mentions
  `docs/source/conf.py`, `CITATION.cff` and `CHANGELOG`, which are not in
  this repository; they may live elsewhere.

## Possible order of work

1. Discuss the points in section 1, starting with 1.1 to 1.4, which are
   well established and simple to fix.
2. Add a synthetic analytical test, to guard those fixes against
   regressions.
3. Read the trajectory once instead of once per target atom.
4. Finish the packaging and CI work (publishing and versioning).
