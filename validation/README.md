# Divergence-source validation

BDS, MOL, cell-centered projection, RZ, embedded boundary geometry and general closed
domains are outside the supported model. Algorithm guards apply only when
`incflo.use_divu_source=1`; existing S=0 configurations remain available.

See [the deferred coarse/fine investigation](COARSE_FINE.md) for known AMR accuracy
limitations, evidence and future work. A regression benchmark tests maintenance,
not convergence or physical-model completeness.

Startup checks (use a non-EB executable):

```sh
python3 validation/check_restrictions.py --exe /path/to/incflo2d.gnu.ex --dim 2
```

Each rejected configuration must exit unsuccessfully with the expected diagnostic;
an S=0 startup control must succeed. Build with `make -j2` or fewer jobs and use
one process in 2D or two MPI ranks in 3D for these small startup checks.

## Exact-solution checks

Build explicitly against chosen dependencies, with at most two jobs:

```sh
make -C test_no_eb_2d -j2 USE_MPI=FALSE AMREX_HOME=/path/to/amrex AMREX_HYDRO_HOME=/path/to/AMReX-Hydro
python3 validation/run.py --dim 2 --exe test_no_eb_2d/incflo2d.gnu.ex --output /tmp/divu-2d
```

For 3D, build with MPI enabled and run on two ranks:

```sh
make -C test_no_eb_3d -j2 USE_MPI=TRUE AMREX_HOME=/path/to/amrex AMREX_HYDRO_HOME=/path/to/AMReX-Hydro
python3 validation/run.py --dim 3 --mpi-ranks 2 --exe test_no_eb_3d/incflo3d.gnu.MPI.ex --output /tmp/divu-3d
```

Runs are sequential, with one thread per rank. 2D defaults to one process; 3D defaults
to two MPI ranks. `--mpi-ranks` accepts other counts (at least two in 3D).
Each output directory must be new; logs, exact commands and
`summary.json` are retained. Plotfile `job_info` records compiled dependency versions.
`--full` adds a finer resolution. `--amr` adds an offset-invariance integration check;
it does not assert second-order coarse/fine convergence. `--cases` selects a subset.
The shell entry points in each test directory call the same driver; the former
`_options` entry point is now an alias, not a claim that all option combinations ran.

603 is ballistic flow with variable density and both tracer forms; 604 adds a
constant source offset. The explicit closed-domain override is accepted only for
603–605. Mean removal changes the predictor's working source, not stored S.
605 is a forced, constant-density wall MMS; 606 is an open warm channel with
S computed from temperature. Initial mapping and end-time assumptions are checked
by the problem implementations. Legacy 600–602 input definitions remain for reference;
the new scope restrictions intentionally reject their unsupported configurations.

The driver checks final-time diagnostic completeness and finite values, field-specific
Linf/L1/L2 rate floors, fixed-dt source-offset invariance, and adaptive dt histories.
The floors are broad historical bounds, not a claim of order 2 for every norm.
Limited scalar extrema and stock-AMReX wall pressure have lower thresholds. Norms
are per level (including covered coarse cells), and compare point samples at cell
centers. They are not composite AMR norms. `python3 validation/test_reports.py`
checks that missing/duplicate/nonfinite reports and failed rates/comparisons are rejected.

## Remaining accuracy limits

- Scalar wall conditions assume zero normal derivatives unless wall values are supplied;
  this can lower local accuracy for nondiffusive scalars with nonzero wall-normal gradients.
- Slip-wall higher-order extrapolation can overshoot, including for S=0. PPM limiting
  and velocity reversal behavior can limit scalar maxima; no Hydro predictor fix is included.
- The AMReX physical-wall gradient correction is a separate, currently optional dependency
  feature. Corrected studies require the matching Hydro wrapper and diffusion boundary order.
  The default acceptance profile here uses stock AMReX and does not claim that fix is active.
- Known coarse/fine issues are preserved in [the separate dossier](COARSE_FINE.md).
- Solver tolerance, pressure staggering and explicit-605 timestep sensitivity require care.
  Passing a regression comparison does not establish convergence or an EOS-based model.

## S=0 compatibility controls

Build the baseline and candidate with identical compiler options and dependency
revisions, then compare two evolved 2D controls:

```sh
python3 validation/check_zero_source.py --baseline /path/to/baseline/incflo2d.gnu.ex \
    --exe /path/to/candidate/incflo2d.gnu.ex --output /tmp/divu-zero-check
```

This compares the final numerical plotfile headers and field bytes for Taylor–Green
and the Godunov Boussinesq bubble at 20 steps; build metadata is excluded.
