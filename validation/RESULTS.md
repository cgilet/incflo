# Validation record — 2026-11-05

These are newly run checks of this series, not the historical tables in the original
candidate commit. The numerical sources were tested in the staging checkout before
final commits were assembled; build banners can therefore identify the parent revision
with `-dirty`. The final third commit adds the tested pulsing-source profile and regression
configuration. No uncommitted dependency changes were used.

## Dependencies and builds

- Upstream code baseline: `a788f2e4`; PR #279 prerequisite: `14f723f0`.
- AMReX: `47ceffc47cea` (stock wall treatment).
- AMReX-Hydro: `fa061e7` (stock PPM velocity predictor).
- GNU make, GNU C++, double precision, release, no EB or OpenMP.
- 2D: `USE_MPI=FALSE`; 3D: `USE_MPI=TRUE`, all numerical runs on **two MPI ranks**.
- Make and CMake builds passed. CMake 2D Release used the same source dependencies
  with MPI and EB off; the evolved S=0 baseline controls also passed with its executable.
- The pending physical-wall correction and its Hydro wrapper were not enabled or validated.

Build commands from the source checkout:

```sh
make -C test_no_eb_2d -j2 USE_MPI=FALSE AMREX_HOME=/home/gcse/amrex AMREX_HYDRO_HOME=/home/gcse/AMReX-Hydro
make -C test_no_eb_3d -j2 USE_MPI=TRUE AMREX_HOME=/home/gcse/amrex AMREX_HYDRO_HOME=/home/gcse/AMReX-Hydro
cmake -S . -B /tmp/divu-cmake-check -DINCFLO_DIM=2 -DINCFLO_MPI=OFF -DINCFLO_EB=OFF -DAMREX_HOME=/home/gcse/amrex -DAMREX_HYDRO_HOME=/home/gcse/AMReX-Hydro
cmake --build /tmp/divu-cmake-check -j2
```

## Exact-solution convergence

The two-resolution checks passed the committed field-specific rate floors. These
coarse-resolution orders are measurements, not asymptotic convergence guarantees.
Pressure in 606 is particularly sensitive to coarse resolution. Velocities that remain
at roundoff are excluded from rate calculations.

| Dimension | Case | Lowest velocity Linf order | Pressure Linf order | Lowest scalar Linf / L1 / L2 order |
| --- | --- | --- | --- | --- |
| 2D | ballistic | 2.06 | 1.80 | 0.79 / 1.98 / 1.80 |
| 2D | walls | 2.05 | 1.93 | 0.75 / 1.96 / 1.66 |
| 2D | mms | 1.97 | 1.09 | — / — / — |
| 2D | temperature | 2.58 | 3.43 | 1.55 / 1.89 / 1.75 |
| 3D | ballistic | 1.97 | 1.45 | 0.94 / 1.78 / 1.61 |
| 3D | walls | 2.19 | 1.82 | 0.68 / 1.88 / 1.65 |
| 3D | mms | 1.83 | 1.60 | — / — / — |
| 3D | temperature | 2.60 | 3.74 | 1.55 / 1.90 / 1.75 |

Resolutions: 2D ballistic/MMS 32–64, walls/temperature 64–128; 3D ballistic/MMS
16–32, walls 32–64, temperature 64–128. Temperature uses n/8 transverse cells.
Pressure and all velocity, density, tracer and temperature diagnostics were checked
for finite values, complete final reports and their stated rate floors. The table
reports the lowest component/scalar order in each group; norms are per level.

Both dimensions passed fixed-dt 603/604 offset invariance, adaptive solution and
**timestep-history** invariance, and a fixed-dt two-level offset comparison. AMR checks
establish integration consistency, not second-order interface accuracy.

Commands (use new output directories):

```sh
python3 validation/run.py --dim 2 --exe test_no_eb_2d/incflo2d.gnu.ex --output /tmp/divu-check-2d --amr
python3 validation/run.py --dim 3 --mpi-ranks 2 --exe test_no_eb_3d/incflo3d.gnu.MPI.ex --output /tmp/divu-check-3d --amr
```

Initial driver attempts correctly failed on a duplicate punctuation character in a parsed
step size, an incorrectly assumed 3D end time, and an undersized warm-channel mesh. Those
harness issues were corrected. A 2D wall tracer maximum at 32–64 had order 0.23; the
acceptance study uses the documented finer 64–128 range instead of weakening its floor.

Local retained evidence: `/tmp/divu-check-2d-r4/summary.json`, the 3D ballistic/wall/MMS
logs under `/tmp/divu-check-3d-mpi2-r2`, and the successful temperature summary under
`/tmp/divu-check-3d-temperature`. Combined checked 3D rates are in
`/tmp/divu-check-3d-summary.json`. These scratch paths are not required by the drivers.

## Scope and maintenance checks

- Ten targeted startup rejections plus a successful S=0 startup control passed in 2D
  and in 3D on two ranks. Each rejection required the expected diagnostic and failed exit.
- Four parser negative-control tests passed: missing/duplicate/nonfinite reports,
  final-time selection, offset mismatch and nonconverging rate rejection.
- The shared regression example parses; both new cases point to existing non-EB inputs
  and use two ranks. Shell syntax and diff whitespace checks passed.
- 606 ran through step 20. Restarting checkpoint 5 reproduced every compared plotfile
  field with **zero difference** in `fcompare`: all velocity/pressure-gradient components,
  density, both tracers, temperature and raw S. Source and dS/dt checkpoint files existed.
- 607 ran through step 40, retained a fine level and exercised level remaking/creation
  during evolution. No exact solution or interface convergence claim is made.
- Evolved S=0 Taylor–Green and Godunov Boussinesq-bubble controls at 32x32, 20 steps,
  dt=0.005 had byte-identical numerical plotfile headers and field data versus a separate
  `14f723f0` baseline built using the same compiler flags and dependencies.

```sh
python3 validation/check_restrictions.py --dim 2 --exe test_no_eb_2d/incflo2d.gnu.ex
python3 validation/check_restrictions.py --dim 3 --mpi-ranks 2 --exe test_no_eb_3d/incflo3d.gnu.MPI.ex
python3 validation/test_reports.py
python3 validation/check_regressions.py --exe test_no_eb_3d/incflo3d.gnu.MPI.ex --fcompare /home/gcse/bin/fcompare.gnu.ex --output /tmp/divu-regression-check-new
python3 validation/check_zero_source.py --baseline /tmp/incflo-divu-baseline/test_no_eb_2d/incflo2d.gnu.ex --exe test_no_eb_2d/incflo2d.gnu.ex --output /tmp/divu-zero-source-new
```

Local regression logs and exact commands remain in `/tmp/divu-regression-check`; S=0
comparison artifacts remain in `/tmp/divu-zero-source-check`. Initial nightly benchmarks
were not generated or enrolled. The deployed upstream regression configuration is external.

## Limits of this verification

No GPU or EB build/run, 3D CMake build, ratio-4 convergence, full option-product study,
explicit/backward-Euler accuracy study, or corrected-AMReX wall study was performed.
The deferred coarse/fine issues remain documented in [COARSE_FINE.md](COARSE_FINE.md).
General closed domains remain unsupported: analytic mean removal is not an EOS or
thermodynamic-pressure model. Historical scalar-wall and slip-wall bounds remain relevant.
