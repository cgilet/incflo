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
one process for these small startup checks.
