#!/usr/bin/env python3
"""Check startup rejections without counting an unrelated abort as a pass."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--mpi-ranks', type=int)
    parser.add_argument('--dim', type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    exe = args.exe.resolve()
    dim = args.dim
    ranks = args.mpi_ranks if args.mpi_ranks is not None else (2 if dim == 3 else 1)
    if ranks < 1 or (dim == 3 and ranks < 2):
        parser.error('3D requires at least two MPI ranks')
    launch = ['mpiexec', '-n', str(ranks), str(exe)] if ranks > 1 else [str(exe)]
    base = [
        'max_step=0', 'incflo.probtype=0', 'incflo.use_divu_source=1',
        'incflo.constant_density=0', 'incflo.advect_tracer=0',
        'amr.n_cell=' + ' '.join(['8'] * dim), 'amr.max_level=0',
        'geometry.prob_lo=' + ' '.join(['0'] * dim),
        'geometry.prob_hi=' + ' '.join(['1'] * dim),
        'geometry.is_periodic=' + ' '.join(['1'] * dim),
        'amr.plot_int=-1', 'amr.check_int=-1',
    ]
    open_bc = ['geometry.is_periodic=' + ' '.join(['0'] + ['1'] * (dim-1)),
               'xlo.type=mi', 'xlo.velocity=' + ' '.join(['1'] + ['0'] * (dim-1)),
               'xhi.type=po', 'xhi.pressure=0']
    cases = [
        ('bds', ['incflo.advection_type=BDS'], 'only implemented for'),
        ('mol', ['incflo.advection_type=MOL'], 'only implemented for'),
        ('cc', ['incflo.use_cc_proj=1'], 'CC projection'),
        ('ratio', ['amr.max_level=1', 'amr.ref_ratio=3'], 'refinement ratios'),
        ('closed', [], 'general closed domains'),
        ('closed_override', ['incflo.divu_source_allow_closed=1'], 'general closed domains'),
        ('legacy_outflow', open_bc + ['incflo.probtype=602'], 'S is nonzero at pressure outflows'),
        ('pressure_inflow', open_bc + ['xlo.type=pi', 'xlo.pressure=1'], 'pressure inflow or mixed'),
        ('direction', open_bc + ['xhi.type=direction_dependent'], 'direction_dependent'),
        ('inflow_without_outflow', open_bc + ['xhi.type=nsw'], 'mass inflow needs'),
    ]
    with tempfile.TemporaryDirectory(prefix='incflo-restrictions-') as scratch:
        for name, overrides, diagnostic in cases:
            folder = Path(scratch) / name
            folder.mkdir()
            result = subprocess.run([*launch, *base, *overrides], cwd=folder,
                                    env={**os.environ, 'OMP_NUM_THREADS': '1'},
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=120)
            if result.returncode == 0 or diagnostic not in result.stdout:
                raise RuntimeError(f'{name}: expected rejection containing {diagnostic!r}\n{result.stdout}')
            print(f'PASS {name}: {diagnostic}')
        # A successful S=0 control proves this executable can initialize normally.
        result = subprocess.run([*launch, *base, 'incflo.use_divu_source=0'], cwd=scratch,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=120,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'})
        if result.returncode:
            raise RuntimeError('S=0 startup control failed\n' + result.stdout)
        print('PASS S=0 startup control')


if __name__ == '__main__':
    main()
