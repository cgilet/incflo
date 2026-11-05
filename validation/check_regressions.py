#!/usr/bin/env python3
"""Run the non-EB 3D channel regressions and compare a checkpoint restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def launch(exe, case, folder, params, ranks):
    folder.mkdir(parents=True, exist_ok=False)
    command = ['mpiexec', '-n', str(ranks), str(exe),
               str(ROOT/'test_no_eb_3d'/f'benchmark.divu_source_{case}'), *params]
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (folder/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                timeout=900, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    if result.returncode:
        raise RuntimeError(f'case failed: see {folder}/run.log')
    return (folder/'run.log').read_text()


def check_plot(path, required):
    lines = (path/'Header').read_text().splitlines()
    count = int(lines[1])
    names = set(lines[2:2+count])
    if not required <= names:
        raise RuntimeError(f'{path}: missing variables {required-names}')
    return int(lines[4+count])  # finest AMR level


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--fcompare', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi-ranks', type=int, default=2)
    args = parser.parse_args()
    if args.mpi_ranks < 2:
        parser.error('3D requires at least two MPI ranks')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    required = {'velx', 'vely', 'velz', 'gpx', 'gpy', 'gpz', 'density',
                'tracer0', 'tracer1', 'divu_source'}
    full, restart, plume = (output/name for name in ('temperature', 'restart', 'plume'))
    launch(exe, 'temperature', full, [], args.mpi_ranks)
    if not (full/'chk00005/Level_0/divu_source_H').is_file() or not (full/'chk00005/Level_0/dsdt_H').is_file():
        raise RuntimeError('restart checkpoint is missing source history')
    check_plot(full/'plt00020', required | {'temperature'})
    launch(exe, 'temperature', restart, [f'amr.restart={full}/chk00005'], args.mpi_ranks)
    check_plot(restart/'plt00020', required | {'temperature'})
    comparison = [str(args.fcompare.resolve()), '--abort_if_not_all_found',
                  '--rel_tol', '1e-10', '--abs_tol', '1e-12',
                  str(full/'plt00020'), str(restart/'plt00020')]
    with (output/'restart-comparison.log').open('w') as log:
        result = subprocess.run(comparison, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError('restart mismatch: see restart-comparison.log')
    print('PASS 606: state-dependent source and step-5 restart comparison', flush=True)
    text = launch(exe, 'plume', plume, [], args.mpi_ranks)
    if check_plot(plume/'plt00040', required) < 1:
        raise RuntimeError('AMR plume did not have a fine level at the final time')
    if 'Remaking level' not in text and 'Making new level 1 from coarse' not in text:
        raise RuntimeError('AMR plume did not exercise level creation/remaking during evolution')
    print('PASS 607: two-level AMR plume with regridding')
    (output/'summary.json').write_text(json.dumps(dict(mpi_ranks=args.mpi_ranks,
        restart_comparison=comparison, checks=['606 restart', '607 AMR regrid']), indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        sys.exit(str(exc))
