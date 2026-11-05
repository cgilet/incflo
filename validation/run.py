#!/usr/bin/env python3
"""Run div(u)=S validation; fail on incomplete data, rates or offset invariance."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NORMS = ('linf', 'l1', 'l2')


def errors(text, expected_time, expected_fields, levels=1):
    """Read the final report only; reject missing, duplicate and nonfinite entries."""
    result = {}
    for line in text.splitlines():
        if not line.startswith('DIVU_ERROR '):
            continue
        row = dict(item.split('=', 1) for item in line.split()[1:])
        time = float(row['time'])
        if not math.isfinite(time):
            raise ValueError('nonfinite diagnostic time')
        if not math.isclose(time, expected_time, rel_tol=1e-9, abs_tol=1e-12):
            continue
        key = (int(row['level']), row['field'])
        if key in result:
            raise ValueError(f'duplicate final diagnostic: {key}')
        result[key] = {name: float(row[name]) for name in NORMS}
        if any(not math.isfinite(x) or x < 0 for x in result[key].values()):
            raise ValueError(f'invalid norm: {key}')
    wanted = {(lev, field) for lev in range(levels) for field in expected_fields}
    if result.keys() != wanted:
        raise ValueError(f'incomplete final diagnostics at t={expected_time}: '
                         f'missing={wanted-result.keys()}, unexpected={result.keys()-wanted}')
    return result


def compare(a, b, rtol=2e-6, atol=2e-11):
    if a.keys() != b.keys():
        raise ValueError('comparison fields differ')
    for key in a:
        for norm in NORMS:
            if not math.isclose(a[key][norm], b[key][norm], rel_tol=rtol, abs_tol=atol):
                raise ValueError(f'offset invariance failed: {key} {norm}: '
                                 f'{a[key][norm]} vs {b[key][norm]}')


def minimum_rate(case, field, norm):
    # Broad historical lower bounds, not an assertion of order 2 for limited extrema.
    # The wall-pressure profile is explicitly for stock AMReX, without the pending fix.
    if case == 'mms' and field == 'p':
        return 0.6
    if field in ('u', 'v', 'w', 'p'):
        return 1.4
    return {'linf': 0.35, 'l1': 1.35, 'l2': 1.1}[norm]


def check_rates(case, coarse, fine):
    rows = []
    for key in coarse:
        for norm in NORMS:
            a, b = coarse[key][norm], fine[key][norm]
            if max(a, b) < 1e-11:
                continue
            if min(a, b) == 0:
                raise ValueError(f'cannot establish rate from zero error: {key} {norm}')
            rate = math.log2(a/b)
            minimum = minimum_rate(case, key[1], norm)
            rows.append(dict(level=key[0], field=key[1], norm=norm, rate=rate, minimum=minimum))
            if rate < minimum:
                raise ValueError(f'{case} {key} {norm}: rate {rate:.3f} < {minimum}')
    return rows


def run(exe, folder, input_file, params):
    folder.mkdir(parents=True, exist_ok=False)
    command = [*exe, str(input_file), *params]
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (folder/'run.log').open('w') as output:
        result = subprocess.run(command, cwd=folder, stdout=output, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '1'}, timeout=900)
    text = (folder/'run.log').read_text()
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: see {folder}/run.log')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--dim', type=int, choices=(2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory, never overwritten')
    parser.add_argument('--mpi-ranks', type=int, help='MPI ranks (default: 2 in 3D, serial in 2D)')
    parser.add_argument('--full', action='store_true', help='add a finer resolution')
    parser.add_argument('--cases', nargs='+', choices=('ballistic', 'walls', 'mms', 'temperature'),
                        default=['ballistic', 'walls', 'mms', 'temperature'])
    parser.add_argument('--amr', action='store_true', help='add a fixed-dt two-level offset check')
    args = parser.parse_args()
    exe, output = args.exe.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dim = args.dim
    ranks = args.mpi_ranks if args.mpi_ranks is not None else (2 if dim == 3 else 1)
    if ranks < 1 or (dim == 3 and ranks < 2):
        parser.error('3D requires at least two MPI ranks; rank counts must be positive')
    command_exe = ['mpiexec', '-n', str(ranks), str(exe)] if ranks > 1 else [str(exe)]
    inputs = ROOT/f'test_no_eb_{dim}d'
    fields = ['u', 'v'] + (['w'] if dim == 3 else []) + ['p']
    scalar_fields = fields + ['density', 'tracer0', 'tracer1']
    summary = dict(dim=dim, mpi_ranks=ranks, executable=str(exe), rates=[], checks=[])
    # job_info in each plotfile also records the compiled dependency revisions.
    summary['incflo_checkout'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    summary['incflo_dirty'] = bool(subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).strip())
    cache = {}
    for case in args.cases:
        if case == 'walls' and dim == 3:
            # A wall input is built by overriding periodic 603's geometry below.
            filename = 'inputs.divu_source_ballistic'
        else:
            filename = 'inputs.divu_source_' + ('ballistic_walls' if case == 'walls' else case)
        ns = [64, 128] if case == 'temperature' or (case == 'walls' and dim == 2) else ([32, 64] if case == 'walls' or dim == 2 else [16, 32])
        if args.full:
            ns.append(ns[-1]*2)
        previous = None
        expected = fields if case == 'mms' else scalar_fields + (['temperature'] if case == 'temperature' else [])
        end = 1.0 if case == 'mms' else (0.375 if dim == 3 and case in ('ballistic', 'walls') else 0.5)
        for n in ns:
            cells = [n] * dim if case != 'temperature' else [n] + [n//8]*(dim-1)
            params = ['amr.n_cell='+' '.join(map(str, cells)), f'incflo.fixed_dt={0.5/n:.16g}']
            if case == 'walls' and dim == 3:
                params += ['geometry.is_periodic=0 0 0', 'incflo.ic_u=0', 'incflo.ic_v=0', 'incflo.ic_w=0',
                           *[f'{axis}{side}.type=nsw' for axis in 'xyz' for side in ('lo', 'hi')],
                           'tensor_diffusion.mg_maxorder=3']
            text = run(command_exe, output/f'{case}-{n}', inputs/filename, params)
            current = errors(text, end, expected)
            cache[case, n] = (current, params, filename, expected)
            if previous is not None:
                summary['rates'].extend(dict(case=case, n=n, **row) for row in check_rates(case, previous, current))
            previous = current
            print(f'PASS {case} n={n}', flush=True)
        if case == 'ballistic':
            n = ns[0]
            baseline, params, filename, expected = cache[case, n]
            text = run(command_exe, output/'offset-fixed', inputs/filename, params + ['incflo.probtype=604'])
            compare(baseline, errors(text, end, expected))
            summary['checks'].append('fixed-dt source-offset invariance')
            # Adaptive runs isolate timestep behavior as well as solution invariance.
            adaptive = []
            for prob in (603, 604):
                text = run(command_exe, output/f'offset-adaptive-{prob}', inputs/filename,
                           [f'amr.n_cell='+ ' '.join([str(n)]*dim), f'incflo.probtype={prob}'])
                adaptive.append((errors(text, end, expected),
                                 [float(x) for x in re.findall(r'with dt = ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)', text)]))
            compare(adaptive[0][0], adaptive[1][0])
            a, b = adaptive[0][1], adaptive[1][1]
            if not a or len(a) != len(b) or any(not math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-12) for x,y in zip(a,b)):
                raise ValueError('adaptive offset timestep histories differ')
            summary['checks'].append('adaptive source-offset invariance, including dt history')
            if args.amr:
                params += ['amr.max_level=1', 'amr.n_error_buf=0', 'amr.regrid_int=-1',
                           'incflo.tag_region=true', 'incflo.tag_region_lo='+' '.join(['0.25']*dim),
                           'incflo.tag_region_hi='+' '.join(['0.75']*dim)]
                amr = []
                for prob in (603, 604):
                    text = run(command_exe, output/f'offset-amr-{prob}', inputs/filename, params+[f'incflo.probtype={prob}'])
                    amr.append(errors(text,end,expected,levels=2))
                compare(*amr)
                summary['checks'].append('AMR offset invariance (integration, not convergence)')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'PASS validation; results: {output}/summary.json')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        sys.exit(str(exc))
