#!/usr/bin/env python3
"""Compare evolved 2D S=0 controls with a baseline built using identical dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for case in ('taylor_green_vortices', 'bouss_bubble_god'):
        plots = []
        for label, exe in (('baseline', args.baseline), ('candidate', args.exe)):
            folder = output/f'{case}-{label}'
            folder.mkdir()
            command = [str(exe.resolve()), str(ROOT/'test_no_eb_2d'/f'benchmark.{case}'),
                       'incflo.use_divu_source=0', 'amr.n_cell=32 32', 'amr.max_level=0',
                       'max_step=20', 'stop_time=-1', 'incflo.fixed_dt=0.005',
                       'amr.plot_per_exact=0', 'amr.plot_per_approx=0', 'amr.plot_int=20',
                       'amr.check_int=-1']
            (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            with (folder/'run.log').open('w') as log:
                subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                               timeout=120, check=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
            plots.append(folder/'plt00020')
        # With identical single-process grids, compare field data exactly; ignore job_info.
        files = ['Header', *[str(p.relative_to(plots[0])) for p in sorted((plots[0]/'Level_0').glob('Cell*'))]]
        if len(files) < 3:
            raise RuntimeError(f'{case}: incomplete plotfile')
        for name in files:
            if (plots[0]/name).read_bytes() != (plots[1]/name).read_bytes():
                raise RuntimeError(f'{case}: S=0 baseline differs in {name}')
        print(f'PASS {case}: identical final plotfile headers and field data')


if __name__ == '__main__':
    main()
