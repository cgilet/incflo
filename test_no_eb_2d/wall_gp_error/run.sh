#!/bin/bash
# O(1) pressure-gradient error in the cells next to no-slip walls, shown with the Burggraf
# cavity (../benchmark.burggraf, unchanged).  See README.md.  Run from an empty directory:
#
#   INCFLO_EXE=/path/to/incflo2d.gnu.MPI.ex /path/to/run.sh
#
# Optional: NPROCS (MPI ranks, default 2), NS (resolutions, default "32 64 128").
# Each case runs to t = 2 with the benchmark's dt = 0.128/n and prints burggraf_gp.py's table
# for its final plotfiles.
HERE=$(cd "$(dirname "$0")" && pwd)
INPUTS=$HERE/../benchmark.burggraf
NPROCS=${NPROCS:-2}
NS=${NS:-"32 64 128"}
: "${INCFLO_EXE:?set INCFLO_EXE to a 2D incflo executable}"

case_run() {
    name=$1; mu=$2; shift 2
    echo "== $name (mu = $mu): $*"
    plts=""
    for n in $NS; do
        d=${name}_$n
        mkdir -p $d
        (cd $d && mpiexec -n $NPROCS $INCFLO_EXE $INPUTS amr.n_cell="$n $n" \
             incflo.fixed_dt=$(python3 -c "print(0.128/$n)") stop_time=2 \
             amr.plot_int=1000000 incflo.mu=$mu "$@" > run.log 2>&1) || echo "$d FAILED"
        plts="$plts $(ls -d $d/plt* | sort | tail -1)"
    done
    python3 $HERE/burggraf_gp.py $mu $plts
    echo
}

case_run tensor        0.01                                    # incflo's default viscous solve
case_run tensor_mu002  0.02                                    # same c: the error is linear in mu
case_run tensor_mo3    0.01 tensor_diffusion.mg_maxorder=3
case_run scalar        0.01 incflo.use_tensor_solve=0
case_run scalar_mo3    0.01 incflo.use_tensor_solve=0 scalar_diffusion.mg_maxorder=3
