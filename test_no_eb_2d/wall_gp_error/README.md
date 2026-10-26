# O(1) pressure-gradient error next to no-slip walls

In every viscous incflo run with no-slip walls, the wall-normal component of `gp` has an error
in the cells next to the wall that does not converge with resolution:

gp_n - dp/dn = -c mu u_n''

Here u_n'' is the second normal derivative of the normal velocity at the wall, and c depends
only on the viscous solve. The pressure is first order next to the wall, and the velocity
stays second order.

The unchanged benchmark `../benchmark.burggraf` (probtype 16) shows it. It is a steady flow in
a lid-driven cavity with an exact solution, and u_n'' is not zero on any wall. incflo has no
exact pressure for it. `burggraf_gp.py` uses the one from Shih, Tan & Hwang (1989), which
satisfies the momentum equations exactly with incflo's forcing.

## Run

Build `test_no_eb_2d` as usual, then from an empty directory:

```shell
INCFLO_EXE=/path/to/incflo2d.gnu.MPI.ex /path/to/test_no_eb_2d/wall_gp_error/run.sh
```

It runs the benchmark to t = 2 at n = 32, 64 and 128 for five cases: the default tensor
solve, the same with twice the viscosity, `tensor_diffusion.mg_maxorder = 3`, the
per-component (scalar) solve, and the scalar solve with `scalar_diffusion.mg_maxorder = 3`.
The 15 runs take about 4 minutes on 2 ranks. `NPROCS` and `NS` set the number of ranks and
the resolutions.

## Output (incflo 18045d08, AMReX 1aebac90c, AMReX-Hydro a153304)

```
== tensor (mu = 0.01):
    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners
   32        0.668   0.666   0.670               1.15e-01   1.56e-02    2.40e-01
   64        0.667   0.666   0.666               1.06e-01   5.59e-03    2.50e-01
  128        0.667   0.666   0.666               1.04e-01   1.57e-03    2.54e-01

== tensor_mu002 (mu = 0.02):
    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners
   32        0.667   0.665   0.665               2.17e-01   1.70e-02    4.74e-01
   64        0.667   0.666   0.665               2.09e-01   5.64e-03    4.94e-01
  128        0.667   0.666   0.666               2.06e-01   1.55e-03    5.01e-01

== tensor_mo3 (mu = 0.01): tensor_diffusion.mg_maxorder=3
    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners
   32        0.499   0.448   0.469               8.45e-02   1.48e-02    1.98e-01
   64        0.472   0.445   0.455               7.35e-02   5.18e-03    1.99e-01
  128        0.458   0.444   0.450               7.02e-02   1.45e-03    1.97e-01

== scalar (mu = 0.01): incflo.use_tensor_solve=0
    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners
   32        0.502   0.505   0.508               9.00e-02   1.56e-02    1.98e-01
   64        0.500   0.501   0.501               8.02e-02   5.62e-03    2.03e-01
  128        0.500   0.500   0.500               7.78e-02   1.58e-03    2.05e-01

== scalar_mo3 (mu = 0.01): incflo.use_tensor_solve=0 scalar_diffusion.mg_maxorder=3
    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners
   32        0.362   0.341   0.351               6.58e-02   1.48e-02    1.71e-01
   64        0.348   0.335   0.340               5.54e-02   5.21e-03    1.70e-01
  128        0.341   0.334   0.336               5.27e-02   1.46e-03    1.67e-01
```

- **c** comes from a least-squares fit of the wall-normal gp error in the wall cells, on each
  wall, leaving out n/8 cells next to each corner.
- **max |gp_n err|** is the largest wall-normal gp error in the wall cells and in the second
  row of cells, with the same cells left out.
- **lid corners** is the largest gp error (either component) in the two cells at the lid
  corners.

What it shows:

- **c is the same on every wall,** and does not change with n or mu. It is 2/3 for the tensor
  solve, 4/9 with `mg_maxorder = 3`, 1/2 for the scalar solve and 1/3 for the scalar solve
  with `mg_maxorder = 3`. With `mg_maxorder = 3` it approaches these values at O(h).
- **The error is in the wall cells only.** In the second row it converges at close to second
  order.
- **The lid-corner cells are worse:** 0.17 to 0.25 depending on the solve, also not
  converging, and also proportional to mu.

## Cause

Two errors add up in the wall cells. Both come from AMReX.

1. **The nodal projection** (`MLNodeLaplacian`, `mlndlap_divu`). At a node on a wall, the
   divergence takes the tangential velocity of the first row of cells as constant over the
   half control volume. After the projection, the wall-normal velocity in the first few cells
   is too large by delta = (h^2/8) u_n''. That is second order, but it is nearly the same in
   each cell, so the 1/h^2 viscous stencil turns it into an O(1) force normal to the wall.
2. **The viscous operator** (`MLABecLaplacian`, `MLTensorOp`). With `mg_maxorder = 2`, the
   default, its Dirichlet wall stencil has an O(1) truncation error in the wall cell, even
   for the exact velocity. With 3 it is O(h).

Each piece, as its contribution to c:

| viscous solve | `mg_maxorder` | (1) projection | (2) operator | c |
|---|---|---|---|---|
| scalar | 2 | 1/4 | 1/4 | 1/2 |
| scalar | 3 | 1/3 | 0 | 1/3 |
| tensor | 2 | 1/3 | 1/3 | 2/3 |
| tensor | 3 | 4/9 | 0 | 4/9 |

The tensor values are 4/3 of the scalar ones.

On its own, (2) would only cost an O(h^2) error in the velocity. But the projection fixes the
wall-normal velocity, so a wall-normal force error cannot be balanced by the velocity. The
pressure at the wall nodes takes it instead. That gives the O(1) gp_n.

An AMReX-only test reproduces both pieces and the four c values above, without incflo or time
stepping: `Tests/LinearSolvers/NodalProjectionWall` on the `nodal_wall_reproducer` branch of
https://github.com/cgilet/amrex. Its README has the details.

## Fix: not yet

A local AMReX change makes the wall divergence second order: at nodes on exactly one wall, it
takes the tangential velocity a quarter cell from the wall, extrapolated from the first two
rows of cells. With it and `mg_maxorder = 3`, c goes to zero at O(h) along the walls.

**Corners remain a problem.** The change keeps the original stencil at corner nodes. Since
u_n'' is not zero at the lid corners, the error in those cells only drops from about 0.25 to
0.165, 0.143 and 0.121 at n = 32, 64 and 128. Using the extrapolation at the corners as well
makes them converge at first order (0.044, 0.023, 0.011). But in a manufactured solution with
div(u) = S != 0, the corner errors then stop converging. So the change is not proposed yet.
