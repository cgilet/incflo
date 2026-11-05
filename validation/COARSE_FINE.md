# Deferred coarse/fine boundary investigation

Prepared 2026-11-05 from local source and archived investigations. No new numerical runs.
This dossier preserves known issues for later examination; fixing them is outside the incflo
divergence-source validation series. AMR regression coverage remains useful with these caveats.

## Keep three effects distinct

1. Composite nodal divergence has a localized corner error even for smooth divergence-free
   shear. Strong evidence places this in shared AMReX nodal projection code.
2. Viscous multilevel time integrations show local pressure/gradient accuracy loss at C/F
   interfaces. The combination of projection and viscous discretization is implicated, but
   the full causal decomposition is not yet established by the recorded runs.
3. The pending physical-wall divergence correction does not repair rebuilt C/F nodal RHS
   contributions. Where a wall meets a C/F interface, both mechanisms can be present.

Do not describe all three as one proven bug, or as a defect caused uniquely by nonzero S.

## 1. Nodal projection corner error

An older incflo/IAMR reproducer projects the same smooth periodic potential flow in two ways:

- Velocity path: start from u=grad(phi), request S=0; the exact projected velocity is zero.
- Cell-source path: start from zero velocity, request S=lap(phi); the exact projected velocity
  is grad(phi). These solve for opposite-signed potentials under the projector convention,
  but compare the same discrete-gradient approximation error up to sign.

Use a unit square, static ratio-2 fine patch [0.25,0.75]^2, one projection (`max_step=0`),
and `nodal_proj.mg_rtol=1e-11`. The potential satisfies
lap(phi)=cos(2*pi*x) cos(2*pi*y). Recorded max errors:

| Coarse n | Multilevel source path | Multilevel velocity path | Velocity path L1 |
| --- | --- | --- | --- |
| 32 | 6.13e-4 | 5.95e-4 | 1.49e-4 |
| 64 | 1.55e-4 | 2.84e-4 | 3.73e-5 |
| 128 | 3.91e-5 | 1.39e-4 | 9.33e-6 |
| 256 | 9.79e-6 | 6.86e-5 | 2.34e-6 |

Single-level velocity-path max errors are 2.5e-4, 6.4e-5, 1.6e-5, 4.0e-6.
Thus the multilevel velocity-path maximum is approximately first order; source-path and
single-level maxima are second order. Largest velocity errors were at fine-patch corners.
The archived note reports second-order L1/L2; the table above gives only measured L1 values.
It also records error propagation away from the interface; localization of the maximum
does not imply the rest of the solution is untouched.

IAMR and incflo gave the same printed digits for the one-projection test. That supports
shared AMReX ownership much more strongly than an application-only multistep comparison.

### Mechanism and its limits

`MLNodeLaplacian::compRHS` assembles contributions from coarse and fine cell-centered velocity
to a nodal RHS suitable for its multigrid hierarchy. It is not a pointwise composite divergence.
The archived independent `cf_rhs.py` reconstructs the same weighted contributions.

Treating velocity as constant within each cell loses cross-derivative terms. At uniform nodes
their leading terms cancel among equal-size quadrants. The leading cancellation also holds
on straight interface segments, but fails at a fine-patch corner with unequal-size quadrants.
For a 2D corner with one fine quadrant, the derived leading term is

    +/- (1/12) (1 - 1/r^2) (du/dy + dv/dx),

with sign depending on corner orientation and the reproducer's normalization. At r=2 this
is +/- (du/dy + dv/dx)/16. The recorded example has corner strain equal to 1 and nodal
RHS errors approaching 0.0625 as n grows from 32 to 256. Straight-edge RHS error is O(H).
The cell-source path's corner error is O(H^2) in that experiment.

An O(1) RHS defect concentrated at a node has integrated strength O(H^2) in 2D. Its gradient
near the node is O(H), explaining a first-order local velocity maximum. Spatially integrated
norms can converge much faster; a 2D point-source scaling can introduce logarithmic factors,
so the observed near-second-order L2 behavior is not a universal theorem for all hierarchies.

Important correction to the older investigation: the trigger is shear/strain, not div(u)
alone. A translating Taylor vortex with du/dy+dv/dx=0 can miss it. A divergence-free shear
with nonzero cross derivatives produces the leading corner defect. Do not use the old
claim that only low-Mach applications receive an O(1) contribution each projection.

The formula's ratio-4 extension is derived, not numerically verified in that old record.
General 3D patch edges/corners, complex patch topology, variable coefficients and EB were
not established by that analysis. The newer wall reproducer adds some 3D extrusion evidence,
not a complete general 3D study.

## 2. Viscous pressure/gradient behavior on AMR

The candidate's ballistic two-level studies report pressure approximately first order along
interfaces and poorer maxima near patch corners, while bulk velocity/scalar norms behave
better. The candidate states that gradient error does not converge and links it to projection
plus viscosity. Its comments also describe substantially smaller errors with viscosity off.

Keep these as historical observations and a mechanism to test. The inviscid one-projection
corner experiment proves that viscosity is not required for the localized divergence issue.
It does not by itself establish the pressure error along every straight interface, its exact
asymptotic order, or whether source interpolation, coefficient interpolation, viscous stencils
and pressure time staggering each contribute in the time-dependent problem.

Per-level RMS norms in existing incflo logs include coarse cells under fine patches; they
are not automatically composite-domain norms. Future comparisons need fine masks and
volume weights, plus explicit interface/edge/corner bands.

Practical consequence: an AMR run can conserve and reproduce a benchmark yet retain a local
first-order velocity error or inaccurate pressure force. Do not advertise uniformly second-
order pressure gradients at refinement interfaces or use those values as unqualified wall-
load/force evidence.

## 3. Intersection with the pending physical-wall fix

Latest inspected AMReX worktree: `/home/gcse/amrex_wall_divu`, HEAD `24218e4fc`.
Matching Hydro wrapper: `/home/gcse/AMReX-Hydro-wall_divu`, HEAD `84e653d`.

AMReX adds `setWallDivuCorrection(int)`; Hydro exposes `nodal_proj.wall_divu_correction`.
The default is off. Mode 1 corrects nodes on one wall/inflow face but leaves physical edges
and corners unchanged; mode 2 also treats those nodes and preserves the discrete flux sum.
This corrects physical-boundary quadrature for velocity and cell-centered source consistently.

However, at a C/F node `compRHS` subsequently reconstructs the composite coarse RHS using
`mlndlap_divu_cf_contrib` and `mlndlap_divu_fine_contrib`. That assembly does not include the
new physical-wall correction. This is why enabling the wall option cannot be assumed to
fix C/F nodes, even where the interface meets a wall.

The updated AMReX-only test includes ratio-2 refinement touching the bottom wall. In its
linear shear u=(y,0), physical-wall curvature is zero, so no wall correction is needed, yet
recorded C/F corner RHS values remain -0.125, +0.125 on the wall and +0.0625 at an interior
patch corner across resolutions. For the divergence-free quadratic field u=(x*y,-y^2/2),
ordinary wall nodes improve, while C/F corners retain O(1) RHS and straight C/F segments
retain O(H) errors. It reports the same output in 2D and 3D with periodic extrusion.

Those results separate physical-wall quadrature from C/F assembly. They do not establish
that every remaining viscous interface-pressure issue has the same cause.

## Source entry points for later work

Names are more durable than archived line numbers:

- AMReX `Src/LinearSolvers/MLMG/AMReX_MLNodeLaplacian_misc.cpp`:
  `MLNodeLaplacian::compRHS`, wall-correction pass, composite RHS rebuilding.
- `AMReX_MLNodeLap_2D_K.H` and `AMReX_MLNodeLap_3D_K.H`:
  `mlndlap_divu_fine_contrib`, `mlndlap_divu_cf_contrib`, `mlndlap_sum_Df`,
  compared with `mlndlap_rhcc_fine_contrib` and corresponding source kernels.
- incflo `src/projection/incflo_apply_nodal_projection.cpp`: absolute/incremental source
  construction and density coefficient; source is -S or -(Snew-Sold) for the projector.
- incflo `src/convection/incflo_compute_advection_term.cpp`: face-velocity ghost filling,
  divergence-preserving interpolation, source used in prediction versus discrete divergence
  used in convective correction. These are distinct from the nodal C/F corner issue.
- incflo `src/incflo_regrid.cpp`: S and dSdt are interpolated on regrid, rather than immediately
  recomputed from regridded state. Characterize this for nonlinear state sources separately.

Do not replace a projected-velocity RHS with an independently evaluated divergence merely
because one reproducer improves: compatibility with the operator, gradient, nullspace,
multigrid restriction and conservation must be checked across applications.

## Preserved evidence and reproduction resources

Older investigation directory: `/home/gcse/incflo_fork/amrex-cf-divergence/`:

- `README.md`: numerical tables, derivation and correction of the earlier causal claim.
- `cf_rhs.py`: independent NumPy RHS reconstruction.
- `incflo_probtype699.patch`, `iamr_probtype14.patch`, `inputs.iamr_p14`, `run.sh`:
  one-projection application comparisons.
- `pltread.py`, `norms_zero.py`: plotfile parsing and regional norms.

Recorded versions: AMReX `47ceffc47` (26.10-74), Hydro `a153304`, IAMR `774ddb30`,
incflo `863fa052` on the old `divu_source` branch. The patches target that old layout and
600-series cases; do not assume they apply to the new branch. Port the minimal reproducer
separately for a future AMReX investigation rather than putting experimental probtype 699
into the present validation series. Archive a portable copy of these resources and exact
revision provenance with that future investigation so this record does not depend on a
particular home-directory checkout.

Later evolution studies: `/home/gcse/incflo_fork/divu-convergence/` and its README, tables,
logs, `extra/results.txt`. Its README identifies incflo `9d01c184`, AMReX `47ceffc47`, Hydro
`fa061e7`. These results predate the new series; correspondence to later amended commits
must be verified before assigning them a new revision label.

Current standalone wall/C/F reproducer and documentation:
`/home/gcse/amrex_wall_divu/Tests/LinearSolvers/NodalProjectionWall/`.
Its README contains older excerpts as well as newer mode-2/conservation/AMR sections; inspect
the section/version before treating an old corner statement as the status of the latest fix.

## Future investigation and acceptance criteria

1. Freeze exact dependency revisions and retain the unfixed control. Reproduce one projection
   with source-path and velocity-path controls, then an S=0 shear that has nonzero corner
   strain. Record nodal RHS and corrected velocity, not only a final plotfile difference.
2. Vary n, patch position, grid-box decomposition, ratio 2/4, coefficient constancy and
   multiple MPI decompositions. Add genuinely 3D shear/edge/corner configurations; periodic
   extrusion alone does not exercise every 3D contribution.
3. Separate corners, straight interfaces, physical walls, intersections and interior. Report
   nonoverlapping composite Linf/L1/L2 norms plus regional norms with specified band widths.
4. Compare wall correction off/on using the final merged API and diffusion maxorder 2/3.
   Keep single-level wall acceptance independent of known C/F-node behavior.
5. Add one-step and time-dependent viscous/inviscid controls, explicit/CN where appropriate,
   constant/variable density, analytic/state-dependent S and S=0. Refine dt separately where
   pressure time discretization could obscure spatial error.
6. Investigate candidate corrections only after verifying operator compatibility, weighted
   RHS conservation and nullspace solvability. Check convergence of velocity, pressure and
   gradient rather than tuning a cancellation in one exact flow.
7. Carry the eventual fix in a separate AMReX PR with an AMReX-only regression and cross-
   application evidence. Keep application adoption/version documentation separate.

No C/F fix is a prerequisite for the current incflo validation/regression work, provided
its support claims and numerical acceptance bounds explicitly retain these limitations.
