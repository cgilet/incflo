Fluid Variables
===============

   +-----------------------+---------------------------------------------------------------------+
   | Variable              | Definition                                                          |
   +=======================+=====================================================================+
   | :math:`\rho`          | Fluid density                                                       |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`U`             | Fluid velocity                                                      |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`p`             | Pressure                                                            |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`\eta`          | Viscosity (``incflo.mu`` for a Newtonian fluid)                     |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`\tau`          | Viscous stress tensor, :math:`\eta (\nabla U + (\nabla U)^T)`       |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`\mu_s`         | scalar diffusivity                                                  |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`\mu_T`         | thermal conductivity                                                |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`c_p`           | Specific heat (``incflo.cp``)                                       |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`{\bf g}`       | Gravitational acceleration                                          |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`{\bf H}_U`     | :math:`= (H_x , H_y , H_z )`, External Forces                       |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`H_s`           | External sources                                                    |
   +-----------------------+---------------------------------------------------------------------+
   | :math:`H_T`           | External heat sources                                               |
   +-----------------------+---------------------------------------------------------------------+

Fluid Equations
===============

Conservation of fluid mass:

.. math:: \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho U)  = 0

Conservation of fluid momentum:

.. math:: \frac{ \partial (\rho U)}{\partial t}
   + \nabla \cdot (\rho U U) + \nabla p = \nabla \cdot \tau  + {\bf H}_U

By default, incflo solves the equivalent convective form,
:math:`\rho ( \partial U / \partial t + U \cdot \nabla U ) + \nabla p = \nabla \cdot \tau + {\bf H}_U`;
``incflo.advect_momentum = true`` selects the conservative form above.
With ``incflo.use_tensor_solve = false``, the velocity components are diffused separately, with
:math:`\nabla \cdot (\eta \nabla U)` in place of :math:`\nabla \cdot \tau` (the two are equal for a
constant viscosity and :math:`\nabla \cdot U = 0`). With ``incflo.use_divu_source = 1`` (see
:ref:`sec:DivuSource`), :math:`\tau` also has the term
:math:`- \frac{2}{3} \eta (\nabla \cdot U) I`, using the three-dimensional deviatoric-stress convention.

Velocity constraint:

.. math:: \nabla \cdot U = 0

or, with ``incflo.use_divu_source = 1``, :math:`\nabla \cdot U = S` (see :ref:`sec:DivuSource`).

Tracer(s): for conservative,

.. math:: \frac{\partial \rho s}{\partial t} + \nabla \cdot (\rho U s)  =  \nabla \cdot \mu_s \nabla s + \rho H_s

or, for non-conservative (see ``incflo.trac_is_conservative``),

.. math:: \frac{\partial s}{\partial t} + U \cdot \nabla s  =  \nabla \cdot \mu_s \nabla s + H_s

Optional temperature:

.. math:: \rho c_p \left( \frac{\partial T}{\partial t} + U \cdot \nabla T \right)  =  \nabla \cdot \mu_T \nabla T + H_T

By default, :math:`H_s = 0`,  :math:`H_T = 0`, and :math:`{\bf H}_U = {\bf 0}`.
If gravity is set (``incflo.gravity``), then :math:`{\bf H}_U = \rho {\bf g}`, except for the
Boussinesq problems (``incflo.probtype`` = 11, 111, 112, 113 and 500), which use
:math:`{\bf H}_U = \rho s_0 {\bf g}`, where :math:`s_0` is the first tracer.

.. _sec:BackgroundPressure:

Background pressure
-------------------

Except for the Boussinesq problems, incflo writes the pressure as :math:`p = p_0 + p'`, where the
background pressure :math:`p_0` has a constant gradient, and solves for :math:`p'`:

- in a direction with gravity, :math:`\nabla p_0 = \rho_0 {\bf g}`, the hydrostatic pressure gradient
  for the density :math:`\rho_0` (``incflo.ro_0``);
- in the direction in which ``incflo.delp`` is non-zero, :math:`\nabla p_0 = - \mathrm{delp} / L`,
  where :math:`L` is the length of the domain in that direction. This pressure drop drives the flow,
  e.g. in a periodic channel;
- in a direction with a pressure inflow at one end and a pressure outflow at the other,
  :math:`\nabla p_0 = (p_{hi} - p_{lo}) / L`, where :math:`p_{lo}` and :math:`p_{hi}` are the
  boundary pressures.

Only one direction can have a ``delp`` or a pressure inflow/outflow pair, and gravity cannot be in
that direction. The pressure written to plotfiles (``p``) is :math:`p'`, while the pressure gradient
(``gpx``, ``gpy`` and ``gpz``) includes :math:`\nabla p_0`.

.. _sec:DivuSource:

Divergence constraint with a source
-----------------------------------

With ``incflo.use_divu_source = 1``, incflo enforces

.. math:: \nabla \cdot U = S

instead of :math:`\nabla \cdot U = 0`, as low Mach number models do: :math:`S` is the rate at which
the fluid expands, or contracts where it is negative. Mass is still conserved, so the density
changes along particle paths as :math:`D\rho/Dt = -\rho S`. With ``incflo.constant_density = 1``
the density is not evolved, and the constraint adds mass at the rate :math:`\rho_0 S` per unit
volume. That is not a low Mach number model, and incflo warns about it, but it has exact solutions
that are useful for testing.

:math:`S` comes from the virtual function ``incflo::compute_divu_source``, which is given the
density, the tracers and the temperature, each with one filled ghost cell. By default it uses the
:math:`S` of the chosen ``incflo.probtype``, which is analytic except for
``incflo.probtype = 606``, where :math:`S = \kappa (T - T_{ref})` is computed from the temperature.
A class derived from ``incflo`` can override it to compute :math:`S` from the state, as a low Mach
number model would. A positive :math:`S` also limits the time step computed from the CFL condition
(see :ref:`sec:InputsTimeStepping`).

Boundaries
~~~~~~~~~~

The domain must be open: the fluid enters through mass inflows and leaves through pressure
outflows, and the other boundaries are walls or periodic. A mass inflow needs a pressure outflow,
through which the volume that :math:`S` makes can leave. Pressure inflow, mixed and
direction-dependent boundaries abort. At a pressure outflow the viscous terms take the normal
derivative of the velocity to be zero, which agrees with :math:`\nabla \cdot U = S` only where
:math:`S = 0`, so keep :math:`S` zero at pressure outflows. In the tests it is zero at mass inflows
too.

General closed domains, with only walls and periodic boundaries, abort.
Only built-in analytic problems 603–605 may explicitly set
``incflo.divu_source_allow_closed = 1`` to exercise the projection numerically. Its projections can only enforce
:math:`\nabla \cdot U = S - \bar{S}`, where :math:`\bar{S}` is the mean of :math:`S`, and incflo
removes :math:`\bar{S}` from the :math:`S` of the advection scheme as well, so that the two agree. A
nonzero :math:`\bar{S}`, however, is a uniform change of the thermodynamic pressure, which incflo
does not evolve, and incflo does not add the compression heating that goes with it. This mathematical exception is not a closed-chamber physical model: EOS coupling,
thermodynamic pressure evolution and matching compression heating are absent.

Test problems
~~~~~~~~~~~~~

The test problems ``incflo.probtype`` = 603 to 606 have exact solutions:

- 603: a ballistic potential flow, periodic or with walls, with an evolved density and tracers;
- 604: 603 with a constant, 1.54, added to :math:`S`, which the projections of the closed domain
  remove, so that every error must be the same as in 603;
- 605: a steady manufactured solution with no-slip walls and a velocity forcing;
- 606: a warm layer carried through a channel from a mass inflow to a pressure outflow, with
  :math:`S = \kappa (T - T_{ref})` computed from the temperature.

Their inputs ``inputs.divu_source_*`` and the scripts ``test_convergence_divu_source`` and
``test_convergence_divu_source_options`` are in ``test_no_eb_2d`` and ``test_no_eb_3d``. The scripts call ``validation/run.py``, which checks final diagnostics, convergence
criteria and fixed/adaptive-timestep source-offset invariance. Linf, L1 and L2 reports
are per level; they are not composite AMR norms. See ``validation/README.md`` for
commands, evidence and the distinction between acceptance checks and characterization. The test problem 607, a pulsing source in a channel with no-slip walls,
refined with AMR, has no exact solution. Its non-EB regression inputs and portable configuration are described in
``regression/README.md``; adding an example entry does not enroll a nightly test.

The historical studies exercise Godunov advection with PPM or PLM; the convective and the conservative form of
the momentum equation; periodic boundaries, no-slip and slip walls, and mass inflow with a pressure
outflow; one or two levels; an evolved or a constant density; tracers in conservative and in
convective form and the temperature, without diffusion; explicit, Crank-Nicolson and implicit
viscous terms (the last first order in time), with the tensor or the per-component solve; and an
:math:`S` that is analytic or computed from the state. MOL and BDS advection, the cell-centered
projection, embedded boundaries, RZ and refinement ratios other than 2 or 4 abort; passing startup does not establish validation of every combination. Non-Newtonian
rheology with nonzero divergence has no validation claim here.

Accuracy and limitations
~~~~~~~~~~~~~~~~~~~~~~~~

**Scalars at walls.** Without a wall value, the boundary conditions of the scalars at walls assume
that their normal derivative there is zero:

- at no-slip walls, the ghost cells of the density and of the tracers copy the cell next to the
  wall, which keeps them bounded;
- at any wall, the ghost cells of the temperature mirror it;
- with PPM, the state on the wall face of a scalar in convective form (a tracer with
  ``incflo.trac_is_conservative = 0``, or the temperature) is the cell average next to the wall.

That is right for a tracer or the temperature that diffuses (``incflo.mu_s`` or ``incflo.mu_T``
above zero), since it then has no flux through the wall. For the density, which does not diffuse,
and for a tracer or the temperature that does not, it is wrong by about :math:`h` times the normal
derivative. That is harmless where :math:`S = 0` on a no-slip wall: the normal velocity then grows
quadratically from the wall, so the error reaches the fluxes only at second order, and the state on
the wall face enters the convective term only multiplied by :math:`\nabla \cdot U`. Where
:math:`S \ne 0` on the wall, the normal velocity grows linearly from it, and these scalars are first
order in the cells next to the wall wherever their normal derivative there is not zero. So are the
tracers in conservative form with a wall value at no-slip walls, wherever the normal derivative of
the density is not zero, since they are advected as the density times the tracer. At a slip wall
the normal velocity also grows linearly wherever the tangential velocity varies along the wall, so
there a temperature that does not diffuse is first order even with :math:`S = 0`. The test problem
603 gives the density and the tracers zero normal derivatives on the walls, so that the walls do not
limit their convergence.

The right boundary condition for these scalars is their value on the wall. The fluid on a wall that
does not move stays there, so a tracer that does not diffuse keeps its initial value there, and the
density changes as :math:`d\rho/dt = -\rho S`; on a slip or moving wall the values are also carried
along the wall. incflo can give a tracer or the temperature one constant value per wall
(``xlo.tracer``, ``xlo.temperature`` and so on), which is right if it starts out uniform along that
wall, and PPM then uses that value as the state on the wall face. The density cannot be given one.

**Extrapolation at slip walls.** At slip walls the ghost cells of the density and of the tracers
without a wall value are extrapolated to second order (``hoextrap``), which is accurate whatever
their normal derivative: the density and the tracers in conservative form are second order next to
slip walls, and the same extrapolation at no-slip walls (the boundary types are set in
``incflo::init_bcs``) would make them second order there too. PLM (``incflo.use_ppm = 0``)
extrapolates the state on the wall face as well, which makes the tracers in convective form second
order next to slip walls.

The extrapolated value can lie outside the range of the data, though, with or without :math:`S`:
densities of 0.01, 1 and 1 in the first three cells give -0.86. PPM uses it, unlimited, as the
value at the wall of the profile in the cell next to the wall, and PLM uses it as the value at the
wall when it limits the slope of that cell. A minimum in that cell then looks like part of a profile
that rises away from the wall, and a maximum like part of one that falls. Where the fluid next to
the wall moves away from it, the cell sends into the domain, through its face opposite the wall, a
state from the far end of that profile instead of its own value. That empties a minimum below the
range of the data and fills a maximum above it: a small density next to a much larger one can
become negative. How far depends on how fast the normal velocity grows away from the wall.
:math:`S` adds to that rate, but so does a tangential velocity that varies along the wall, and with
:math:`S = 0` the overshoot is about as large for the same rate. Copying the cell next to the wall,
as at no-slip walls, keeps the ghost cells bounded.

**Physical-wall pressure gradients.** Historical runs on AMReX ``47ceffc47`` show
first-order pressure and nonconverging wall-normal pressure-gradient errors in viscous
wall cells where the normal velocity has curvature, including S=0 flows. Velocity can
remain second order. The nodal wall divergence and viscous boundary stencil both
contribute. An independent AMReX correction is pending, with matching Hydro support.
The local interface is ``nodal_proj.wall_divu_correction`` (default off); its full
edge/corner mode preserves the discrete flux sum. The final merged API and dependency
versions must be checked before interpreting results as corrected. Quadratic diffusion
boundary extrapolation (``tensor_diffusion.mg_maxorder = 3``, or the scalar equivalent)
addresses a separate contribution and does not alone fix the nodal divergence.
These incflo changes do not enable an unmerged dependency feature or change solver defaults.

**Coarse/fine boundaries.** Known localized nodal divergence errors at fine-patch corners
and viscous-interface pressure/gradient accuracy loss remain outside this work. These
are distinct from physical-wall quadrature; the wall correction does not repair the
rebuilt coarse/fine RHS. The complete evidence, corrected interpretation and future
investigation are recorded in ``validation/COARSE_FINE.md``. AMR regression agreement
must not be interpreted as uniformly second-order interface accuracy.

**Solver tolerance.** The right-hand side of the nodal projection is the difference of
:math:`\nabla \cdot U^*` and :math:`S`, which can be much smaller than either, so at fine
resolutions MLMG may stop converging above the default ``nodal_proj.mg_rtol``. Then relax it; the
606 inputs use :math:`10^{-10}`.

Validation scope and evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact solutions compare analytic point values at cell centers; the reported L1
is the mean absolute error and L2 the RMS over valid cells on each level. Pressure
is compared at its half-step time. Its arbitrary gauge is removed only when no boundary
prescribes pressure; open-channel errors retain the prescribed absolute pressure level.
An initial pressure output is not a final-time acceptance measurement.

Automatic timesteps use positive expansion of the old-time source, with the composite
mean removed only for the closed analytic exception. Fixed timesteps bypass automatic
estimation. Original S remains in checkpoints and plotfiles; the half-time correction
used by MAC projection and advection is printed separately with its time and size relative
to max(abs(S)). This is a solvability adjustment, not a projection residual.

Legacy problem definitions 600–602 are retained, but the new restrictions intentionally
reject their unsupported configurations. Use 603 for closed analytic validation and 606
for an open source with zero S at the pressure outflow. In particular 602 prescribes
nonzero S at that outflow and now fails explicitly.

Historical convergence tables are not results from this branch. Broad acceptance criteria
in ``validation/run.py`` distinguish scalar limiter effects and stock-AMReX wall-pressure
limits; they do not assert second order for all fields and norms. Implicit diffusion is
first order in time. Historical explicit 3D case 605 pressure is timestep-sensitive and
needs separate dt refinement; it is not covered by a blanket explicit-scheme accuracy claim.
