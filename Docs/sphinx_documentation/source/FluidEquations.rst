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
constant viscosity and :math:`\nabla \cdot U = 0`). With ``incflo.use_divu_source = 1`` (see below),
:math:`\tau` also has the term :math:`- \frac{2}{3} \eta (\nabla \cdot U) I`, which makes it
trace-free.

Velocity constraint:

.. math:: \nabla \cdot U = 0

or, with ``incflo.use_divu_source = 1``, the constraint

.. math:: \nabla \cdot U = S

for a given source :math:`S`, as in low Mach number models. Mass is still conserved, so
the density changes along particle paths as :math:`D\rho/Dt = -\rho S`. With
``incflo.constant_density = 1`` the density is not evolved, so mass is not conserved:
the constraint adds mass at the rate :math:`\rho_0 S` per unit volume. That is not a low
Mach number model, and incflo warns about it, but it has exact solutions that are useful
for testing. :math:`S` comes from ``incflo::compute_divu_source``, which by default
uses the :math:`S` defined for the chosen ``incflo.probtype``: an analytic :math:`S`, except for
``incflo.probtype = 606``, where :math:`S = \kappa (T - T_{ref})` is computed from the temperature.
It is virtual, so a class derived from ``incflo`` can compute :math:`S` from the state instead,
as a low Mach number model would.
With ``incflo.use_divu_source = 1`` the fluid enters the domain through mass inflows and leaves
it through pressure outflows; pressure inflow, mixed and direction-dependent boundaries abort.
The volume that :math:`S` makes has to leave through a boundary where the pressure is set, so mass
inflow needs a pressure outflow, and a closed domain (only walls and periodic boundaries) is not
supported: it aborts unless ``incflo.allow_closed_with_divu_source = 1``. In a closed domain the
projections can only enforce :math:`\nabla \cdot U = S - \bar{S}`, where :math:`\bar{S}` is the
mean of :math:`S`, and incflo removes :math:`\bar{S}` from the :math:`S` that the advection scheme
uses as well, so that the two agree. A nonzero :math:`\bar{S}` is a uniform change of the
thermodynamic pressure, which incflo does not evolve, and incflo does not add the compression
heating that goes with it; only the closed-domain test problems (603 and 604) are validated.

Tests with exact solutions validate ``incflo.use_divu_source``: ``incflo.probtype`` = 603 (a
ballistic potential flow with an evolved density), 604 (a manufactured solution with no-slip walls)
and 606 (a warm layer carried through a channel, with :math:`S` computed from the temperature), with
the inputs ``inputs.divu_source_*`` and the scripts ``test_convergence_divu_source*`` in
``test_no_eb_2d`` and ``test_no_eb_3d``, which describe the results. The regression tests
``divu_source_plume`` (605, with AMR) and ``divu_source_temperature`` (606, with a restart) are in
``test_3d``. They cover Godunov advection with PPM or PLM, the convective or conservative form of
the momentum equation, periodic boundaries, no-slip and slip walls, mass inflow with a pressure
outflow, one or two levels, an evolved or a constant density, tracers in conservative and
convective form and the temperature (without diffusion), explicit, Crank-Nicolson and implicit
viscous terms (the last first order in time) with the tensor or the per-component solve, and an
:math:`S` that is analytic or computed from the state. MOL and BDS advection, the cell-centered
projection, embedded boundaries, RZ, and pressure inflow, mixed and direction-dependent boundaries
abort; other options and combinations are not tested. Known limitations:

- At a pressure outflow the viscous terms take a zero normal derivative of the velocity, which is
  only consistent with :math:`\nabla \cdot U = S` where :math:`S = 0`, so keep :math:`S` zero there.
  At a mass inflow :math:`S` is zero in the tests too.
- Where :math:`S \ne 0` on a no-slip wall, the density and the tracers are first order in the
  cells next to the wall; at slip walls, only the tracers in convective form are.
- With viscosity, the pressure is first order along coarse/fine boundaries; this also happens with
  :math:`S = 0`.
- The terms of the right-hand side of the nodal projection, :math:`\nabla \cdot U^*` and
  :math:`S`, can be much larger than the right-hand side itself, so at fine resolutions its residual
  can stop decreasing above the default ``nodal_proj.mg_rtol``; then relax it (the 606 inputs use
  :math:`10^{-10}`).

Next to walls the velocity and the scalars converge at second order, but with viscosity the
pressure converges at first order in the cells next to a wall wherever
:math:`\partial^2 u_n / \partial n^2 \ne 0` there (where the shear varies along a no-slip wall,
or :math:`\partial S / \partial n \ne 0` at any wall), with or without :math:`S`. The nodal
projection leaves an :math:`O(h^2)` error in the wall-normal velocity next to a wall, because its
divergence at wall nodes treats the first row of cells as uniform over the half control volume, and
the viscous operator, which divides by :math:`h^2` and has its own :math:`O(1)` truncation error at
the wall, turns that into an :math:`O(1)` wall-normal force, which the pressure absorbs since the
constraint fixes the normal velocity. Quadratic ghost-cell extrapolation in the viscous solve
(``tensor_diffusion.mg_maxorder = 3``, or ``scalar_diffusion.mg_maxorder = 3`` with the
per-component solve, instead of the default 2), which the test problems with walls use, makes that
truncation error :math:`O(h)` and the pressure error next to the wall a third smaller, but still
first order.

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
