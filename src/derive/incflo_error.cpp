#include <AMReX_Box.H>

#include <incflo.H>
#include <incflo_derive_K.H>
#include <prob_divu_ballistic.H>
#include <prob_divu_mms.H>
#include <prob_divu_temperature.H>

using namespace amrex;

void incflo::DiffFromExact (int /*lev*/, Geometry& lev_geom, Real time, Real dt,
                            MultiFab& error, int soln_comp, int err_comp) const
{
    auto const& dx = lev_geom.CellSizeArray();
    auto const& problo = lev_geom.ProbLoArray();

    // err_comp: 0 to AMREX_SPACEDIM-1 velocity, AMREX_SPACEDIM pressure, AMREX_SPACEDIM+1
    // MAC pressure, AMREX_SPACEDIM+2 density, AMREX_SPACEDIM+3+n tracer n and
    // AMREX_SPACEDIM+3+ntrac temperature
    if (err_comp >= AMREX_SPACEDIM+2 && m_probtype != 603 && m_probtype != 606) {
        amrex::Abort("error_density, error_tracer and error_temperature are only implemented for "
                     "incflo.probtype = 603 and 606");
    }
    bool const is_temperature = (err_comp == AMREX_SPACEDIM+3+m_ntrac);
    int const itrac = is_temperature ? -1 : err_comp - (AMREX_SPACEDIM+3);
    if (itrac >= m_ntrac) {
        amrex::Abort("error_tracer: no such tracer");
    }

    // Taylor-Green vortices
    if (1 == m_probtype)
    {
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real  twopi = Real(2.0)*Real(3.1415926535897932);
                constexpr Real fourpi = Real(4.0)*Real(3.1415926535897932);

                Real x = Real(i+0.5)*dx[0];
                Real y = Real(j+0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
                Real z = Real(k+0.5)*dx[2];
#endif
                Real exact = Real(0.0); // quiet compiler warning
                if (err_comp == AMREX_SPACEDIM || err_comp == AMREX_SPACEDIM+1) {  // pressure
                    exact = Real(0.25) * std::cos(fourpi*x) + Real(0.25) * std::cos(fourpi*y);

                } else if (err_comp == 0) { // u
                    exact =  std::sin(twopi*x) * std::cos(twopi*y);
#if (AMREX_SPACEDIM == 3)
                    exact *= std::cos(twopi*z);
#endif

                } else if (err_comp == 1) { // v
                    exact = -std::cos(twopi*x) * std::sin(twopi*y);
#if (AMREX_SPACEDIM == 3)
                    exact *= std::cos(twopi*z);
#endif

#if (AMREX_SPACEDIM == 3)
                } else if (err_comp == 2) { // w
                    exact = 0.;
#endif
                }

                err(i,j,k,soln_comp) -= exact;
            });
        }
    // Decaying Taylor vortex
    } else if (2 == m_probtype) {

        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            constexpr Real     pi = Real(3.1415926535897932);
            constexpr Real  twopi = Real(2.0)*pi;

            constexpr Real u0 = Real(1.0);
            constexpr Real v0 = Real(1.0);

            Real visc_coef = m_mu;

            Real omega = pi * pi * visc_coef;

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                Real x = problo[0] + Real(i+0.5)*dx[0];
                Real y = problo[1] + Real(j+0.5)*dx[1];
                Real exact = Real(0.0); // quiet compiler warning
                if (err_comp == AMREX_SPACEDIM || err_comp == AMREX_SPACEDIM+1) {  // pressure

                    Real t_p = time - Real(0.5)*dt;
                    exact = -Real(0.25) * ( std::cos(twopi*(x-u0*t_p)) + std::cos(twopi*(y-v0*t_p)) ) * std::exp(-Real(4.0)*omega*t_p);

                } else if (err_comp == 0) { // u
                    exact =  u0 - std::cos(pi*(x-u0*time)) * std::sin(pi*(y-v0*time)) * std::exp(-Real(2.0)*omega*time);
#if (AMREX_SPACEDIM == 3)
                    exact =  u0 - std::cos(pi*(x-u0*time)) * std::sin(pi*(y-v0*time)) * std::exp(-Real(2.0)*omega*time);
#endif

                } else if (err_comp == 1) { // v
                    exact = v0 + std::sin(pi*(x-u0*time)) * std::cos(pi*(y-v0*time)) * std::exp(-Real(2.0)*omega*time);
#if (AMREX_SPACEDIM == 3)
                    exact = v0 + std::sin(pi*(x-u0*time)) * std::cos(pi*(y-v0*time)) * std::exp(-Real(2.0)*omega*time);
#endif

#if (AMREX_SPACEDIM == 3)
                } else if (err_comp == 2) { // w
                    exact = Real(0.0);
#endif
                }

                err(i,j,k,soln_comp) -= exact;
            });
        }

    // Burggraf problem
    } else if (16 == m_probtype) {
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real x = Real(i+0.5)*dx[0];
                Real y = Real(j+0.5)*dx[1];

                Real exact = Real(0.0); // quiet compiler warning

                if (err_comp == AMREX_SPACEDIM || err_comp == AMREX_SPACEDIM+1) {  // pressure
                    exact = 0.0;

                } else if (err_comp == 0) { // u
                    exact =  Real(8) * (x*x*x*x - Real(2) * x*x*x + x*x) * (Real(4)*y*y*y - Real(2)*y);

                } else if (err_comp == 1) { // v
                    exact = -Real(8) * (Real(4)*x*x*x - Real(6) * x*x + Real(2)*x) * (y*y*y*y - y*y);
                }

                err(i,j,k,soln_comp) -= exact;
            });
        }
    // Ballistic potential flow (prob_divu_ballistic.H): velocity, pressure, density and
    // tracers.  The exact pressure has zero mean, as error_p assumes, and is taken at
    // t - dt/2, since incflo's pressure is p^(n+1/2).
    } else if (603 == m_probtype) {
        if (err_comp == AMREX_SPACEDIM+1) {
            amrex::Abort("No exact MAC pressure implemented for incflo.probtype = 603");
        }
        if (err_comp == AMREX_SPACEDIM+2 && m_constant_density) {
            amrex::Abort("error_density for incflo.probtype = 603 needs incflo.constant_density = 0");
        }
        if (is_temperature) {
            amrex::Abort("No exact temperature implemented for incflo.probtype = 603");
        }
        auto const& probhi = lev_geom.ProbHiArray();
        GpuArray<Real,AMREX_SPACEDIM> const U{AMREX_D_DECL(m_ic_u, m_ic_v, m_ic_w)};
        GpuArray<int,AMREX_SPACEDIM> wall;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { wall[d] = lev_geom.isPeriodic(d) ? 0 : 1; }
        // p = (4/3) mu S with the tensor viscous solve, mu S with the per-component one
        Real const pfac = (use_tensor_solve ? Real(4.0)/Real(3.0) : Real(1.0)) * m_mu;
        Real const ro_0 = m_ro_0;
        // With a constant density, a tracer in conservative form is diluted by the expansion
        bool const dilute = m_constant_density && itrac >= 0 && m_iconserv_tracer[itrac];
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                GpuArray<Real,AMREX_SPACEDIM> x;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
                }
                Real const t_exact = (err_comp == AMREX_SPACEDIM) ? time - Real(0.5)*dt : time;
                IncfloDivuBallistic r;
                incflo_divu_ballistic(x, t_exact, U, problo, probhi, wall, r);
                Real exact;
                if (err_comp < AMREX_SPACEDIM) {
                    exact = r.vel[err_comp];
                } else if (err_comp == AMREX_SPACEDIM) {
                    exact = pfac * r.S;
                } else if (err_comp == AMREX_SPACEDIM+2) {
                    exact = ro_0 * r.rho0 / r.detF;
                } else {
                    exact = dilute ? r.s0 / r.detF : r.s0;
                }
                err(i,j,k,soln_comp) -= exact;
            });
        }

    // Steady manufactured solution with div(u) = S (prob_divu_mms.H): velocity and
    // pressure.  The exact pressure has zero mean, as error_p assumes.
    } else if (604 == m_probtype) {
        if (err_comp == AMREX_SPACEDIM+1) {
            amrex::Abort("No exact MAC pressure implemented for incflo.probtype = 604");
        }
        auto const& probhi = lev_geom.ProbHiArray();
        Real const mu = m_mu;
        Real const ro_0 = m_ro_0;
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                GpuArray<Real,AMREX_SPACEDIM> x;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
                }
                IncfloDivuMMS r;
                incflo_divu_mms(x, problo, probhi, mu, ro_0, r);
                err(i,j,k,soln_comp) -= (err_comp < AMREX_SPACEDIM) ? r.vel[err_comp] : r.p;
            });
        }
    // Warm layer carried through a channel, with S = kappa (T - T_ref)
    // (prob_divu_temperature.H): velocity, pressure, density, tracers and temperature.  The
    // pressure is 0 at the outflow, and is taken at t - dt/2, since incflo's pressure is
    // p^(n+1/2).
    } else if (606 == m_probtype) {
        if (err_comp == AMREX_SPACEDIM+1) {
            amrex::Abort("No exact MAC pressure implemented for incflo.probtype = 606");
        }
        Orientation const xlo(0, Orientation::low);
        Real const U = m_bc_velocity[xlo][0];
        Real const rho0 = m_bc_density[xlo];
        Real const T_ref = m_bc_temperature[xlo];
        Real const xhi = lev_geom.ProbHi(0);
        // p = (4/3) mu S with the tensor viscous solve, mu S with the per-component one
        Real const pfac = (use_tensor_solve ? Real(4.0)/Real(3.0) : Real(1.0)) * m_mu;
        Real const t_exact = (err_comp == AMREX_SPACEDIM) ? time - Real(0.5)*dt : time;
        auto const q = incflo_divu_temperature_parm(problo[0], xhi);
        auto const tot = incflo_divu_temperature_int(q.amax, t_exact, q);
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
                IncfloDivuTemperature r;
                incflo_divu_temperature(x, t_exact, U, rho0, pfac, xhi, q, tot, r);
                Real exact;
                if (err_comp == 0) {
                    exact = r.u;
                } else if (err_comp < AMREX_SPACEDIM) {
                    exact = Real(0.0);
                } else if (err_comp == AMREX_SPACEDIM) {
                    exact = r.p;
                } else if (err_comp == AMREX_SPACEDIM+2) {
                    exact = r.rho;
                } else if (is_temperature) {
                    exact = T_ref + r.theta;
                } else {
                    exact = r.theta / q.dT;
                }
                err(i,j,k,soln_comp) -= exact;
            });
        }
    } else {
         amrex::Abort("No exact solution implemented for this incflo.probtype");
    }
}
