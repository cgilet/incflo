#include <AMReX_Box.H>

#include <incflo.H>
#include <incflo_derive_K.H>
#include <prob_divu_channel.H>
#include <prob_divu_mms.H>

using namespace amrex;

void incflo::DiffFromExact (int /*lev*/, Geometry& lev_geom, Real time, Real dt,
                            MultiFab& error, int soln_comp, int err_comp) const
{
    auto const& dx = lev_geom.CellSizeArray();
    auto const& problo = lev_geom.ProbLoArray();

    // err_comp: 0 to AMREX_SPACEDIM-1 velocity, AMREX_SPACEDIM pressure, AMREX_SPACEDIM+1
    // MAC pressure, AMREX_SPACEDIM+2 density, AMREX_SPACEDIM+3 the first tracer
    if (err_comp >= AMREX_SPACEDIM+2 && m_probtype != 602 && m_probtype != 603) {
        amrex::Abort("error_density and error_tracer are only implemented for incflo.probtype = 602 and 603");
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
    // div(u) = S with S = f(t) prod_d cos(2 pi x_d / L_d) in a periodic box: the
    // potential flow u = f(t) grad(phi0), lap(phi0) = prod_d cos(2 pi x_d / L_d).
    // Only the velocity is checked.
    } else if (600 == m_probtype || 601 == m_probtype) {
        constexpr Real twopi = Real(2.0) * Real(3.1415926535897932);
        Real const f = (600 == m_probtype) ? Real(1.0) : std::sin(twopi * time);
        auto const& probhi = lev_geom.ProbHiArray();
        GpuArray<Real, AMREX_SPACEDIM> kw;
        Real ksq = Real(0.0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            kw[d] = twopi / (probhi[d] - problo[d]);
            ksq += kw[d] * kw[d];
        }
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                Real exact = Real(0.0);
                if (err_comp < AMREX_SPACEDIM) {
                    exact = f * kw[err_comp] / ksq;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        Real const x = (Real(iv[d]) + Real(0.5)) * dx[d];
                        exact *= (d == err_comp) ? std::sin(kw[d]*x) : std::cos(kw[d]*x);
                    }
                }
                err(i,j,k,soln_comp) -= exact;
            });
        }

    // div(u) = S(t) in a channel with outflow at both ends in x: u = S(t) (x - x_center),
    // with S = 1 (602) or S = 1 + sin(2 pi t)/2 (603).  The density and the first tracer
    // are uniform for 602 and follow prob_divu_channel.H for 603.
    } else if (602 == m_probtype || 603 == m_probtype) {
        if (err_comp == AMREX_SPACEDIM || err_comp == AMREX_SPACEDIM+1) {
            amrex::Abort("No exact pressure implemented for incflo.probtype = 602 or 603");
        }
        if (err_comp >= AMREX_SPACEDIM+2 && m_constant_density) {
            amrex::Abort("error_density and error_tracer for incflo.probtype = 602 and 603 need incflo.constant_density = 0");
        }
        if (err_comp == AMREX_SPACEDIM+3 && m_ntrac < 1) {
            amrex::Abort("error_tracer needs a tracer");
        }
        auto const& probhi = lev_geom.ProbHiArray();
        Real const xc = Real(0.5) * (problo[0] + probhi[0]);
        Real const L = probhi[0] - problo[0];
        bool const varying = (603 == m_probtype);
        Real const S = varying ? incflo_divu_channel_S(time) : Real(1.0);
        Real const ro_0 = m_ro_0;
        Real const ic_t = (m_ntrac > 0) ? m_ic_t[0] : Real(0.0);
        for(MFIter mfi(error, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();

            // When we enter this routine, this holds the computed solution
            Array4<Real> const& err = error.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const xr = problo[0] + Real(i+0.5)*dx[0] - xc;
                Real exact = Real(0.0);
                if (err_comp == 0) {
                    exact = S * xr;
                } else if (err_comp == AMREX_SPACEDIM+2) {
                    exact = varying ? incflo_divu_channel_density(xr, L, time, ro_0)
                                    : ro_0 * std::exp(-time);
                } else if (err_comp == AMREX_SPACEDIM+3) {
                    exact = varying ? incflo_divu_channel_tracer(xr, L, time) : ic_t;
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
    } else {
         amrex::Abort("No exact solution implemented for this incflo.probtype");
    }
}
