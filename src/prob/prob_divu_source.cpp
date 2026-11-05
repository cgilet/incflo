#include <incflo.H>
#include <prob_divu_ballistic.H>
#include <prob_divu_mms.H>
#include <prob_divu_temperature.H>

using namespace amrex;

//
// S for each probtype, the default for compute_divu_source: an analytic S(x,t), except for
// probtype 606, whose S depends on the temperature.  The state has at least one filled ghost
// cell (temperature is null unless incflo.use_temperature = 1); only the valid region of
// divu_source is filled.
//
void incflo::prob_divu_source (int lev, MultiFab& divu_source, MultiFab const* /*density*/,
                               MultiFab const* /*tracer*/, MultiFab const* temperature,
                               Real time) const
{
    auto const& dx = geom[lev].CellSizeArray();
    auto const& problo = geom[lev].ProbLoArray();
    auto const& probhi = geom[lev].ProbHiArray();

    if (600 == m_probtype || 601 == m_probtype)
    {
        // S = f(t) prod_d cos(2 pi x_d / L_d), with f = 1 (600) or sin(2 pi t) (601).
        // In a periodic box, u = grad(phi) with lap(phi) = S at all times.
        constexpr Real twopi = Real(2.0) * Real(3.1415926535897932);
        Real const f = (600 == m_probtype) ? Real(1.0) : std::sin(twopi * time);
        for (MFIter mfi(divu_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& S = divu_source.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                Real s = f;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    Real const x = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
                    s *= std::cos(twopi * (x - problo[d]) / (probhi[d] - problo[d]));
                }
                S(i,j,k) = s;
            });
        }
    }
    else if (602 == m_probtype)
    {
        // Uniform S = 1
        divu_source.setVal(Real(1.0));
    }
    else if (603 == m_probtype || 604 == m_probtype)
    {
        // The ballistic potential flow (prob_divu_ballistic.H).  604 adds a constant to S.  The
        // domain is closed, so the projections remove it, and so does incflo from the S of the
        // advection scheme (compute_MAC_projected_velocities): the solution, and every error,
        // must be the same as in 603.
        Real const offset = (604 == m_probtype) ? Real(1.54) : Real(0.0);
        GpuArray<Real,AMREX_SPACEDIM> const U{AMREX_D_DECL(m_ic_u, m_ic_v, m_ic_w)};
        GpuArray<int,AMREX_SPACEDIM> wall;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { wall[d] = geom[lev].isPeriodic(d) ? 0 : 1; }
        for (MFIter mfi(divu_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& S = divu_source.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                GpuArray<Real,AMREX_SPACEDIM> x;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
                }
                IncfloDivuBallistic r;
                incflo_divu_ballistic(x, time, U, problo, probhi, wall, r);
                S(i,j,k) = r.S + offset;
            });
        }
    }
    else if (605 == m_probtype)
    {
        // The divergence of the steady manufactured velocity (prob_divu_mms.H)
        Real const mu = m_mu;
        Real const ro_0 = m_ro_0;
        for (MFIter mfi(divu_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& S = divu_source.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                GpuArray<Real,AMREX_SPACEDIM> x;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
                }
                IncfloDivuMMS r;
                incflo_divu_mms(x, problo, probhi, mu, ro_0, r);
                S(i,j,k) = r.S;
            });
        }
    }
    else if (606 == m_probtype)
    {
        // S = kappa (T - T_ref) (prob_divu_temperature.H), computed from the state as the S of
        // a low Mach number model would be
        AMREX_ALWAYS_ASSERT(temperature != nullptr);
        auto const q = incflo_divu_temperature_parm(geom[0].ProbLo(0), geom[0].ProbHi(0));
        Real const kappa = q.kappa;
        Real const T_ref = m_bc_temperature[Orientation(0,Orientation::low)];
        for (MFIter mfi(divu_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& S = divu_source.array(mfi);
            Array4<Real const> const& T = temperature->const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                S(i,j,k) = kappa * (T(i,j,k) - T_ref);
            });
        }
    }
    else
    {
        amrex::Abort("prob_divu_source: no S defined for this incflo.probtype");
    }
}
