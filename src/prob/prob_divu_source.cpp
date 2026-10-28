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

    if (603 == m_probtype)
    {
        // The ballistic potential flow (prob_divu_ballistic.H)
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
                S(i,j,k) = r.S;
            });
        }
    }
    else if (604 == m_probtype)
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
    else if (605 == m_probtype)
    {
        // A pulsing, localized source in a channel along x, like the heat release in a
        // low Mach number flow:
        //   S = A (1 + sin(2 pi t/T)/2) exp(-((x - x0)/sigma)^2) prod_{d>0} sin^2(pi y_d/L_d)
        // with y_d measured from the low end of the domain, x0 a quarter of the way along the
        // channel and sigma = L_x/20, so that S is negligible (exp(-25) or less) at the
        // inflow and outflow ends. The sin^2 factors are smooth across periodic boundaries
        // and zero on walls.
        // S >= 0: the fluid expands as it passes.
        constexpr Real pi = Real(3.1415926535897932);
        constexpr Real A = Real(2.0);
        constexpr Real T = Real(0.2);
        Real const Lx = probhi[0] - problo[0];
        Real const x0 = problo[0] + Real(0.25) * Lx;
        Real const sigma = Real(0.05) * Lx;
        Real const f = A * (Real(1.0) + Real(0.5) * std::sin(Real(2.0) * pi * time / T));
        for (MFIter mfi(divu_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& S = divu_source.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                Real const xr = (problo[0] + (Real(i) + Real(0.5)) * dx[0] - x0) / sigma;
                Real s = f * std::exp(-xr * xr);
                for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                    Real const y = (Real(iv[d]) + Real(0.5)) * dx[d];
                    Real const sy = std::sin(pi * y / (probhi[d] - problo[d]));
                    s *= sy * sy;
                }
                S(i,j,k) = s;
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

    // A constant added to S, for testing (incflo.divu_source_offset).  In a closed domain the
    // projections remove it, so the exact solutions of probtypes 603 and 604 still hold, and
    // the errors must not change.
    Real offset = Real(0.0);
    ParmParse pp("incflo");
    pp.query("divu_source_offset", offset);
    if (offset != Real(0.0)) {
        divu_source.plus(offset, 0, 1, 0);
    }
}
