#include <incflo.H>
#include <prob_divu_channel.H>
#include <prob_divu_mms.H>

using namespace amrex;

//
// Analytic S(x,t) for each probtype: the default for compute_divu_source
//
void incflo::prob_divu_source (int lev, MultiFab& divu_source, Real time) const
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
    else if (603 == m_probtype)
    {
        // Uniform S = 1 + sin(2 pi t)/2 (prob_divu_channel.H)
        divu_source.setVal(incflo_divu_channel_S(time));
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
    else
    {
        amrex::Abort("prob_divu_source: no S defined for this incflo.probtype");
    }
}
