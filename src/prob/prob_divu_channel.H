#ifndef INCFLO_PROB_DIVU_CHANNEL_H_
#define INCFLO_PROB_DIVU_CHANNEL_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

// Exact solution for incflo.probtype = 603: div(u) = S(t), uniform in space, in a
// channel of length L in x with pressure outflow (p = 0) at both ends and periodic in
// the other directions.  S(t) = 1 + sin(2 pi t)/2 stays positive, so both ends are
// always outflow.  With A(t) the integral of S from 0 to t, and xr = x - x_center,
//
//   u     = S(t) xr, the other components are zero
//   rho   = rho_0(xr e^{-A}) e^{-A},  rho_0(xr) = ro_0 (1 + cos(2 pi xr / L)/2)
//   s     = s_0(xr e^{-A}),           s_0(xr)   = 1 + sin(pi xr / L)/2
//   dp/dx = -rho (dS/dt + S^2) xr
//
// The flow depends on x only, so the pressure balances any force along x; rho_0 is
// symmetric about the center so that p is the same at both ends.  The tracer is the
// same for the conservative and the non-conservative forms, because the density is
// evolved (incflo.constant_density = 0).

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_channel_S (amrex::Real t)
{
    constexpr amrex::Real twopi = amrex::Real(2.0) * amrex::Real(3.1415926535897932);
    return amrex::Real(1.0) + amrex::Real(0.5) * std::sin(twopi * t);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_channel_A (amrex::Real t)
{
    constexpr amrex::Real twopi = amrex::Real(2.0) * amrex::Real(3.1415926535897932);
    return t + (amrex::Real(1.0) - std::cos(twopi * t)) / (amrex::Real(2.0) * twopi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_channel_density (amrex::Real xr, amrex::Real L, amrex::Real t,
                                         amrex::Real ro_0)
{
    constexpr amrex::Real twopi = amrex::Real(2.0) * amrex::Real(3.1415926535897932);
    amrex::Real const e = std::exp(-incflo_divu_channel_A(t));
    return ro_0 * (amrex::Real(1.0) + amrex::Real(0.5) * std::cos(twopi * xr * e / L)) * e;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_channel_tracer (amrex::Real xr, amrex::Real L, amrex::Real t)
{
    constexpr amrex::Real pi = amrex::Real(3.1415926535897932);
    amrex::Real const e = std::exp(-incflo_divu_channel_A(t));
    return amrex::Real(1.0) + amrex::Real(0.5) * std::sin(pi * xr * e / L);
}

#endif
