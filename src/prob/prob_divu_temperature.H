#ifndef INCFLO_PROB_DIVU_TEMPERATURE_H_
#define INCFLO_PROB_DIVU_TEMPERATURE_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

// Exact solution for incflo.probtype = 606: a warm layer carried through a channel along x,
// with an S that depends on the state,
//   S = kappa (T - T_ref).
// Fluid enters at xlo (mass inflow) with the velocity U, density rho0 and temperature T_ref of
// that boundary, and leaves at xhi (pressure outflow, p = 0); the other directions are
// periodic.  With no conduction (incflo.mu_T = 0) and no heat source DT/Dt = 0, so each particle
// keeps the temperature T_ref + theta0(a), where a is its position at t = 0, and expands at the
// constant rate kappa theta0(a).  The flow stays 1D, and with J = dx/da = exp(kappa theta0(a) t),
//   x   = U t + a + E(a,t),   E = int^a (J - 1) da'
//   u   = U + dE/dt,          so du/dx = kappa theta0(a) = S
//   rho = rho0 / J,           which solves the continuity equation
//   T   = T_ref + theta0(a),  and both tracers s = theta0(a) / dT
// The momentum equation, rho Du/Dt = -dp/dx + d/dx((4/3) mu S) with Du/Dt = d2E/dt2, gives with
// p = 0 and S = 0 at xhi
//   p   = (4/3) mu S + rho0 int_a^aL d2E/dt2 da'
// where aL is the label of the particle at xhi.  With the per-component viscous solve
// (incflo.use_tensor_solve = 0) the first term is mu S.
//
// theta0(a) = dT exp(-((a - a0)/sigma)^2) with a0 = xlo + L/4 and sigma = L/20, L the length of
// the domain in x, so it is below exp(-25) dT outside [amin, amax] = [xlo, xlo + L/2], where the
// integrals over a stop.  The run must end before the layer reaches xhi; incflo checks this at
// startup.  The integrals are computed by Gauss-Legendre quadrature, and the label of a point
// by Newton's method.

struct IncfloDivuTemperatureParm
{
    amrex::Real dT;     // peak of theta0
    amrex::Real kappa;  // S = kappa (T - T_ref)
    amrex::Real a0;     // centre of the layer at t = 0
    amrex::Real sigma;  // width of the layer
    amrex::Real amin;   // theta0 is negligible outside [amin, amax]
    amrex::Real amax;
};

// The integrals from amin to a of J - 1, kappa theta0 J, (kappa theta0)^2 J and
// a (kappa theta0)^2 J: E, dE/dt, d2E/dt2 and the first moment of d(d2E/dt2)/da
struct IncfloDivuTemperatureInt
{
    amrex::Real E;
    amrex::Real Et;
    amrex::Real Ett;
    amrex::Real M;
};

struct IncfloDivuTemperature
{
    amrex::Real u;
    amrex::Real rho;
    amrex::Real theta;  // T - T_ref
    amrex::Real p;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
IncfloDivuTemperatureParm incflo_divu_temperature_parm (amrex::Real xlo, amrex::Real xhi) noexcept
{
    amrex::Real const L = xhi - xlo;
    IncfloDivuTemperatureParm q;
    q.dT = amrex::Real(1.0);
    q.kappa = amrex::Real(2.0);
    q.a0 = xlo + amrex::Real(0.25) * L;
    q.sigma = amrex::Real(0.05) * L;
    q.amin = q.a0 - amrex::Real(5.0) * q.sigma;
    q.amax = q.a0 + amrex::Real(5.0) * q.sigma;
    return q;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_temperature_theta0 (amrex::Real a, IncfloDivuTemperatureParm const& q) noexcept
{
    amrex::Real const r = (a - q.a0) / q.sigma;
    return q.dT * std::exp(-r*r);
}

// Five-point Gauss-Legendre quadrature on panels of width sigma/4 at most
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
IncfloDivuTemperatureInt incflo_divu_temperature_int (amrex::Real a, amrex::Real t,
                                                      IncfloDivuTemperatureParm const& q) noexcept
{
    constexpr amrex::Real xg[5] = {amrex::Real(-0.9061798459386640), amrex::Real(-0.5384693101056831),
                                   amrex::Real(0.0),
                                   amrex::Real( 0.5384693101056831), amrex::Real( 0.9061798459386640)};
    constexpr amrex::Real wg[5] = {amrex::Real(0.2369268850561891), amrex::Real(0.4786286704993665),
                                   amrex::Real(0.5688888888888889),
                                   amrex::Real(0.4786286704993665), amrex::Real(0.2369268850561891)};
    IncfloDivuTemperatureInt r{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
    amrex::Real const b = amrex::min(amrex::max(a, q.amin), q.amax);
    if (b <= q.amin) { return r; }
    int const np = static_cast<int>(std::ceil((b - q.amin) / (amrex::Real(0.25) * q.sigma)));
    amrex::Real const h = (b - q.amin) / amrex::Real(np);
    for (int ip = 0; ip < np; ++ip) {
        amrex::Real const c = q.amin + (amrex::Real(ip) + amrex::Real(0.5)) * h;
        for (int ig = 0; ig < 5; ++ig) {
            amrex::Real const ap = c + amrex::Real(0.5) * h * xg[ig];
            amrex::Real const ks = q.kappa * incflo_divu_temperature_theta0(ap, q);
            amrex::Real const J = std::exp(ks * t);
            amrex::Real const w = amrex::Real(0.5) * h * wg[ig];
            r.E   += w * std::expm1(ks * t);
            r.Et  += w * ks * J;
            r.Ett += w * ks * ks * J;
            r.M   += w * ap * ks * ks * J;
        }
    }
    return r;
}

// The exact solution at x and t.  tot holds the integrals at amax, at the same t; pfac is
// (4/3) mu, or mu with the per-component viscous solve.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void incflo_divu_temperature (amrex::Real x, amrex::Real t, amrex::Real U, amrex::Real rho0,
                              amrex::Real pfac, amrex::Real xhi,
                              IncfloDivuTemperatureParm const& q,
                              IncfloDivuTemperatureInt const& tot,
                              IncfloDivuTemperature& r) noexcept
{
    // The label a of x solves a + E(a,t) = x - U t, where E = 0 below amin and E = tot.E above
    // amax.  In between use Newton's method, kept inside a bracket by bisection.
    amrex::Real const xs = x - U * t;
    amrex::Real a;
    if (xs <= q.amin) {
        a = xs;
    } else if (xs - tot.E >= q.amax) {
        a = xs - tot.E;
    } else {
        amrex::Real lo = q.amin;
        amrex::Real hi = q.amax;
        a = amrex::min(amrex::max(xs - amrex::Real(0.5) * tot.E, lo), hi);
        for (int it = 0; it < 100; ++it) {
            amrex::Real const f = a + incflo_divu_temperature_int(a, t, q).E - xs;
            if (f > amrex::Real(0.0)) { hi = a; } else { lo = a; }
            amrex::Real const J = std::exp(q.kappa * incflo_divu_temperature_theta0(a, q) * t);
            amrex::Real an = a - f / J;
            if (!(an > lo && an < hi)) { an = amrex::Real(0.5) * (lo + hi); }
            bool const done = std::abs(an - a) <= amrex::Real(4.0) * (q.amax - q.amin)
                * std::numeric_limits<amrex::Real>::epsilon();
            a = an;
            if (done) { break; }
        }
    }

    auto const I = incflo_divu_temperature_int(a, t, q);
    amrex::Real const theta = incflo_divu_temperature_theta0(a, q);
    r.theta = theta;
    r.u = U + I.Et;
    r.rho = rho0 * std::exp(-q.kappa * theta * t);

    // int_a^aL d2E/dt2 da', with d2E/dt2 = 0 below amin and tot.Ett above amax
    amrex::Real const aL = xhi - U * t - tot.E;
    amrex::Real const Q = (a >= q.amax) ? (aL - a) * tot.Ett
                                        : aL * tot.Ett - a * I.Ett - (tot.M - I.M);
    r.p = pfac * q.kappa * theta + rho0 * Q;
}

#endif
