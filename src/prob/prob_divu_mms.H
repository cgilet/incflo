#ifndef INCFLO_PROB_DIVU_MMS_H_
#define INCFLO_PROB_DIVU_MMS_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

// Manufactured solution for incflo.probtype = 604: a steady flow with div(u) = S != 0
// in a closed box, adapted from the 3D test of Bouloumou, Serre, Bontoux and Froehlich,
// Computers and Fluids 66 (2012) 107-120, eqs. (25)-(28) with beta = 0, at constant
// density.  Every velocity component vanishes on the walls and dp/dn = 0 there.
//
// 3D: periodic in x, walls in y and z, with xt = 2 pi (x - x_lo)/L_x in [0, 2 pi] and
// yt, zt = 2 (y - y_center)/L_y, 2 (z - z_center)/L_z in [-1, 1]:
//   u = 1/2 sin^2(xt) sin(2 pi yt) sin(2 pi zt)
//   v = 1/4 sin(2 xt) sin^2(pi yt) sin(2 pi zt)
//   w = -1/2 sin(2 xt) sin(pi yt) sin^2(pi zt)
//   p = cos(xt) cos(pi yt) cos(pi zt)
// 2D: the paper's (v, w, p) in its (y, z) plane without the x factors, walls on all
// sides, xt and yt in [-1, 1]:
//   u = 1/4 sin^2(pi xt) sin(2 pi yt)
//   v = -1/2 sin(pi xt) sin^2(pi yt)
//   p = cos(pi xt) cos(pi yt)
//
// S = div(u), and the velocity forcing (per unit mass) that makes this a steady
// solution of incflo's momentum equation with the tensor viscous stress
// tau = mu (grad u + grad u^T) - 2/3 mu div(u) I is
//   f = u . grad(u) + grad(p)/rho - (mu/rho) (lap(u) + grad(S)/3).
//
// Each field is amp * prod_d F_d(xt_d), with F_d one of sin(a xi), sin^2(a xi) and
// cos(a xi), so all derivatives are products of 1D factors.

struct IncfloDivuMMS
{
    amrex::Real vel[AMREX_SPACEDIM];
    amrex::Real p;
    amrex::Real gradp[AMREX_SPACEDIM];
    amrex::Real S;
    amrex::Real force[AMREX_SPACEDIM];
};

// type 0: sin(a xi), 1: sin^2(a xi), 2: cos(a xi)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void incflo_divu_mms_factor (int type, amrex::Real a, amrex::Real xi,
                             amrex::Real& f, amrex::Real& df, amrex::Real& d2f) noexcept
{
    if (type == 0) {
        f = std::sin(a*xi);
        df = a * std::cos(a*xi);
        d2f = -a * a * f;
    } else if (type == 1) {
        amrex::Real const s = std::sin(a*xi);
        f = s * s;
        df = a * std::sin(amrex::Real(2.0)*a*xi);
        d2f = amrex::Real(2.0) * a * a * std::cos(amrex::Real(2.0)*a*xi);
    } else {
        f = std::cos(a*xi);
        df = -a * std::sin(a*xi);
        d2f = -a * a * f;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void incflo_divu_mms (amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& x,
                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& probhi,
                      amrex::Real mu, amrex::Real rho, IncfloDivuMMS& r) noexcept
{
    constexpr int nf = AMREX_SPACEDIM + 1;   // the velocity components, then p
    constexpr amrex::Real pi = amrex::Real(3.1415926535897932);

    // Coordinate xt_d = (x_d - c_d) k_d
    amrex::Real c[AMREX_SPACEDIM], k[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        c[d] = amrex::Real(0.5) * (problo[d] + probhi[d]);
        k[d] = amrex::Real(2.0) / (probhi[d] - problo[d]);
    }
#if (AMREX_SPACEDIM == 3)
    c[0] = problo[0];
    k[0] = amrex::Real(2.0) * pi / (probhi[0] - problo[0]);

    constexpr amrex::Real amp[nf] = {0.5, 0.25, -0.5, 1.0};
    constexpr int type[nf][3] = {{1,0,0}, {0,1,0}, {0,0,1}, {2,2,2}};
    amrex::Real const a[nf][3] = {{1, 2*pi, 2*pi}, {2, pi, 2*pi}, {2, pi, pi}, {1, pi, pi}};
#else
    constexpr amrex::Real amp[nf] = {0.25, -0.5, 1.0};
    constexpr int type[nf][2] = {{1,0}, {0,1}, {2,2}};
    amrex::Real const a[nf][2] = {{pi, 2*pi}, {pi, pi}, {pi, pi}};
#endif

    amrex::Real f[nf][AMREX_SPACEDIM], df[nf][AMREX_SPACEDIM], d2f[nf][AMREX_SPACEDIM];
    for (int n = 0; n < nf; ++n) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            incflo_divu_mms_factor(type[n][d], a[n][d], (x[d] - c[d]) * k[d],
                                   f[n][d], df[n][d], d2f[n][d]);
        }
    }

    // Value, first and second derivatives of field n in physical coordinates
    auto value = [&] (int n) {
        amrex::Real v = amp[n];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { v *= f[n][d]; }
        return v;
    };
    auto deriv = [&] (int n, int d1) {
        amrex::Real v = amp[n] * k[d1];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { v *= (d == d1) ? df[n][d] : f[n][d]; }
        return v;
    };
    auto deriv2 = [&] (int n, int d1, int d2) {
        amrex::Real v = amp[n] * k[d1] * k[d2];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (d == d1 && d == d2) {
                v *= d2f[n][d];
            } else if (d == d1 || d == d2) {
                v *= df[n][d];
            } else {
                v *= f[n][d];
            }
        }
        return v;
    };

    int const ip = AMREX_SPACEDIM;
    r.p = value(ip);
    r.S = amrex::Real(0.0);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        r.vel[i] = value(i);
        r.gradp[i] = deriv(ip, i);
        r.S += deriv(i, i);
    }
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        amrex::Real conv = amrex::Real(0.0);
        amrex::Real lap = amrex::Real(0.0);
        amrex::Real dSdxi = amrex::Real(0.0);
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            conv += r.vel[j] * deriv(i, j);
            lap += deriv2(i, j, j);
            dSdxi += deriv2(j, i, j);
        }
        r.force[i] = conv + r.gradp[i] / rho
            - (mu / rho) * (lap + dSdxi / amrex::Real(3.0));
    }
}

#endif
