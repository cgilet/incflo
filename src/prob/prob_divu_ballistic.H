#ifndef INCFLO_PROB_DIVU_BALLISTIC_H_
#define INCFLO_PROB_DIVU_BALLISTIC_H_

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

// Exact solution for incflo.probtype = 603: a ballistic potential flow with div(u) = S.
// Every particle moves in a straight line at constant velocity,
//   x = a + t (U + grad phi0(a)),
// where a is the position of the particle at t = 0, U = (incflo.ic_u, ic_v, ic_w) is uniform
// and phi0 is a smooth potential.  With H = grad grad phi0 at a and F = I + t H = dx/da,
//   u   = U + grad phi0(a)       so Du/Dt = 0, and the flow stays a potential flow
//   S   = div(u) = tr(H F^-1)    which varies in space and time
//   rho = rho0(a) / det(F)       which solves the continuity equation, with no source
//   s   = s0(a)                  which solves both tracer equations when rho is evolved
//   p   = (4/3) mu S             since div(tau) = (4/3) mu grad(S) for a potential flow, the
//                                pressure gradient balances the viscous force
// so no forcing is needed.  With the per-component viscous solve (incflo.use_tensor_solve = 0)
// div(tau) = mu lap(u) = mu grad(S), and p = mu S.  With incflo.constant_density = 1, u, p
// and a tracer in convective form are the same, and a tracer in conservative form is
// s0(a) / det(F).  The solution holds while F is positive definite, until the paths cross.
//
// The map and the density are those of the Zel'dovich approximation for self-gravitating
// dust (Zel'dovich, Astronomy and Astrophysics 5 (1970) 84-89).  There gravity accelerates
// the particles, so it is approximate; here no net force acts on them, so it is exact.
//
// phi0 = A W g, with xi = (a - prob_lo) / L in [0, 1]^d:
//   g = 1 + sum_m c_m sin(2 pi k_m . xi + theta_m), oblique modes that are periodic,
//   W = the product over the non-periodic directions d of sin^2(pi xi_d),
// so that grad(phi0) vanishes on the faces of the non-periodic directions, which are no-slip
// walls (and then U = 0).  The amplitude is for a unit domain: the paths first cross at
// t = 1.4 in 2D and t = 1.1 in 3D.  The label a of a point x is found by Newton's method.

struct IncfloDivuBallistic
{
    amrex::Real vel[AMREX_SPACEDIM];
    amrex::Real S;     // div(u)
    amrex::Real detF;  // det(dx/da), the ratio of the current to the initial volume
    amrex::Real rho0;  // rho0(a) / ro_0
    amrex::Real s0;    // s0(a)
};

// grad(phi0) and H = grad grad(phi0) at the label a
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void incflo_divu_ballistic_phi (amrex::Real const a[AMREX_SPACEDIM],
                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& probhi,
                                amrex::GpuArray<int,AMREX_SPACEDIM> const& wall,
                                amrex::Real grad[AMREX_SPACEDIM],
                                amrex::Real H[AMREX_SPACEDIM][AMREX_SPACEDIM])
{
    constexpr amrex::Real pi = amrex::Real(3.1415926535897932);
    constexpr amrex::Real amp = amrex::Real(0.005);
#if (AMREX_SPACEDIM == 2)
    constexpr int nm = 3;
    constexpr int k[nm][2] = {{1, 1}, {1, -2}, {2, 1}};
    amrex::Real const c[nm] = {amrex::Real(1.0), amrex::Real(0.5), amrex::Real(0.3)};
    amrex::Real const theta[nm] = {amrex::Real(0.0), amrex::Real(0.3), amrex::Real(1.1)};
#else
    constexpr int nm = 4;
    constexpr int k[nm][3] = {{1, 1, 1}, {1, -2, 1}, {2, 1, -1}, {0, 1, -2}};
    amrex::Real const c[nm] = {amrex::Real(1.0), amrex::Real(0.5), amrex::Real(0.3),
                               amrex::Real(0.4)};
    amrex::Real const theta[nm] = {amrex::Real(0.0), amrex::Real(0.3), amrex::Real(1.1),
                                   amrex::Real(2.0)};
#endif
    constexpr int dim = AMREX_SPACEDIM;

    // W and its derivatives, from w_d = sin^2(pi xi_d) (1 in the periodic directions)
    amrex::Real w[dim], w1[dim], w2[dim];
    for (int d = 0; d < dim; ++d) {
        amrex::Real const L = probhi[d] - problo[d];
        amrex::Real const xi = (a[d] - problo[d]) / L;
        if (wall[d]) {
            amrex::Real const s = std::sin(pi*xi);
            w[d] = s*s;
            w1[d] = pi / L * std::sin(amrex::Real(2.0)*pi*xi);
            w2[d] = amrex::Real(2.0) * (pi/L) * (pi/L) * std::cos(amrex::Real(2.0)*pi*xi);
        } else {
            w[d] = amrex::Real(1.0);
            w1[d] = amrex::Real(0.0);
            w2[d] = amrex::Real(0.0);
        }
    }
    amrex::Real W = amrex::Real(1.0);
    for (int d = 0; d < dim; ++d) { W *= w[d]; }
    amrex::Real gW[dim], HW[dim][dim];
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            amrex::Real v = (i == j) ? w2[i] : w1[i]*w1[j];
            for (int d = 0; d < dim; ++d) {
                if (d != i && d != j) { v *= w[d]; }
            }
            HW[i][j] = v;
        }
        amrex::Real v = w1[i];
        for (int d = 0; d < dim; ++d) {
            if (d != i) { v *= w[d]; }
        }
        gW[i] = v;
    }

    // g and its derivatives
    amrex::Real g = amrex::Real(1.0);
    amrex::Real gg[dim], Hg[dim][dim];
    for (int i = 0; i < dim; ++i) {
        gg[i] = amrex::Real(0.0);
        for (int j = 0; j < dim; ++j) { Hg[i][j] = amrex::Real(0.0); }
    }
    for (int m = 0; m < nm; ++m) {
        amrex::Real kap[dim];
        amrex::Real ph = theta[m];
        for (int d = 0; d < dim; ++d) {
            kap[d] = amrex::Real(2.0) * pi * amrex::Real(k[m][d]) / (probhi[d] - problo[d]);
            ph += kap[d] * (a[d] - problo[d]);
        }
        amrex::Real const sn = std::sin(ph);
        amrex::Real const cs = std::cos(ph);
        g += c[m] * sn;
        for (int i = 0; i < dim; ++i) {
            gg[i] += c[m] * cs * kap[i];
            for (int j = 0; j < dim; ++j) { Hg[i][j] -= c[m] * sn * kap[i] * kap[j]; }
        }
    }

    // phi0 = amp W g
    for (int i = 0; i < dim; ++i) {
        grad[i] = amp * (g*gW[i] + W*gg[i]);
        for (int j = 0; j < dim; ++j) {
            H[i][j] = amp * (g*HW[i][j] + gW[i]*gg[j] + gg[i]*gW[j] + W*Hg[i][j]);
        }
    }
}

// The inverse and the determinant of F = I + t H
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_ballistic_inverse (amrex::Real t,
                                           amrex::Real const H[AMREX_SPACEDIM][AMREX_SPACEDIM],
                                           amrex::Real Finv[AMREX_SPACEDIM][AMREX_SPACEDIM])
{
    amrex::Real F[AMREX_SPACEDIM][AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            F[i][j] = (i == j ? amrex::Real(1.0) : amrex::Real(0.0)) + t*H[i][j];
        }
    }
#if (AMREX_SPACEDIM == 2)
    amrex::Real const det = F[0][0]*F[1][1] - F[0][1]*F[1][0];
    Finv[0][0] =  F[1][1]/det;
    Finv[0][1] = -F[0][1]/det;
    Finv[1][0] = -F[1][0]/det;
    Finv[1][1] =  F[0][0]/det;
#else
    amrex::Real C[3][3];  // cofactors
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int const i1 = (i+1)%3, i2 = (i+2)%3, j1 = (j+1)%3, j2 = (j+2)%3;
            C[i][j] = F[i1][j1]*F[i2][j2] - F[i1][j2]*F[i2][j1];
        }
    }
    amrex::Real const det = F[0][0]*C[0][0] + F[0][1]*C[0][1] + F[0][2]*C[0][2];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { Finv[i][j] = C[j][i]/det; }
    }
#endif
    return det;
}

// The smallest leading principal minor of F = I + t H(a): F is positive definite, and the
// particle paths have not crossed by time t, if this is positive at every label a
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_divu_ballistic_minor (amrex::Real const a[AMREX_SPACEDIM], amrex::Real t,
                                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& probhi,
                                         amrex::GpuArray<int,AMREX_SPACEDIM> const& wall)
{
    amrex::Real g[AMREX_SPACEDIM], H[AMREX_SPACEDIM][AMREX_SPACEDIM];
    incflo_divu_ballistic_phi(a, problo, probhi, wall, g, H);
    amrex::Real const F00 = amrex::Real(1.0) + t*H[0][0];
    amrex::Real const F11 = amrex::Real(1.0) + t*H[1][1];
    amrex::Real m = amrex::min(F00, F00*F11 - t*t*H[0][1]*H[1][0]);
#if (AMREX_SPACEDIM == 3)
    amrex::Real Finv[3][3];
    m = amrex::min(m, incflo_divu_ballistic_inverse(t, H, Finv));
#endif
    return m;
}

// The exact solution at the point x and time t
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void incflo_divu_ballistic (amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& x, amrex::Real t,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& U,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& probhi,
                            amrex::GpuArray<int,AMREX_SPACEDIM> const& wall,
                            IncfloDivuBallistic& r)
{
    constexpr int dim = AMREX_SPACEDIM;
    constexpr amrex::Real pi = amrex::Real(3.1415926535897932);
    amrex::Real const tol = amrex::Real(10.0) * std::numeric_limits<amrex::Real>::epsilon();

    // The label a solves a + t grad(phi0)(a) = y = x - t U.  Start from one fixed-point step,
    // then Newton: F is close to the identity, so a few iterations reach rounding error.
    amrex::Real y[dim], a[dim], g[dim], H[dim][dim], Finv[dim][dim];
    for (int d = 0; d < dim; ++d) { y[d] = x[d] - t*U[d]; }
    incflo_divu_ballistic_phi(y, problo, probhi, wall, g, H);
    for (int d = 0; d < dim; ++d) { a[d] = y[d] - t*g[d]; }
    for (int it = 0; it < 30; ++it) {
        incflo_divu_ballistic_phi(a, problo, probhi, wall, g, H);
        amrex::Real res[dim];
        amrex::Real rmax = amrex::Real(0.0);
        for (int d = 0; d < dim; ++d) {
            res[d] = a[d] + t*g[d] - y[d];
            rmax = amrex::max(rmax, std::abs(res[d]) / (probhi[d] - problo[d]));
        }
        if (rmax <= tol) { break; }
        incflo_divu_ballistic_inverse(t, H, Finv);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) { a[i] -= Finv[i][j] * res[j]; }
        }
    }

    incflo_divu_ballistic_phi(a, problo, probhi, wall, g, H);
    r.detF = incflo_divu_ballistic_inverse(t, H, Finv);
    r.S = amrex::Real(0.0);
    for (int i = 0; i < dim; ++i) {
        r.vel[i] = U[i] + g[i];
        for (int j = 0; j < dim; ++j) { r.S += H[i][j] * Finv[j][i]; }
    }

    // rho0 / ro_0 = 1 + 0.3 cos(2 pi kr . xi + 1), s0 = 1 + 0.5 sin(2 pi ks . xi + 0.5)
    constexpr int kr[3] = {2, 1, 1};
    constexpr int ks[3] = {1, -1, 1};
    amrex::Real pr = amrex::Real(1.0);
    amrex::Real ps = amrex::Real(0.5);
    for (int d = 0; d < dim; ++d) {
        amrex::Real const xi = (a[d] - problo[d]) / (probhi[d] - problo[d]);
        pr += amrex::Real(2.0) * pi * amrex::Real(kr[d]) * xi;
        ps += amrex::Real(2.0) * pi * amrex::Real(ks[d]) * xi;
    }
    r.rho0 = amrex::Real(1.0) + amrex::Real(0.3) * std::cos(pr);
    r.s0 = amrex::Real(1.0) + amrex::Real(0.5) * std::sin(ps);
}

#endif
