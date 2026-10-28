#include <incflo.H>
#include <prob_divu_ballistic.H>
#include <prob_divu_mms.H>
#include <prob_divu_temperature.H>

using namespace amrex;

void incflo::prob_init_fluid (int lev)
{
    auto& ld = *m_leveldata[lev];
    Box const& domain = geom[lev].Domain();
    auto const& dx = geom[lev].CellSizeArray();
    auto const& problo = geom[lev].ProbLoArray();
    auto const& probhi = geom[lev].ProbHiArray();

    if (m_use_cc_proj) {
        ld.p_cc.setVal(0.0);
    } else {
        ld.p_nd.setVal(0.0);
    }

    ld.gp.setVal(0.0);

    ld.density.setVal(m_ro_0);
    ld.density_o.setVal(m_ro_0);

    AMREX_D_TERM(ld.velocity.setVal(m_ic_u, 0, 1);,
                 ld.velocity.setVal(m_ic_v, 1, 1);,
                 ld.velocity.setVal(m_ic_w, 2, 1););

    for (int comp = 0; comp < m_ntrac; comp++) {
        ld.tracer.setVal(m_ic_t[comp], comp, 1);
    }

    if (m_use_temperature) {
        ld.temperature.setVal(m_ic_tem);
    }

    if (603 == m_probtype) {
        check_divu_ballistic(lev);
    }
    if (606 == m_probtype) {
        check_divu_temperature(lev);
    }

    // The forcing of the manufactured solution assumes these
    if (604 == m_probtype) {
        if (!m_use_divu_source || !m_constant_density ||
            m_fluid_model != FluidModel::Newtonian || !use_tensor_solve) {
            amrex::Abort("incflo.probtype = 604 requires incflo.use_divu_source = 1, "
                         "incflo.constant_density = 1, a Newtonian fluid and "
                         "incflo.use_tensor_solve = 1");
        }
    }

    for (MFIter mfi(ld.density); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        const Box& gbx = mfi.fabbox();
        if (0 == m_probtype || 114 == m_probtype )
        { }
        else if (603 == m_probtype)
        {
            init_divu_ballistic(vbx, gbx,
                                ld.velocity.array(mfi),
                                ld.density.array(mfi),
                                ld.tracer.array(mfi),
                                domain, dx, problo, probhi);
        }
        else if (604 == m_probtype)
        {
            init_divu_mms(vbx, gbx,
                          ld.velocity.array(mfi),
                          ld.density.array(mfi),
                          ld.tracer.array(mfi),
                          domain, dx, problo, probhi);
        }
        else if (605 == m_probtype)
        {
            init_divu_plume(vbx, gbx,
                            ld.velocity.array(mfi),
                            ld.density.array(mfi),
                            ld.tracer.array(mfi),
                            domain, dx, problo, probhi);
        }
        else if (606 == m_probtype)
        {
            init_divu_temperature(vbx, gbx,
                                  ld.velocity.array(mfi),
                                  ld.density.array(mfi),
                                  ld.tracer.array(mfi),
                                  ld.temperature.array(mfi),
                                  domain, dx, problo, probhi);
        }
        else if (1 == m_probtype)
        {
            init_taylor_green(vbx, gbx,
                              ld.velocity.array(mfi),
                              ld.density.array(mfi),
                              ld.tracer.array(mfi),
                              domain, dx, problo, probhi);
        }
        else if (2 == m_probtype)
        {
            init_taylor_vortex(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
        else if (3 == m_probtype)
        {
            init_taylor_green3d(vbx, gbx,
                                ld.velocity.array(mfi),
                                ld.density.array(mfi),
                                ld.tracer.array(mfi),
                                domain, dx, problo, probhi);
        }
        else if (4 == m_probtype)
        {
            init_couette(vbx, gbx,
                         ld.velocity.array(mfi),
                         ld.density.array(mfi),
                         ld.tracer.array(mfi),
                         domain, dx, problo, probhi);
        }
        else if (5 == m_probtype)
        {
            init_rayleigh_taylor(vbx, gbx,
                                 ld.velocity.array(mfi),
                                 ld.density.array(mfi),
                                 ld.tracer.array(mfi),
                                 domain, dx, problo, probhi);
        }
        else if (6 == m_probtype)
        {
            init_channel_slant(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
        else if (11 == m_probtype)
        {
            init_tuscan(vbx, gbx,
                        ld.velocity.array(mfi),
                        ld.density.array(mfi),
                        ld.tracer.array(mfi),
                        domain, dx, problo, probhi);
        }
        else if (1100 == m_probtype || 1101 == m_probtype || 1102 == m_probtype)
        {
#if (AMREX_SPACEDIM == 2)
            if (1100 == m_probtype || 1102 == m_probtype) {
                amrex::Abort("prob_init_fluid: probtypes 1100 and 1102 involve the z direction "
                             "and need a 3D build");
            }
#endif
            init_jump(vbx, gbx,
                      ld.velocity.array(mfi),
                      ld.density.array(mfi),
                      ld.tracer.array(mfi),
                      domain, dx, problo, probhi);
        }
        else if (111 == m_probtype || 112 == m_probtype || 113 == m_probtype)
        {
            init_boussinesq_bubble(vbx, gbx,
                                   ld.velocity.array(mfi),
                                   ld.density.array(mfi),
                                   ld.tracer.array(mfi),
                                   domain, dx, problo, probhi);
        }
        else if (12 == m_probtype || 122 == m_probtype)
        {
            init_periodic_tracer(vbx, gbx,
                                 ld.velocity.array(mfi),
                                 ld.density.array(mfi),
                                 ld.tracer.array(mfi),
                                 domain, dx, problo, probhi);
        }
        else if (13 == m_probtype)
        {
            init_flow_in_box(vbx, gbx,
                             ld.velocity.array(mfi),
                             ld.density.array(mfi),
                             ld.tracer.array(mfi),
                             domain, dx, problo, probhi);
        }
        else if (14 == m_probtype)
        {
            init_circ_traceradvect(vbx, gbx,
                                   ld.velocity.array(mfi),
                                   ld.density.array(mfi),
                                   ld.tracer.array(mfi),
                                   domain, dx, problo, probhi);
        }
        else if (15 == m_probtype)
        {
            init_gaussian_traceradvect(vbx, gbx,
                                       ld.velocity.array(mfi),
                                       ld.density.array(mfi),
                                       ld.tracer.array(mfi),
                                       domain, dx, problo, probhi);
        }
        else if (16 == m_probtype)
        {
            init_burggraf(vbx, gbx,
                          ld.velocity.array(mfi),
                          ld.density.array(mfi),
                          ld.tracer.array(mfi),
                          domain, dx, problo, probhi);
        }
        else if (66 == m_probtype)
        {
            init_vortex_in_sphere(vbx, gbx,
                                  ld.velocity.array(mfi),
                                  ld.density.array(mfi),
                                  ld.tracer.array(mfi),
                                  domain, dx, problo, probhi);
        }
        else if (21 == m_probtype || 22 == m_probtype || 23 == m_probtype)
        {
            init_double_shear_layer(vbx, gbx,
                                    ld.velocity.array(mfi),
                                    ld.density.array(mfi),
                                    ld.tracer.array(mfi),
                                    domain, dx, problo, probhi);
        }
        else if (31  == m_probtype || 32  == m_probtype || 33  == m_probtype ||
                 311 == m_probtype || 322 == m_probtype || 333 == m_probtype ||
                 41  == m_probtype || 42 == m_probtype  || 43 == m_probtype)
        {
            init_plane_poiseuille(vbx, gbx,
                                  ld.velocity.array(mfi),
                                  ld.density.array(mfi),
                                  ld.tracer.array(mfi),
                                  domain, dx, problo, probhi);
        }
#if 0
        else if (500 == m_probtype)
        {
            init_heated_ground(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);
        }
#endif
        else if (2000 == m_probtype)
        {
            init_rotating_flow(vbx, gbx,
                               ld.velocity.array(mfi),
                               ld.density.array(mfi),
                               ld.tracer.array(mfi),
                               domain, dx, problo, probhi);

        }
        else
        {
            amrex::Abort("prob_init_fluid: unknown m_probtype");
        };
    }
}

void incflo::init_rotating_flow (Box const& vbx, Box const& /*gbx*/,
                                 Array4<Real> const& vel,
                                 Array4<Real> const& /*density*/,
                                 Array4<Real> const& tracer,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = (Real(i)+Real(0.5))*dx[0] - Real(0.5);
        Real y = (Real(j)+Real(0.5))*dx[1] - Real(0.5);
        Real r = std::sqrt(x*x + y*y);

        Real r0 = Real(0.45);
        Real const_val = Real(0.1);

        // We want flow to be 0 at origin and at cylinder boundary
        //    This will peak at half the radius
        Real magvel = const_val * (r) * (r0-r);

        // clockwise rotation with flow decreasing to 0 at cylinder boundary
        vel(i,j,k,0) = -(y/r) * magvel;
        vel(i,j,k,1) =  (x/r) * magvel;
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0);
#endif
        Real x_t = Real(i+Real(0.5))*dx[0] - Real(0.5);
        Real y_t = Real(j+Real(0.5))*dx[1] - Real(0.3);
#if (AMREX_SPACEDIM == 3)
        Real z_t = Real(k+Real(0.5))*dx[2] - Real(0.5);
        Real r_t = std::sqrt(x_t*x_t + y_t*y_t + z_t*z_t);
#else
        // The velocity field is a purely planar rotation, so in 2D the blob is
        // the mid-plane slice of the 3D one (z_t = 0).
        Real r_t = std::sqrt(x_t*x_t + y_t*y_t);
#endif
        Real width = Real(0.25);

        tracer(i,j,k) = Real(0.5)*(Real(1)-std::tanh((r_t)/width));

        //if (r_t < Real(0.2)) {
        //    tracer(i,j,k) = Real(1);
        //} else {
        //    tracer(i,j,k) = Real(0);
       // }
    });
}

void incflo::init_taylor_green (Box const& vbx, Box const& /*gbx*/,
                                Array4<Real> const& vel,
                                Array4<Real> const& /*density*/,
                                Array4<Real> const& /*tracer*/,
                                Box const& /*domain*/,
                                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = (Real(i)+Real(0.5))*dx[0];
        Real y = (Real(j)+Real(0.5))*dx[1];
        constexpr Real twopi = Real(2.0)*Real(3.1415926535897932);
        vel(i,j,k,0) =  std::sin(twopi*x) * std::cos(twopi*y);
        vel(i,j,k,1) = -std::cos(twopi*x) * std::sin(twopi*y);
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}

void incflo::init_taylor_green3d (Box const& vbx, Box const& /*gbx*/,
                                  Array4<Real> const& vel,
                                  Array4<Real> const& /*density*/,
                                  Array4<Real> const& /*tracer*/,
                                  Box const& /*domain*/,
                                  GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                  GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                  GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
#else
        Real z = 0.0_rt;
#endif
        constexpr Real twopi = Real(2.0)*Real(3.1415926535897932);
        AMREX_D_TERM(vel(i,j,k,0) =  std::sin(twopi*x) * std::cos(twopi*y) * std::cos(twopi*z);,
                     vel(i,j,k,1) = -std::cos(twopi*x) * std::sin(twopi*y) * std::cos(twopi*z);,
                     vel(i,j,k,2) = 0.0;);
    });
}

void incflo::init_taylor_vortex (Box const& vbx, Box const& /*gbx*/,
                                 Array4<Real> const& vel,
                                 Array4<Real> const& /*density*/,
                                 Array4<Real> const& /*tracer*/,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
        constexpr Real pi = Real(3.1415926535897932);
        constexpr Real u0 = Real(1.0);
        constexpr Real v0 = Real(1.0);
        vel(i,j,k,0) =  u0 - std::cos(pi*x) * std::sin(pi*y);
        vel(i,j,k,1) =  v0 + std::sin(pi*x) * std::cos(pi*y);
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}


void incflo::init_vortex_in_sphere (Box const& vbx, Box const& /*gbx*/,
                               Array4<Real> const& vel,
                               Array4<Real> const& /*density*/,
                               Array4<Real> const& /*tracer*/,
                               Box const& /*domain*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               GpuArray<Real, AMREX_SPACEDIM> const& problo,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
        Real deltax = x;
        Real deltay = y;
        Real d_sq = deltax*deltax + deltay*deltay;
        Real r_sq = Real(0.003)*Real(0.003);
        Real u_vort = -Real(0.2)*deltay/r_sq * std::exp(-d_sq/r_sq/Real(2.0));
        Real v_vort =  Real(0.2)*deltax/r_sq * std::exp(-d_sq/r_sq/Real(2.0));
        vel(i,j,k,0) =  u_vort;
        vel(i,j,k,1) =  v_vort;
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0.0);
#endif
    });
}

void incflo::init_flow_in_box (Box const& vbx, Box const& /*gbx*/,
                               Array4<Real> const& vel,
                               Array4<Real> const& /*density*/,
                               Array4<Real> const& tracer,
                               Box const& /*domain*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                               GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParmParse pp("box");

    Vector<Real> boxLo(AMREX_SPACEDIM), boxHi(AMREX_SPACEDIM);
    Real offset = Real(1.0e-15);

    for(int i = 0; i < AMREX_SPACEDIM; i++)
    {
        boxLo[i] = geom[0].ProbLo(i);
        boxHi[i] = geom[0].ProbHi(i);
    }

    pp.queryarr("Lo", boxLo, 0, AMREX_SPACEDIM);
    pp.queryarr("Hi", boxHi, 0, AMREX_SPACEDIM);

#if (AMREX_SPACEDIM == 3)
    int periodic_dir;
    pp.get("periodic_dir", periodic_dir);
#endif

    pp.query("offset", offset);

    Real xlo = boxLo[0] + offset;
    Real xhi = boxHi[0] - offset;

    Real ylo = boxLo[1] + offset;
    Real yhi = boxHi[1] - offset;

#if (AMREX_SPACEDIM == 3)
    Real zlo = boxLo[2] + offset;
    Real zhi = boxHi[2] - offset;
#endif

#if (AMREX_SPACEDIM == 3)
    if (periodic_dir == 0)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5)*dx[1]*(yhi-ylo) + ylo;
            Real z = Real(k+0.5)*dx[2]*(zhi-zlo) + zlo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,1) =  std::sin(pi*y) * std::cos(pi*z);
            vel(i,j,k,2) = -std::cos(pi*y) * std::sin(pi*z);
            vel(i,j,k,0) = Real(1.0);
            if (y < 0.5)
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
    } else if (periodic_dir == 1)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dx[0]*(xhi-xlo) + xlo;
            Real z = Real(k+0.5)*dx[2]*(zhi-zlo) + zlo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,2) =  std::sin(pi*z) * std::cos(pi*x);
            vel(i,j,k,0) = -std::cos(pi*z) * std::sin(pi*x);
            vel(i,j,k,1) = Real(1.0);
            if (z < Real(0.5))
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
    } else if (periodic_dir == 2)
#endif
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dx[0]*(xhi-xlo) + xlo;
            Real y = Real(j+0.5)*dx[1]*(yhi-ylo) + ylo;
            constexpr Real pi = Real(3.1415926535897932);
            vel(i,j,k,0) =  std::sin(pi*x) * std::cos(pi*y);
            vel(i,j,k,1) = -std::cos(pi*x) * std::sin(pi*y);
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = Real(1.0);
#endif
            if (x < Real(0.5))
                tracer(i,j,k) = Real(0.0);
            else
                tracer(i,j,k) = Real(1.0);
        });
#if (AMREX_SPACEDIM == 3)
    } else {
       amrex::Error("flow_in_box assumes a periodic direction");
#endif
    }
}

void incflo::init_circ_traceradvect (Box const& vbx, Box const& /*gbx*/,
                                     Array4<Real> const& vel,
                                     Array4<Real> const& density,
                                     Array4<Real> const& tracer,
                                     Box const& /*domain*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{

#if (AMREX_SPACEDIM == 2)
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(1);
        vel(i,j,k,1) = Real(0.5);

        density(i,j,k) = Real(1);

        Real sum = Real(0);
        for (int jj=0; jj<10; ++jj) {
            Real yy = (j + (jj+Real(0.5))/Real(10)) * dx[1];
            for (int ii=0; ii<10; ++ii) {
                Real xx = (i + (ii+Real(0.5))/Real(10)) * dx[0];

                Real r = std::sqrt( (xx-Real(0.5))*(xx-Real(0.5)) + (yy-Real(0.5))*(yy-Real(0.5)) );

                if (r < Real(0.1)) {
                    sum += Real(1);
                } else if (r == Real(0.1)) {
                    sum += Real(0.5);
                }

            }
        }

        tracer(i,j,k) = sum * Real(0.01);

    });

#elif (AMREX_SPACEDIM == 3)
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(1);
        vel(i,j,k,1) = Real(0.5);
        vel(i,j,k,2) = Real(0.25);

        density(i,j,k) = Real(1);

        Real sum = 0.;
        for (int kk=0; kk<10; ++kk) {
            Real zz = (k + (kk+Real(0.5))/Real(10)) * dx[2];
            for (int jj=0; jj<10; ++jj) {
                Real yy = (j + (jj+Real(0.5))/Real(10)) * dx[1];
                for (int ii=0; ii<10; ++ii) {
                    Real xx = (i + (ii+Real(0.5))/Real(10)) * dx[0];

                    Real r = std::sqrt( (xx-Real(0.5))*(xx-Real(0.5)) + (yy-Real(0.5))*(yy-Real(0.5))
                                       +(zz-Real(0.5))*(zz-Real(0.5)) );

                    if (r < Real(0.1)) {
                        sum += Real(1);
                    } else if (r == Real(0.1)) {
                        sum += Real(0.5);
                    }

                }
            }
        }

        tracer(i,j,k) = sum * Real(0.001);

    });
#endif

}void incflo::init_gaussian_traceradvect (Box const& vbx, Box const& /*gbx*/,
                                          Array4<Real> const& vel,
                                          Array4<Real> const& density,
                                          Array4<Real> const& tracer,
                                          Box const& /*domain*/,
                                          GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                          GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                          GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];

        vel(i,j,k,0) = Real(1);
        vel(i,j,k,1) = Real(1);

        density(i,j,k) = Real(1);

#if (AMREX_SPACEDIM == 2)
        Real r = std::sqrt( (x-Real(0.5))*(x-Real(0.5)) + (y-Real(0.5))*(y-Real(0.5)) );
#elif (AMREX_SPACEDIM == 3)
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
        vel(i,j,k,2) = Real(1);

        Real r = std::sqrt( (x-Real(0.5))*(x-Real(0.5)) + (y-Real(0.5))*(y-Real(0.5)) + (z-Real(0.5))*(z-Real(0.5)) );
#endif

        tracer(i,j,k) = std::exp(-Real(300)*r*r);
    });

}

void incflo::init_couette (Box const& vbx, Box const& /*gbx*/,
                           Array4<Real> const& vel,
                           Array4<Real> const& /*density*/,
                           Array4<Real> const& /*tracer*/,
                           Box const& domain,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                           GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    Real num_cells_y = static_cast<Real>(domain.length(1));
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real y = Real(j+0.5) / num_cells_y;
        AMREX_D_TERM(vel(i,j,k,0) *= (y-Real(0.5));,
                     vel(i,j,k,1) = Real(0.0);,
                     vel(i,j,k,2) = Real(0.0););
    });
}

void incflo::init_channel_slant (Box const& vbx, Box const& /*gbx*/,
                                 Array4<Real> const& /*vel*/,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Box const& domain,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    const auto dhi = amrex::ubound(domain);
    int direction  = -1;

    // Get cylinder information from inputs file.                               *
    ParmParse pp("cylinder");
    pp.get("direction",  direction);

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (density(i,j,k)>0) {

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n)
                tracer(i,j,k,n) = Real(0.0);

            if (direction == 0) {
                if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = Real(3.0);
            } else if (direction == 1) {
                if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = Real(3.0);
#if (AMREX_SPACEDIM == 3)
            } else {
                if (nt > 0 && k <= dhi.z/8)   tracer(i,j,k,0) = Real(1.0);
                if (nt > 1 && k <= dhi.z/2)   tracer(i,j,k,1) = Real(2.0);
                if (nt > 2 && k <= dhi.z*3/4) tracer(i,j,k,2) = Real(3.0);
#endif
            }
        }
    });
}

void incflo::init_rayleigh_taylor (Box const& vbx, Box const& /*gbx*/,
                                   Array4<Real> const& vel,
                                   Array4<Real> const& density,
                                   Array4<Real> const& tracer,
                                   Box const& /*domain*/,
                                   GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                   GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                   GpuArray<Real, AMREX_SPACEDIM> const& probhi)
{
    static constexpr Real pi    = Real(3.1415926535897932);
    static constexpr Real rho_1 = Real(0.5);
    static constexpr Real rho_2 = Real(2.0);
    static constexpr Real tra_1 = Real(0.0);
    static constexpr Real tra_2 = Real(1.0);

    static constexpr Real width = Real(0.005);

    const Real splitx = Real(0.5)*(problo[0] + probhi[0]);
    const Real L_x    = probhi[0] - problo[0];

#if (AMREX_SPACEDIM == 2)
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(0.0);
        vel(i,j,k,1) = Real(0.0);

        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];

        const Real r2d = amrex::min(std::abs(x-splitx), Real(0.5)*L_x);
        const Real pertheight = Real(0.5) - Real(0.01)*std::cos(Real(2.0)*pi*r2d/L_x);

        density(i,j,k) = rho_1 + (Real(0.5)*(rho_2-rho_1))*(Real(1.0)+std::tanh((y-pertheight)/width));
        tracer(i,j,k)  = tra_1 + (Real(0.5)*(tra_2-tra_1))*(Real(1.0)+std::tanh((y-pertheight)/width));
    });

#elif (AMREX_SPACEDIM == 3)

    const Real splity = Real(0.5)*(problo[1] + probhi[1]);
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,0) = Real(0.0);
        vel(i,j,k,1) = Real(0.0);
        vel(i,j,k,2) = Real(0.0);

        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];

        const Real r2d = amrex::min(std::hypot((x-splitx),(y-splity)), Real(0.5)*L_x);
        const Real pertheight = Real(0.5) - Real(0.01)*std::cos(Real(2.0)*pi*r2d/L_x);

        density(i,j,k) = rho_1 + (Real(0.5)*(rho_2-rho_1))*(Real(1.0)+std::tanh((z-pertheight)/width));
        tracer(i,j,k)  = tra_1 + (Real(0.5)*(tra_2-tra_1))*(Real(1.0)+std::tanh((z-pertheight)/width));
    });
#endif
}

void incflo::init_tuscan (Box const& vbx, Box const& /*gbx*/,
                          Array4<Real> const& vel,
                          Array4<Real> const& density,
                          Array4<Real> const& tracer,
                          Box const& domain,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                          GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)

{
    int half_num_cells = domain.length(AMREX_SPACEDIM-1) / 2;
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                     vel(i,j,k,1) = Real(0.0);,
                     vel(i,j,k,2) = Real(0.0););
        density(i,j,k) = Real(1.0);
        // half_num_cells comes from the vertical dimension, which is y in 2D and z in 3D
#if (AMREX_SPACEDIM == 2)
        const int kvert = j;
#else
        const int kvert = k;
#endif
        if (kvert <= half_num_cells) {
            tracer(i,j,k) = Real(0.0);
        } else {
            tracer(i,j,k) = Real(0.01);
        }
    });
}

void incflo::init_jump (Box const& vbx, Box const& /*gbx*/,
                        Array4<Real> const& vel,
                        Array4<Real> const& /*density*/,
                        Array4<Real> const& /*tracer*/,
                        Box const& domain,
                        GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                        GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                        GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/) const
{
    // Only 1101 (jump along y, flipping u) is meaningful in 2D; 1100 and 1102 flip w
    // / index the z direction and are refused for a 2D build in prob_init_fluid.
    int direction = 0;
    if (1101 == m_probtype) {
        direction = 1;
    }
#if (AMREX_SPACEDIM == 3)
    else if (1102 == m_probtype) {
        direction = 2;
    }
#endif

    int half_num_cells = domain.length(direction) / 2;
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
#if (AMREX_SPACEDIM == 3)
        if (direction == 0) {
            if (i <= half_num_cells) {
                vel(i,j,k,2) = -vel(i,j,k,2);
            }
        }
        else
#endif
        if (direction == 1) {
            if (j <= half_num_cells) {
                vel(i,j,k,0) = -vel(i,j,k,0);
            }
        }
#if (AMREX_SPACEDIM == 3)
        else if (direction == 2) {
            if (k <= half_num_cells) {
                vel(i,j,k,1) = -vel(i,j,k,1);
            }
        }
#endif
    });
}

void incflo::init_boussinesq_bubble (Box const& vbx, Box const& /*gbx*/,
                                     Array4<Real> const& vel,
                                     Array4<Real> const& density,
                                     Array4<Real> const& tracer,
                                     Box const& /*domain*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                     GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/) const

{
#if (AMREX_SPACEDIM == 3)
    constexpr Real   m_half = Real(0.50);
    constexpr Real m_fourth = Real(0.25);
#endif

    if (111 == m_probtype)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = 0.0;
            vel(i,j,k,1) = 0.0;
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = 0.0;
#endif
            density(i,j,k) = 1.0;

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
#if (AMREX_SPACEDIM == 2)
            Real r = std::sqrt((x-Real(0.25))*(x-Real(0.25)) + (y-Real(0.5))*(y-Real(0.5)));
#elif  (AMREX_SPACEDIM == 3)
            Real z = Real(k+0.5)*dx[2];
            Real r = std::sqrt((x-m_half)*(x-m_half) + (y-m_fourth)*(y-m_fourth) + (z-m_fourth)*(z-m_fourth));
#endif
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#if (AMREX_SPACEDIM == 3)
    else if (112 == m_probtype) {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = Real(0.0);
            vel(i,j,k,1) = Real(0.0);
            vel(i,j,k,2) = Real(0.0);
            density(i,j,k) = Real(1.0);

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
            Real z = Real(k+0.5)*dx[2];

            Real r = std::sqrt( (x-m_fourth)*(x-m_fourth) + (y-m_half )*(y-m_half)
                               +(z-m_fourth)*(z-m_fourth));

            if(r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    } else if (113 == m_probtype) {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = 0.0;
            vel(i,j,k,1) = 0.0;
            vel(i,j,k,2) = 0.0;
            density(i,j,k) = 1.0;

            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
            Real z = Real(k+0.5)*dx[2];

            Real r = std::sqrt((x-m_fourth)*(x-m_fourth) + (y-m_fourth)*(y-m_fourth) + (z-m_half)*(z-m_half));

            if(r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#endif
}

void incflo::init_periodic_tracer (Box const& vbx, Box const& /*gbx*/,
                                   Array4<Real> const& vel,
                                   Array4<Real> const& density,
                                   Array4<Real> const& tracer,
                                   Box const& /*domain*/,
                                   GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                   GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                   GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    Real L = probhi[0]-problo[0];
    Real C = Real(2.0)*Real(3.1415926535897932) / L;

    if (m_probtype == 12)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            constexpr Real A = Real(1.0);
            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
            Real z = Real(k+0.5)*dx[2];
#else
            Real z = 0.0_rt;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = Real(1.0);,
                         vel(i,j,k,1) = Real(0.1)*(std::sin(C*(x+z) - Real(0.00042)) + Real(1.0)) * std::exp(y);,
                         vel(i,j,k,2) = Real(0.1)*(std::sin(C*(x+y) - Real(0.00042)) + Real(1.0)) * std::exp(z););
            tracer(i,j,k,0) = A *(std::sin(C*(y+z) - Real(0.00042)) + Real(1.0)) * std::exp(x);
        });
    }
    else if (m_probtype == 122)
    {
        Real B = Real(0.1);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            constexpr Real A = Real(1.0);
            Real x = Real(i+0.5)*dx[0];
            Real y = Real(j+0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
            Real z = Real(k+0.5)*dx[2];
#else
            Real z = 0.0_rt;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = Real(1.0);,
                         vel(i,j,k,1) = Real(0.1)*(std::sin(C*(x+z) - Real(0.00042)) + Real(1.0)) * std::exp(y);,
                         vel(i,j,k,2) = Real(0.1)*(std::sin(C*(x+y) - Real(0.00042)) + Real(1.0)) * std::exp(z););
            density(i,j,k)  = A + B*(x + y + z);
            tracer(i,j,k,0) = A *(std::sin(C*(y+z) - Real(0.00042)) + Real(1.0)) * std::exp(x);
            if (tracer.nComp() > 1) {
                tracer(i,j,k,1) = A *(std::sin(C*(y+z) - Real(0.00042)) + Real(1.0)) * std::exp(x);
            }
        });
    } else {
        Abort("Unknown periodic tracer probtype");
    }
}

void incflo::init_double_shear_layer (Box const& vbx, Box const& /*gbx*/,
                                      Array4<Real> const& vel,
                                      Array4<Real> const& /*density*/,
                                      Array4<Real> const& tracer,
                                      Box const& /*domain*/,
                                      GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                      GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                      GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/) const
{
    static constexpr Real twopi = Real(2.0) * Real(3.1415926535897932);
    if (21 == m_probtype)
    {
        constexpr Real m_fourth = Real(0.25);
        constexpr Real m_half   = Real(0.5);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5) * dx[0];
            Real y = Real(j+0.5) * dx[1];
            vel(i,j,k,0) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(y-Real(0.5))));
            vel(i,j,k,1) = Real(0.05)*std::sin(twopi*x);
#if (AMREX_SPACEDIM == 3)
            vel(i,j,k,2) = Real(0.0);
#endif
            Real r = std::sqrt((x-m_half)*(x-m_half) + (y-m_fourth)*(y-m_fourth));
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#if (AMREX_SPACEDIM == 3)
    else if (22 == m_probtype)
    {
        constexpr Real m_half = Real(0.5);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5) * dx[1];
            Real z = Real(k+0.5) * dx[2];
            vel(i,j,k,1) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(z-m_half)));
            vel(i,j,k,2) = Real(0.05)*std::sin(twopi*y);
            vel(i,j,k,0) = Real(0.0);

            Real r = std::sqrt((y-m_half)*(y-m_half) + (z-m_half)*(z-m_half));
            if (r < Real(0.1))
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
    else if (23 == m_probtype)
    {
        constexpr Real m_half = Real(0.5);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5) * dx[0];
            Real z = Real(k+0.5) * dx[2];
            vel(i,j,k,2) = std::tanh(Real(30.0)*(Real(0.25)-amrex::Math::abs(x-m_half)));
            vel(i,j,k,0) = Real(0.05)*std::sin(twopi*z);
            vel(i,j,k,1) = Real(0.0);

            Real r = std::sqrt((x-m_half)*(x-m_half) + (z-m_half)*(z-m_half));
            if (r < .1)
                tracer(i,j,k,0) = Real(0.0);
            else
                tracer(i,j,k,0) = Real(0.01);
        });
    }
#endif
    else
    {
        amrex::Abort("Unknown double shear layer m_probtype");
    };
}

void incflo::init_plane_poiseuille (Box const& vbx, Box const& /*gbx*/,
                                    Array4<Real> const& vel,
                                    Array4<Real> const& /*density*/,
                                    Array4<Real> const& tracer,
                                    Box const& domain,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*dx*/,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                                    GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/) const
{
    Real dxinv = Real(1.0) / domain.length(0);
    Real dyinv = Real(1.0) / domain.length(1);
#if (AMREX_SPACEDIM == 3)
    Real dzinv = Real(1.0) / domain.length(2);
#else
    Real dzinv = 0.0;
#endif
    const auto dhi = amrex::ubound(domain);

    if (31 == m_probtype)
    {
        Real u = m_ic_u;
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5)*dyinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(6.0) * u * y * (Real(1.0)-y);,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (311 == m_probtype)
    {
        Real u = m_ic_u;
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(6.0) * u * z * (Real(1.0)-z);,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (41 == m_probtype)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.5)*z;,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && i <= dhi.x/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && i <= dhi.x/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && i <= dhi.x*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (42 == m_probtype)
    {
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vel(i,j,k,0) = 0.0;
            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
        });
    }
    else if (43 == m_probtype)
    {
        // Start from the zero-net-flux in/out profile that IncfloVelFill imposes on
        // the x faces (prob_bc.H), so that the outflow part of the direction_dependent
        // boundary has a non-zero velocity when the initial projection enforces
        // in/out solvability.
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = Real(j+0.5)*dyinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(6.0) * y * (Real(1.0)-y) - Real(1.0);,
                         vel(i,j,k,1) = Real(0.0);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
        });
    }
    else if (32 == m_probtype)
    {
        Real v = m_ic_v;
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = Real(k+0.5)*dzinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                         vel(i,j,k,1) = Real(6.0) * v * z * (Real(1.0)-z);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = 1.0;
            if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = 2.0;
            if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = 3.0;
        });
    }
    else if (322 == m_probtype)
    {
        Real v = m_ic_v;
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i+0.5)*dxinv;
            AMREX_D_TERM(vel(i,j,k,0) = Real(0.0);,
                         vel(i,j,k,1) = Real(6.0) * v * x * (Real(1.0)-x);,
                         vel(i,j,k,2) = Real(0.0););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && j <= dhi.y/8)   tracer(i,j,k,0) = Real(1.0);
            if (nt > 1 && j <= dhi.y/2)   tracer(i,j,k,1) = Real(2.0);
            if (nt > 2 && j <= dhi.y*3/4) tracer(i,j,k,2) = Real(3.0);
        });
    }
    else if (33 == m_probtype)
    {
#if (AMREX_SPACEDIM == 3)
        Real w = m_ic_w;
#endif
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 3)
            Real x = Real(i+0.5)*dxinv;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = 0.0;,
                         vel(i,j,k,1) = 0.0;,
                         vel(i,j,k,2) = Real(6.0) * w * x * (Real(1.0)-x););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && k <= dhi.z/8)   tracer(i,j,k,0) = Real(1.0);
            if (nt > 1 && k <= dhi.z/2)   tracer(i,j,k,1) = Real(2.0);
            if (nt > 2 && k <= dhi.z*3/4) tracer(i,j,k,2) = Real(3.0);
        });
    }
    else if (333 == m_probtype)
    {
#if (AMREX_SPACEDIM == 3)
        Real w = m_ic_w;
#endif
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 3)
            Real y = Real(j+0.5)*dyinv;
#endif
            AMREX_D_TERM(vel(i,j,k,0) = 0.0;,
                         vel(i,j,k,1) = 0.0;,
                         vel(i,j,k,2) = Real(6.0) * w * y * (Real(1.0)-y););

            const int nt = tracer.nComp();
            for (int n = 0; n < nt; ++n) {
                tracer(i,j,k,n) = 0.0;
            }
            if (nt > 0 && k <= dhi.z/Real(8))   tracer(i,j,k,0) = Real(1);
            if (nt > 1 && k <= dhi.z/Real(2))   tracer(i,j,k,1) = Real(2);
            if (nt > 2 && k <= dhi.z*3/Real(4)) tracer(i,j,k,2) = Real(3);
        });
    }
    else
    {
        amrex::Abort("Unknown plane poiseuille m_probtype");
    };
}

void incflo::init_burggraf (Box const& vbx, Box const& /*gbx*/,
                            Array4<Real> const& vel,
                            Array4<Real> const& /*density*/,
                            Array4<Real> const& /*tracer*/,
                            Box const& /*domain*/,
                            GpuArray<Real, AMREX_SPACEDIM> const& dx,
                            GpuArray<Real, AMREX_SPACEDIM> const& /*problo*/,
                            GpuArray<Real, AMREX_SPACEDIM> const& /*probhi*/)
{
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Same cell coordinates as the probtype-16 forcing (incflo_compute_forces.cpp),
        // the lid profile (prob_bc.H) and DiffFromExact: the cavity is [0,1]^2.
        Real x = (Real(i)+Real(0.5))*dx[0];
        Real y = (Real(j)+Real(0.5))*dx[1];
        vel(i,j,k,0) =  Real(8) * (x*x*x*x - Real(2) * x*x*x + x*x) * (Real(4)*y*y*y - Real(2)*y);
        vel(i,j,k,1) = -Real(8) * (Real(4)*x*x*x - Real(6) * x*x + Real(2)*x) * (y*y*y*y - y*y);
#if (AMREX_SPACEDIM == 3)
        vel(i,j,k,2) = Real(0);
#endif
    });
}

// probtype 603: the ballistic potential flow (prob_divu_ballistic.H) at t = 0, where the
// label of each point is the point itself
void incflo::init_divu_ballistic (Box const& vbx, Box const& /*gbx*/,
                                  Array4<Real> const& vel,
                                  Array4<Real> const& density,
                                  Array4<Real> const& tracer,
                                  Box const& /*domain*/,
                                  GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                  GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                  GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    GpuArray<Real,AMREX_SPACEDIM> const U{AMREX_D_DECL(m_ic_u, m_ic_v, m_ic_w)};
    GpuArray<int,AMREX_SPACEDIM> wall;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { wall[d] = geom[0].isPeriodic(d) ? 0 : 1; }
    Real const ro_0 = m_ro_0;
    bool const evolve_density = !m_constant_density;
    int const ntrac = m_ntrac;

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        IntVect const iv(AMREX_D_DECL(i,j,k));
        GpuArray<Real,AMREX_SPACEDIM> x;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
        }
        IncfloDivuBallistic r;
        incflo_divu_ballistic(x, Real(0.0), U, problo, probhi, wall, r);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            vel(i,j,k,d) = r.vel[d];
        }
        if (evolve_density) {
            density(i,j,k) = ro_0 * r.rho0;
        }
        for (int n = 0; n < ntrac; ++n) {
            tracer(i,j,k,n) = r.s0;
        }
    });
}

// probtype 603: the exact solution assumes these.  Also check that the particle paths do not
// cross before the end of the run (with a 10% margin), where the solution stops existing.
void incflo::check_divu_ballistic (int lev) const
{
    bool has_wall = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!geom[lev].isPeriodic(d)) {
            has_wall = true;
            if (m_bc_type[Orientation(d,Orientation::low)] != BC::no_slip_wall ||
                m_bc_type[Orientation(d,Orientation::high)] != BC::no_slip_wall) {
                amrex::Abort("incflo.probtype = 603 needs no-slip walls in the non-periodic directions");
            }
        }
    }
    bool zero_gravity = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { zero_gravity = zero_gravity && m_gravity[d] == 0.0; }
    bool zero_tracer_diffusion = true;
    if (m_advect_tracer) {
        for (int n = 0; n < m_ntrac; ++n) { zero_tracer_diffusion = zero_tracer_diffusion && m_mu_s[n] == 0.0; }
    }
    if (!m_use_divu_source || m_fluid_model != FluidModel::Newtonian || !zero_gravity ||
        !zero_tracer_diffusion) {
        amrex::Abort("incflo.probtype = 603 requires incflo.use_divu_source = 1, a Newtonian fluid, "
                     "no gravity and no tracer diffusion");
    }
    if (has_wall && (m_ic_u != 0.0 || m_ic_v != 0.0 || m_ic_w != 0.0)) {
        amrex::Abort("incflo.probtype = 603 with walls needs incflo.ic_u = ic_v = ic_w = 0");
    }

    Real t_end = m_stop_time;
    if (t_end <= 0.0 && m_fixed_dt > 0.0 && m_max_step > 0) { t_end = m_max_step * m_fixed_dt; }
    if (t_end <= 0.0) {
        amrex::Print() << "WARNING: incflo.probtype = 603: cannot check that the particle paths "
                       << "do not cross before the end of the run\n";
        return;
    }
    Real const t_check = Real(1.1) * t_end;
    auto const& dx = geom[lev].CellSizeArray();
    auto const& problo = geom[lev].ProbLoArray();
    auto const& probhi = geom[lev].ProbHiArray();
    GpuArray<int,AMREX_SPACEDIM> wall;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { wall[d] = geom[lev].isPeriodic(d) ? 0 : 1; }
    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    for (MFIter mfi(m_leveldata[lev]->density); mfi.isValid(); ++mfi) {
        reduce_op.eval(mfi.validbox(), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real>
        {
            amrex::ignore_unused(k);
            IntVect const iv(AMREX_D_DECL(i,j,k));
            Real a[AMREX_SPACEDIM];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                a[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
            }
            return { incflo_divu_ballistic_minor(a, t_check, problo, probhi, wall) };
        });
    }
    Real minor = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealMin(minor);
    if (minor <= 0.0) {
        amrex::Abort("incflo.probtype = 603: the particle paths cross before (or within 10% of) "
                     "the end of the run; stop earlier");
    }
}

void incflo::init_divu_mms (Box const& vbx, Box const& /*gbx*/,
                            Array4<Real> const& vel,
                            Array4<Real> const& /*density*/,
                            Array4<Real> const& /*tracer*/,
                            Box const& /*domain*/,
                            GpuArray<Real, AMREX_SPACEDIM> const& dx,
                            GpuArray<Real, AMREX_SPACEDIM> const& problo,
                            GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    // The steady manufactured velocity (prob_divu_mms.H)
    Real const mu = m_mu;
    Real const ro_0 = m_ro_0;

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        IntVect const iv(AMREX_D_DECL(i,j,k));
        GpuArray<Real,AMREX_SPACEDIM> x;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            x[d] = problo[d] + (Real(iv[d]) + Real(0.5)) * dx[d];
        }
        IncfloDivuMMS r;
        incflo_divu_mms(x, problo, probhi, mu, ro_0, r);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            vel(i,j,k,d) = r.vel[d];
        }
    });
}

// probtype 605: a channel along x with a pulsing source of S (prob_divu_source). The
// velocity and density keep their uniform initial values (incflo.ic_u, incflo.ro_0). Every
// tracer gets the same bump just upstream of the source, so the conservative and convective
// forms (incflo.trac_is_conservative) can be compared after it passes the expanding region.
void incflo::init_divu_plume (Box const& vbx, Box const& /*gbx*/,
                              Array4<Real> const& /*vel*/,
                              Array4<Real> const& /*density*/,
                              Array4<Real> const& tracer,
                              Box const& /*domain*/,
                              GpuArray<Real, AMREX_SPACEDIM> const& dx,
                              GpuArray<Real, AMREX_SPACEDIM> const& problo,
                              GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    // The source is centered at x = 0.25 L_x; the bump is centered 0.075 L_x upstream of it
    // and is negligible (exp(-34)) at the inflow, where the tracers are 1
    Real const Lx = probhi[0] - problo[0];
    Real const xt = problo[0] + Real(0.175) * Lx;
    Real const w = Real(0.03) * Lx;
    int const ntrac = m_ntrac;
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real const xr = (problo[0] + (Real(i) + Real(0.5)) * dx[0] - xt) / w;
        Real const s = Real(1.0) + std::exp(-xr * xr);
        for (int n = 0; n < ntrac; ++n) {
            tracer(i,j,k,n) = s;
        }
    });
}

// probtype 606: a warm layer carried through a channel along x, with S = kappa (T - T_ref)
// (prob_divu_temperature.H)
void incflo::init_divu_temperature (Box const& vbx, Box const& /*gbx*/,
                                    Array4<Real> const& vel,
                                    Array4<Real> const& density,
                                    Array4<Real> const& tracer,
                                    Array4<Real> const& temperature,
                                    Box const& /*domain*/,
                                    GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                    GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                    GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    Orientation const xlo(0, Orientation::low);
    Real const U = m_bc_velocity[xlo][0];
    Real const rho0 = m_bc_density[xlo];
    Real const T_ref = m_bc_temperature[xlo];
    auto const q = incflo_divu_temperature_parm(problo[0], probhi[0]);
    auto const tot = incflo_divu_temperature_int(q.amax, Real(0.0), q);
    Real const xhi = probhi[0];
    int const ntrac = m_ntrac;

    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
        IncfloDivuTemperature r;
        incflo_divu_temperature(x, Real(0.0), U, rho0, Real(0.0), xhi, q, tot, r);
        vel(i,j,k,0) = r.u;
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            vel(i,j,k,d) = Real(0.0);
        }
        density(i,j,k) = r.rho;
        temperature(i,j,k) = T_ref + r.theta;
        for (int n = 0; n < ntrac; ++n) {
            tracer(i,j,k,n) = r.theta / q.dT;
        }
    });
}

// probtype 606: the exact solution assumes these.  Also check that the warm layer stays away
// from the outflow until the end of the run (with a 10% margin), so that S = 0 there.
void incflo::check_divu_temperature (int lev) const
{
    Orientation const xlo(0, Orientation::low);
    Orientation const xhi(0, Orientation::high);
    bool periodic_across = true;
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        periodic_across = periodic_across && geom[lev].isPeriodic(d);
    }
    if (m_bc_type[xlo] != BC::mass_inflow || m_bc_velocity[xlo][0] <= 0.0 ||
        m_bc_type[xhi] != BC::pressure_outflow || m_bc_pressure[xhi] != 0.0 || !periodic_across) {
        amrex::Abort("incflo.probtype = 606 needs mass inflow with a positive velocity at xlo, "
                     "pressure outflow with xhi.pressure = 0, and periodic boundaries across the "
                     "channel");
    }
    bool zero_gravity = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { zero_gravity = zero_gravity && m_gravity[d] == 0.0; }
    bool zero_tracer_diffusion = true;
    if (m_advect_tracer) {
        for (int n = 0; n < m_ntrac; ++n) { zero_tracer_diffusion = zero_tracer_diffusion && m_mu_s[n] == 0.0; }
    }
    if (!m_use_divu_source || !m_use_temperature || m_mu_T != 0.0 || m_constant_density ||
        m_fluid_model != FluidModel::Newtonian || !zero_gravity || !zero_tracer_diffusion) {
        amrex::Abort("incflo.probtype = 606 requires incflo.use_divu_source = 1, "
                     "incflo.use_temperature = 1, incflo.mu_T = 0, incflo.constant_density = 0, "
                     "a Newtonian fluid, no gravity and no tracer diffusion");
    }

    Real t_end = m_stop_time;
    if (t_end <= 0.0 && m_fixed_dt > 0.0 && m_max_step > 0) { t_end = m_max_step * m_fixed_dt; }
    if (t_end <= 0.0) {
        amrex::Print() << "WARNING: incflo.probtype = 606: cannot check that the warm layer stays "
                       << "away from the outflow until the end of the run\n";
        return;
    }
    Real const t_check = Real(1.1) * t_end;
    auto const q = incflo_divu_temperature_parm(geom[lev].ProbLo(0), geom[lev].ProbHi(0));
    Real const aL = geom[lev].ProbHi(0) - m_bc_velocity[xlo][0] * t_check
        - incflo_divu_temperature_int(q.amax, t_check, q).E;
    if (aL < q.amax) {
        amrex::Abort("incflo.probtype = 606: the warm layer reaches the outflow before (or within "
                     "10% of) the end of the run; stop earlier");
    }
}
