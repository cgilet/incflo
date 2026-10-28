#include <incflo.H>

using namespace amrex;

//
// Source term S in the divergence constraint div(u) = S.
//
// S^n is the S^(n+1) computed at the end of the previous step: the projection does not
// change the scalars, so there is nothing to recompute at the start of a step.  The MAC
// projection and the Godunov edge-state prediction use S^(n+1/2) = S^n + dt/2 dS/dt,
// where dS/dt is lagged from the previous step, and the nodal projection uses S^(n+1).
// This follows IAMR.
//

void
incflo::compute_divu_source (Vector<MultiFab      *> const& divu_source,
                             Vector<MultiFab const*> const& density,
                             Vector<MultiFab const*> const& tracer,
                             Vector<MultiFab const*> const& temperature,
                             Real time)
{
    BL_PROFILE("incflo::compute_divu_source()");

    for (int lev = 0; lev <= finest_level; ++lev) {
        // temperature is empty unless incflo.use_temperature = 1
        prob_divu_source(lev, *divu_source[lev], density[lev], tracer[lev],
                         temperature.empty() ? nullptr : temperature[lev], time);
    }
}

// Compute S from the new-time state into divu_source.  The coarse data under the fine
// levels is replaced by the average of the fine data, as for the state.
void
incflo::update_divu_source (Real time)
{
    if (!m_use_divu_source) { return; }

    // Fill one ghost cell of the state, so that compute_divu_source can use a stencil
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto& ld = *m_leveldata[lev];
        fillpatch_density(lev, m_t_new[lev], ld.density, 1);
        if (m_advect_tracer) {
            fillpatch_tracer(lev, m_t_new[lev], ld.tracer, 1);
        }
        if (m_use_temperature) {
            fillpatch_temperature(lev, m_t_new[lev], ld.temperature, 1);
        }
    }

    compute_divu_source(get_divu_source_new(), get_density_new_const(),
                        get_tracer_new_const(), get_temperature_new_const(), time);

    for (int lev = finest_level-1; lev >= 0; --lev) {
        amrex::average_down(m_leveldata[lev+1]->divu_source, m_leveldata[lev]->divu_source,
                            0, 1, refRatio(lev));
    }
}

void
incflo::compute_divu_source_nph ()
{
    if (!m_use_divu_source) { return; }

    Real const half_dt = Real(0.5) * m_dt;
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto& ld = *m_leveldata[lev];
        MultiFab::LinComb(ld.divu_source_nph, Real(1.0), ld.divu_source_o, 0,
                          half_dt, ld.dsdt, 0, 0, 1, 0);
    }

    // The edge-state prediction reads S in ghost cells, including those at the
    // coarse/fine boundary
    Real const time_nph = m_cur_time + half_dt;
    auto divu_source_nph = get_divu_source_nph();
    for (int lev = 0; lev <= finest_level; ++lev) {
        fillpatch_divu_source(lev, time_nph, *divu_source_nph[lev], divu_source_nph,
                              nghost_divu_source_nph());
    }
}

void
incflo::compute_dsdt ()
{
    if (!m_use_divu_source) { return; }

    Real const dtinv = Real(1.0) / m_dt;
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto& ld = *m_leveldata[lev];
        MultiFab::LinComb(ld.dsdt, dtinv, ld.divu_source, 0,
                          -dtinv, ld.divu_source_o, 0, 0, 1, 0);
    }
}

bool
incflo::closed_domain () const
{
    for (OrientationIter oit; oit; ++oit) {
        auto const bct = m_bc_type[oit()];
        if (bct != BC::periodic && bct != BC::slip_wall && bct != BC::no_slip_wall) {
            return false;
        }
    }
    return true;
}

//
// A closed domain (only walls and periodic boundaries) needs
// incflo.allow_closed_with_divu_source = 1.  The projections are then singular, and MLMG removes
// the mean of their right-hand sides, so they enforce div(u) = S - mean(S).  The Godunov
// edge-state prediction also reads S^(n+1/2) as div(u_mac): for the quantities advected in
// convective form, and in 3D, through the corner coupling, for all of them.  So
// compute_MAC_projected_velocities removes that mean from S^(n+1/2) itself, with the function
// MLMG uses; otherwise the prediction would be off by mean(S), and those quantities would be
// only first order in time.  The S^(n+1) of the nodal projection is not used anywhere else, so
// there MLMG's removal is enough.
//
// With a uniform 1/(gamma p0), S - mean(S) is the closed-chamber low Mach number constraint
// S - dp0/dt / (gamma p0), but incflo neither evolves p0 nor adds the compression heating
// dp0/dt / (rho cp) to the temperature equation.  The integral of S is a sum over the cells, so
// it is zero only to truncation error even when the exact integral is zero.
//
void
incflo::print_divu_source_integral () const
{
    if (!m_use_divu_source || !closed_domain() || m_verbose <= 0) { return; }

    Vector<MultiFab const*> S(finest_level+1);
    Vector<MultiFab> abs_S(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        S[lev] = &m_leveldata[lev]->divu_source;
        abs_S[lev].define(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        MultiFab::Copy(abs_S[lev], *S[lev], 0, 0, 1, 0);
        abs_S[lev].abs(0, 1);
    }
    Real const sum_S     = amrex::volumeWeightedSum(S, 0, Geom(), refRatio());
    Real const sum_abs_S = amrex::volumeWeightedSum(GetVecOfConstPtrs(abs_S), 0, Geom(),
                                                    refRatio());
    Real const ratio = (sum_abs_S > Real(0.0)) ? std::abs(sum_S) / sum_abs_S : Real(0.0);

    amrex::Print() << "Closed domain: integral of S = " << sum_S << ", " << ratio
                   << " of the integral of |S|\n";
}
