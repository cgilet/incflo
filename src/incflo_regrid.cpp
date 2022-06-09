#include <incflo.H>

using namespace amrex;

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void incflo::MakeNewLevelFromCoarse (int lev,
                                     Real time,
                                     const BoxArray& ba,
                                     const DistributionMapping& dm)
{
    BL_PROFILE("incflo::MakeNewLevelFromCoarse()");

    if (m_verbose > 0) {
        amrex::Print() << "Making new level " << lev << " from coarse" << std::endl;
    }

#ifdef AMREX_USE_EB
    std::unique_ptr<EBFArrayBoxFactory> new_fact = makeEBFabFactory(geom[lev], ba, dm,
                                                                    {nghost_eb_basic(),
                                                                    nghost_eb_volume(),
                                                                    nghost_eb_full()},
                                                                    EBSupport::full);
#else
    std::unique_ptr<EBFabFactory<FArrayBox> > new_fact(new FArrayBoxFactory());
#endif

    std::unique_ptr<LevelData> new_leveldata
        (new LevelData(ba, dm, *new_fact, m_ntrac, nghost_state(),
                       m_advection_type,
                       m_diff_type==DiffusionType::Implicit,
                       use_tensor_correction,
                       m_advect_tracer));

    fillcoarsepatch_velocity(lev, time, new_leveldata->velocity, 0);
    fillcoarsepatch_density(lev, time, new_leveldata->density, 0);
    if (m_ntrac > 0) {
        fillcoarsepatch_tracer(lev, time, new_leveldata->tracer, 0);
    }
    fillcoarsepatch_gradp(lev, time, new_leveldata->gp, 0);
    new_leveldata->p_nd.setVal(0.0);
    new_leveldata->p_cc.setVal(0.0);

    m_leveldata[lev] = std::move(new_leveldata);
#ifdef INCFLO_USE_MOVING_EB
    m_old_factory[lev] = std::move(m_new_factory[lev]);
    m_new_factory[lev] = std::move(new_fact);
#else
    m_factory[lev] = std::move(new_fact);
#endif

    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();

#ifdef AMREX_USE_EB
    macproj.reset(new Hydro::MacProjector(Geom(0,finest_level),
                      MLMG::Location::FaceCentroid,  // Location of mac_vec
                      MLMG::Location::FaceCentroid,  // Location of beta
                      MLMG::Location::CellCenter  ) ); // Location of solution variable phi
#else
    macproj.reset(new Hydro::MacProjector(Geom(0,finest_level)));
#endif
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void incflo::RemakeLevel (int lev, Real time, const BoxArray& ba,
                          const DistributionMapping& dm)
{
    BL_PROFILE("incflo::RemakeLevel()");

    if (m_verbose > 0) {
        amrex::Print() << "Remaking level " << lev << std::endl;
    }

#ifdef AMREX_USE_EB
    std::unique_ptr<EBFArrayBoxFactory> new_fact = makeEBFabFactory(geom[lev], ba, dm,
                                                                    {nghost_eb_basic(),
                                                                     nghost_eb_volume(),
                                                                     nghost_eb_full()},
                                                                     EBSupport::full);
#else
    std::unique_ptr<FabFactory<FArrayBox> > new_fact(new FArrayBoxFactory());
#endif
    std::unique_ptr<LevelData> new_leveldata
        (new LevelData(ba, dm, *new_fact, m_ntrac, nghost_state(),
                       m_advection_type,
                       m_diff_type==DiffusionType::Implicit,
                       use_tensor_correction,
                       m_advect_tracer));
    fillpatch_velocity(lev, time, new_leveldata->velocity, 0);
    fillpatch_density(lev, time, new_leveldata->density, 0);
    if (m_ntrac > 0) {
        fillpatch_tracer(lev, time, new_leveldata->tracer, 0);
    }
    fillpatch_gradp(lev, time, new_leveldata->gp, 0);
    new_leveldata->p_nd.setVal(0.0);
    new_leveldata->p_cc.setVal(0.0);

    m_leveldata[lev] = std::move(new_leveldata);
#ifdef INCFLO_USE_MOVING_EB
    m_old_factory[lev] = std::move(m_new_factory[lev]);
    m_new_factory[lev] = std::move(new_fact);
#else
    m_factory[lev] = std::move(new_fact);
#endif

    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();

#ifdef AMREX_USE_EB
    macproj.reset(new Hydro::MacProjector(Geom(0,finest_level),
                      MLMG::Location::FaceCentroid,  // Location of mac_vec
                      MLMG::Location::FaceCentroid,  // Location of beta
                      MLMG::Location::CellCenter  ) ); // Location of solution variable phi
#else
    macproj.reset(new Hydro::MacProjector(Geom(0,finest_level)));
#endif
}

#ifdef AMREX_USE_EB
#ifdef INCFLO_USE_MOVING_EB
// Remake an existing level with a new geometry but nothing else changed
void incflo::RemakeLevelWithNewGeometry (int lev, Real time)
{
    BL_PROFILE("incflo::RemakeLevelWithNewGeometry()");

    if (m_verbose > 0) {
        amrex::Print() << "Remaking level " << lev << " with new geometry" << std::endl;
    }

    // Erase old EB
    EB2::IndexSpace::erase(const_cast<EB2::IndexSpace*>(eb_old));

    // Build a new EB
    MakeEBGeometry(time);
    eb_old = eb_new;
    eb_new = &(EB2::IndexSpace::top());

    m_old_factory[lev] = std::move(m_new_factory[lev]);

    m_new_factory[lev] = makeEBFabFactory(eb_new, geom[lev], grids[lev], dmap[lev],
                                          {nghost_eb_basic(),
                                           nghost_eb_volume(),
                                           nghost_eb_full()},
                                          EBSupport::full);

    std::unique_ptr<LevelData> new_leveldata
        (new LevelData(grids[lev], dmap[lev], *m_new_factory[lev], m_ntrac, nghost_state(),
                       m_advection_type,
                       m_diff_type==DiffusionType::Implicit,
                       use_tensor_correction,
                       m_advect_tracer));

    MultiFab::Copy(new_leveldata->velocity, m_leveldata[lev]->velocity,0,0,AMREX_SPACEDIM,0);
    MultiFab::Copy(new_leveldata->density , m_leveldata[lev]->density ,0,0,1,0);
    if (m_ntrac > 0) {
        MultiFab::Copy(new_leveldata->tracer, m_leveldata[lev]->tracer,0,0,1,0);
    }
    MultiFab::Copy(new_leveldata->gp   , m_leveldata[lev]->gp   ,0,0,AMREX_SPACEDIM,0);
    MultiFab::Copy(new_leveldata->p_nd , m_leveldata[lev]->p_nd ,0,0,1,0);
    MultiFab::Copy(new_leveldata->p_cc , m_leveldata[lev]->p_cc ,0,0,1,0);

    // Let's fill the newly covered cells with 1e45 to be different
    EB_set_covered( new_leveldata->velocity, 1.e45);
    EB_set_covered( new_leveldata->density , 1.e45);
    if (m_ntrac > 0) 
        EB_set_covered( new_leveldata->tracer  , 1.e45);
    EB_set_covered( new_leveldata->gp , 1.e45);

    m_leveldata[lev] = std::move(new_leveldata);
}
#endif
#endif

// Delete level data
// overrides the pure virtual function in AmrCore
void incflo::ClearLevel (int lev)
{
    BL_PROFILE("incflo::ClearLevel()");
    m_leveldata[lev].reset();
#ifdef INCFLO_USE_MOVING_EB
    m_new_factory[lev].reset();
    m_old_factory[lev].reset();
#else
    m_factory[lev].reset();
#endif
    m_diffusion_tensor_op.reset();
    m_diffusion_scalar_op.reset();
    macproj.reset();
}
