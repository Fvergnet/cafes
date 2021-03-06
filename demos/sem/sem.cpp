#include <cafes.hpp>
#include <petsc.h>

void zeros(const PetscReal x[], PetscScalar *u){
  *u = 0.;
}

void ones(const PetscReal x[], PetscScalar *u){
  *u = 1.;
}

void ones_m(const PetscReal x[], PetscScalar *u){
  *u = -1.;
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    std::size_t const dim = 2;

    ierr = PetscInitialize(&argc, &argv,  (char *)0, (char *)0);CHKERRQ(ierr);

    auto bc = cafes::make_bc<dim>({ {{zeros , zeros}} 
                                  , {{zeros , zeros}}
                                  , {{zeros, zeros}}
                                  , {{zeros, zeros}}
                                  });
    auto rhs = cafes::make_rhs<dim>({{ zeros, zeros }});
    auto st = cafes::make_stokes<dim>(bc, rhs);

    // auto se1 = cafes::make_ellipsoid<2>( {.3,.5}, {.1,.2}, 2., M_PI/2);
    // auto se3 = cafes::make_ellipsoid<2>( { .5, .5}, {.1,.1}, 2., 0);
    // auto se2 = cafes::make_ellipsoid<2>( {.7,.5}, {.1,.1}, 1., 0);

    double r = 0.05;
    auto se1 = cafes::make_circle( {.5 - r - r/4, .5}, r);
    auto se2 = cafes::make_circle( {.5 + r + r/4, .5}, r);
    std::vector<cafes::particle<decltype(se1)>> pt{ cafes::make_particle_with_force(se1, {1., 0}, 1.),
                                                    cafes::make_particle_with_force(se2, {1., 0}, 1.)
                                                 };

    auto s = cafes::make_SEM(pt, st, st.ctx->h[0]);
    ierr = s.create_Mat_and_Vec();CHKERRQ(ierr);
    ierr = s.setup_RHS();CHKERRQ(ierr);
    ierr = s.setup_KSP();CHKERRQ(ierr);
    ierr = s.solve();CHKERRQ(ierr);

    s.get_new_velocities();
    std::cout << s.parts_[0].velocity_[0] << " " << s.parts_[0].velocity_[1] << "\n";
    std::cout << s.parts_[1].velocity_[0] << " " << s.parts_[1].velocity_[1] << "\n";
    ierr = cafes::io::save_VTK("Resultats", "test", st.sol, st.ctx->dm, st.ctx->h);CHKERRQ(ierr);
    // // PetscViewer      viewer;
    // // ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "solution.h5", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    // // ierr = VecView(st.sol, viewer);CHKERRQ(ierr);
    // // ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscFinalize();CHKERRQ(ierr);

    return 0;
}