#include <cafes.hpp>
#include <petsc.h>

#include <particle/singularity/add_singularity.hpp>
#include "fenicsfunction.hpp"

#include <iostream>
#include <fstream>

void zeros(const PetscReal x[], PetscScalar *u)
{
    *u = 0.;
}

void ones(const PetscReal x[], PetscScalar *u)
{
    *u = 1.;
}

void ones_m(const PetscReal x[], PetscScalar *u)
{
    *u = -1.;
}

int main(int argc, char **argv)
{

    PetscErrorCode ierr;
    std::size_t const dim = 2;

    ierr = PetscInitialize(&argc, &argv, (char *)0, (char *)0);
    CHKERRQ(ierr);

    auto bc = cafes::make_bc<dim>({
        {{zeros, zeros}} // gauche
        ,
        {{zeros, zeros}} // droite
        ,
        {{zeros, zeros}} // bas
        ,
        {{zeros, zeros}} // haut
    });

    auto rhs = cafes::make_rhs<dim>({{zeros, zeros}});

    auto st = cafes::make_stokes<dim>(bc, rhs);
    int const mx = st.opt.mx[0] - 1;
    bool singularity = st.opt.compute_singularity;

    double R1 = .1;
    double distance = st.opt.distance;
    std::string saverep = st.opt.saverep;

    auto se1 = cafes::make_circle({.5 - .5*distance - R1, .5}, R1, 0);
    auto se2 = cafes::make_circle({.5 + .5*distance + R1, .5}, R1, 0);
    std::vector<cafes::particle<decltype(se1)>> pt{
        cafes::make_particle_with_velocity(se1, {1., 0.}, 0.),
        cafes::make_particle_with_velocity(se2, {-1., 0.}, 0.)
    };

    auto s = cafes::make_DtoN(pt, st, st.ctx->h[0]);


    ierr = s.create_Mat_and_Vec();
    CHKERRQ(ierr);

    st.setup_RHS();
    st.setup_KSP();

    ierr = cafes::singularity::add_singularity_in_fluid<dim>(*s.ctx);CHKERRQ(ierr);

    st.solve();
    ierr = cafes::io::save_hdf5(saverep.c_str(), "stokes_singular_rhs", st.sol, st.ctx->dm, st.ctx->h);

    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}
