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

    // auto c = cafes::myparticle({.5,.5}, {1.,0.}, 0.1);
    // c.discretize_surface(10);

    double R1 = .1;
    double distance = st.opt.distance;
    bool truncature = 1;

    auto se1 = cafes::make_circle({.5 - .5*distance - R1, .5}, R1, 0);
    auto se2 = cafes::make_circle({.5 + .5*distance + R1, .5}, R1, 0);
    std::vector<cafes::particle<decltype(se1)>> pt{
        cafes::make_particle_with_velocity(se1, {1., 0.}, 0.),
        cafes::make_particle_with_velocity(se2, {-1., 0.}, 0.)
    };

    auto s = cafes::make_DtoN(pt, st, st.ctx->h[0]);

    ierr = s.create_Mat_and_Vec();
    CHKERRQ(ierr);

    // ierr = s.setup_RHS();
    // CHKERRQ(ierr);
    // ierr = s.setup_KSP();
    // CHKERRQ(ierr);

    cafes::singularity::add_singularity_to_ureg(st.ctx->dm, st.ctx->h, st.sol, pt, truncature);

    std::string stout = "Babic_singular_fields_";
    stout.append("distance_is_R_over_");
    stout.append(std::to_string(int(std::round(R1/distance))));
    const char * stw = stout.c_str();
    ierr = cafes::io::save_hdf5("singular_fields", stw, st.sol, st.ctx->dm,
                              st.ctx->h);
    // CHKERRQ(ierr);

    // ierr = s.solve();
    // CHKERRQ(ierr);

    // std::ofstream myfile;
    // std::string filename = "simulation_distance_is_radius_over_"+std::to_string(int(std::round(R1/distance)))+"_compute_sing_is_"+std::to_string(singularity)+".txt";
    // myfile.open(filename,std::ios::app);
    // myfile << R1 << " " << distance << " " << R1/distance << " " << singularity << " " << mx << " " << s.kspiter << "\n";
    // myfile.close();

    // // ff.loadmesh("test.txt");
    // // auto test = mesh.interpolate(ctx, st.sol);

    // // cafes::singularity::add_singularity_to_ureg(st.ctx->dm, st.ctx->h, st.sol, pt);

    // std::string stout = "two_parts_solution_compute_sing_is_"+std::to_string(singularity);
    // // std::string stout = "rhs_sing_";
    // stout.append("_distance_");
    // stout.append(std::to_string(distance));
    // stout.append("_mx_"+std::to_string(mx));
    // const char * stw = stout.c_str();
    // ierr = cafes::io::save_hdf5("Resultats", stw, st.sol, st.ctx->dm,
    //                           st.ctx->h);
    // CHKERRQ(ierr);

    // auto ctx = make_interpolation_context(st.ctx->dm, {2*st.ctx->h[0], 2*st.ctx->h[1]}, pt[0], pt[1], singularity);
    // auto ff = FenicsFunction("reference_mesh_distance_is_radius_over_"+std::to_string(int(std::round(R1/distance)))+"_velocity.txt", "reference_mesh_distance_is_radius_over_"+std::to_string(int(std::round(R1/distance)))+"_pressure.txt");
    // ff.interpolate(ctx, st.sol);
    // ff.save(stout);

    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}
