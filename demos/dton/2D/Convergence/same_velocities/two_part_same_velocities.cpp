#include <cafes.hpp>
#include <petsc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <particle/singularity/add_singularity.hpp>

void zeros(const PetscReal x[], PetscScalar *u)
{
    *u = 0.;
}

void ones(const PetscReal x[], PetscScalar *u)
{
    *u = 10.;
}

void ones_m(const PetscReal x[], PetscScalar *u)
{
    *u = -1.;
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    std::size_t const dim = 2;
    int const nref = 256;
    std::string saverep = "Results/";
    std::string refrep = "Reference/";
    const char * srep = saverep.c_str();
    const char * rrep = refrep.c_str();


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
    

    double R1 = .05;
    double R2 = .05;
    double dist = .05/5;

    auto se1 = cafes::make_circle({.5 - R1 - dist / 2, .5}, R1, 0);
    auto se2 = cafes::make_circle({.5 + R2 + dist / 2, .5}, R2, 0);

    std::vector<cafes::particle<decltype(se1)>> pt{
        cafes::make_particle_with_velocity(se1, {1., 0.}, 0.),
        cafes::make_particle_with_velocity(se2, {1., 0.}, 0.)};

    auto s = cafes::make_DtoN(pt, st, .1);

    ierr = s.create_Mat_and_Vec();
    CHKERRQ(ierr);

    ierr = s.setup_RHS();
    CHKERRQ(ierr);
    ierr = s.setup_KSP();
    CHKERRQ(ierr);
    ierr = s.solve();
    CHKERRQ(ierr);

    // COARSE DMDA INFO
    DM davCoarse, dapCoarse;
    DMDALocalInfo infopCoarse, infovCoarse;
    ierr = DMCompositeGetEntries(st.ctx->dm, &davCoarse, &dapCoarse);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(davCoarse, &infovCoarse);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dapCoarse, &infopCoarse);CHKERRQ(ierr);
    std::cout << "Size Coarse Mesh.   Pressure : " << infopCoarse.mx << " " << infopCoarse.my << " Velocity : " << infovCoarse.mx << " " << infovCoarse.my  << std::endl;
    
    
    // SAVE SOLUTIONS

    std::string stout0 = "two_part_rhs_";
    stout0.append(std::to_string(mx));
    const char * stw0 = stout0.c_str();
    ierr = cafes::io::save_VTK(srep, stw0, s.sol_rhs, st.ctx->dm,
                               st.ctx->h);
    CHKERRQ(ierr);

    stout0 = "two_part_tmp_";
    stout0.append(std::to_string(mx));
    stw0 = stout0.c_str();
    ierr = cafes::io::save_VTK(srep, stw0, s.sol_tmp, st.ctx->dm,
                               st.ctx->h);
    CHKERRQ(ierr);

    stout0 = "two_part_reg_";
    stout0.append(std::to_string(mx));
    stw0 = stout0.c_str();
    ierr = cafes::io::save_VTK(srep, stw0, s.sol_reg, st.ctx->dm,
                               st.ctx->h);
    CHKERRQ(ierr);

    // REFINEMENT AND INTERPOLATION
    int const fine = nref/mx;
    std::array<int, dim> refine = {fine,fine};
    std::cout << "Refinment factors : " << refine[0] << " " << refine[1] << std::endl;
    Vec sol_refine;
    DM dm_refine;
    std::array<double, dim> h_refine;
    h_refine[0] = st.ctx->h[0]/refine[0];
    h_refine[1] = st.ctx->h[1]/refine[1];

    ierr = cafes::posttraitement::linear_interpolation(st.ctx->dm, s.sol_reg, dm_refine, sol_refine, refine, st.ctx->h);CHKERRQ(ierr);
    

    // DMDA INFO
    DM dav, dap;
    DMDALocalInfo infop, infov;

    ierr = DMCompositeGetEntries(dm_refine, &dav, &dap);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dav, &infov);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dap, &infop);CHKERRQ(ierr);

    // ZEROS IN PARTICLES (REFINED SOLUTION)
    auto solu = cafes::petsc::petsc_vec<dim>(dm_refine, sol_refine, 0, false);
    auto solp = cafes::petsc::petsc_vec<dim>(dm_refine, sol_refine, 1, false);

    for (std::size_t j =0; j< infov.my; j++ )
    {
        for (std::size_t i = 0; i <infov.mx; i++)
        {
            auto pos = cafes::geometry::position<int, dim>{i,j};
            auto pts = cafes::geometry::position<double, dim>{i*h_refine[0], j*h_refine[1]};
            auto usol = solu.at_g(pos);
            if (se1.contains(pts) or se2.contains(pts))
            {
                usol[0] = 0.;
                usol[1] = 0.;
            }
        }
    }

    for (std::size_t j =0; j<infop.my; j++ )
    {
        for (std::size_t i = 0; i <infop.mx; i++)
        {
            auto pos = cafes::geometry::position<int, dim>{i,j};
            auto pts = cafes::geometry::position<double, dim>{2*i*h_refine[0], 2*j*h_refine[1]};
            auto psol = solp.at_g(pos);
            if (se1.contains(pts) or se2.contains(pts))
            {
                psol[0] = 0.;
            }
        }
    }
    
    std::string stout1 = "two_part_refined";
    stout1.append(std::to_string(mx));
    const char * stw1 = stout1.c_str();
    ierr = cafes::io::save_VTK(srep, stw1, sol_refine, dm_refine, h_refine);CHKERRQ(ierr);

    stout1 = "two_part_total_";
    stout1.append(std::to_string(mx));
    stw1 = stout1.c_str();
    ierr = cafes::io::save_VTK(srep, stw1, sol_refine, dm_refine, h_refine);CHKERRQ(ierr);

    
    // READ REFERENCE SOLUTION
    Vec solref;
    ifstream file;
    std::stringstream op, ov;
    double tmp, tmpx, tmpy;

    std::cout << "Size Reference Mesh.   Pressure : " << infop.mx << " " << infop.my << " Velocity : " << infov.mx << " " << infov.my  << std::endl;

    ierr = DMCreateGlobalVector(dm_refine, &solref);CHKERRQ(ierr);
    auto solrefu = cafes::petsc::petsc_vec<dim>(dm_refine, solref, 0, false);
    auto solrefp = cafes::petsc::petsc_vec<dim>(dm_refine, solref, 1, false);


    // Fill pressure
    op << rrep << "reference_sem_pressure_" << nref << ".txt";
    file.open(op.str());
    for (std::size_t j =0; j<infop.my; j++ )
    {
        for (std::size_t i = 0; i < infop.mx; i++)
        {
            auto pos = cafes::geometry::position<int, dim>{i,j};
            auto pts = cafes::geometry::position<double, dim>{2*i*h_refine[0], 2*j*h_refine[1]};
            auto u = solrefp.at_g(pos);
            file >> tmp;
            if (se1.contains(pts) or se2.contains(pts))
            {
                u[0] = 0.;
            }
            else
            {
                u[0] = tmp;
            }
            //cout << u[0] << endl;
        }
    }
    file.close();

    // Fill velocity
    //ov << "reference_sem_velocity.txt";
    ov << rrep << "reference_sem_velocity_" << nref << ".txt";
    file.open(ov.str());
    for (std::size_t j =0; j<infov.my; j++ )
    {
        for (std::size_t i = 0; i < infov.mx; i++)
        {
            auto pos = cafes::geometry::position<int, dim>{i,j};
            auto pts = cafes::geometry::position<double, dim>{i*h_refine[0], j*h_refine[1]};
            auto u = solrefu.at_g(pos);
            file >> tmpx;
            file >> tmpy;
            if (se1.contains(pts) or se2.contains(pts))
            {
                u[0] = 0.;
                u[1] = 0.;
            }
            else
            {
                u[0] = tmpx;
                u[1] = tmpy;
            }
        }
    }
    file.close();


    ierr = cafes::io::save_VTK(srep, "reference_solution", solref, dm_refine, h_refine);CHKERRQ(ierr);

    // COMPUTE ERROR
    Vec error;
    VecDuplicate(sol_refine, &error);
    VecWAXPY(error, -1., sol_refine, solref);

    std::string stout2 = "error_";
    stout2.append(std::to_string(mx));
    const char * stw2 = stout2.c_str();
    ierr = cafes::io::save_VTK(srep, stw2, error, dm_refine, h_refine);CHKERRQ(ierr);

    // INTEGRATION KERNEL
    auto const kernelL2x = [](auto& errorL2, const auto& error, const auto& h)
    {
    auto const kernel_pos = [&](auto const &pos)
    {
        auto M = getMatElemMass(h);
        auto ielem = cafes::fem::get_element(pos);
        for (std::size_t k=0; k<ielem.size(); ++k)
        {
            auto u1 = error.at(ielem[k]);
            for (std::size_t l=0; l<ielem.size(); ++l)
            {
                auto u2 = error.at(ielem[l]);
                //for (std::size_t d=0; d<dim; d++)
                //{
                    errorL2 += M[k][l]*u1[0]*u2[0];
                //}
            }
        }
    };
    return kernel_pos;
    };

    auto const kernelL2y = [](auto& errorL2, const auto& error, const auto& h)
    {
    auto const kernel_pos = [&](auto const &pos)
    {
        auto M = getMatElemMass(h);
        auto ielem = cafes::fem::get_element(pos);
        for (std::size_t k=0; k<ielem.size(); ++k)
        {
            auto u1 = error.at(ielem[k]);
            for (std::size_t l=0; l<ielem.size(); ++l)
            {
                auto u2 = error.at(ielem[l]);
                //for (std::size_t d=0; d<dim; d++)
                //{
                    errorL2 += M[k][l]*u1[1]*u2[1];
                //}
            }
        }
    };
    return kernel_pos;
    };

    auto const kernelH1x = [](auto& errorH1, const auto& error, const auto& h)
    {
    auto const kernel_pos = [&](auto const &pos)
    {
        auto M = getMatElemLaplacian(h);
        auto ielem = cafes::fem::get_element(pos);
        for (std::size_t k=0; k<ielem.size(); k++)
        {
            auto u1 = error.at(ielem[k]);
            for (std::size_t l=0; l<ielem.size(); l++)
            {
                auto u2 = error.at(ielem[l]);
                //for (std::size_t d=0; d<dim; d++)
                //{
                    errorH1 += M[k][l]*u1[0]*u2[0];
                    // if ( (l==0) and (k==0) )
                    // {
                    //     cout << axe << " " << u1[axe] << " " << u2[axe] << "\n";
                    // }
                //}
            }
        }
    };
    return kernel_pos;
    };

    auto const kernelH1y = [](auto& errorH1, const auto& error, const auto& h)
    {
    auto const kernel_pos = [&](auto const &pos)
    {
        auto M = getMatElemLaplacian(h);
        auto ielem = cafes::fem::get_element(pos);
        for (std::size_t k=0; k<ielem.size(); k++)
        {
            auto u1 = error.at(ielem[k]);
            for (std::size_t l=0; l<ielem.size(); l++)
            {
                auto u2 = error.at(ielem[l]);
                //for (std::size_t d=0; d<dim; d++)
                //{
                    errorH1 += M[k][l]*u1[1]*u2[1];
                    // if ( (l==0) and (k==0) )
                    // {
                    //     cout << axe << " " << u1[axe] << " " << u2[axe] << "\n";
                    // }
                //}
            }
        }
    };
    return kernel_pos;
    };

    // VELOCITY INTEGRATION
    double erroruL2x=0., erroruL2y=0., erroruH1x=0., erroruH1y = 0.;
    auto box = cafes::fem::get_DM_bounds<dim>(dav);
    auto errortest = cafes::petsc::petsc_vec<dim>(dm_refine, error, 0, true);
    //ierr = errortest.global_to_local(INSERT_VALUES);CHKERRQ(ierr);
    cafes::algorithm::iterate(box, kernelL2x(erroruL2x, errortest, h_refine));
    cafes::algorithm::iterate(box, kernelL2y(erroruL2y, errortest, h_refine));
    cafes::algorithm::iterate(box, kernelH1x(erroruH1x, errortest, h_refine));
    cafes::algorithm::iterate(box, kernelH1y(erroruH1y, errortest, h_refine));
    erroruL2x = std::sqrt(erroruL2x);
    erroruL2y = std::sqrt(erroruL2y);
    erroruH1x = std::sqrt(erroruH1x);
    erroruH1y = std::sqrt(erroruH1y);
    std::cout << "L2 error velocity : " << std::setprecision(10) << erroruL2x << " " << std::setprecision(10) << erroruL2y << std::endl;
    std::cout << "H1 error velocity : " << std::setprecision(10) << erroruH1x << " " << std::setprecision(10) << erroruH1y << std::endl;

    // PRESSURE INTEGRATION
    double errorpL2=0.;
    auto boxp = cafes::fem::get_DM_bounds<dim>(dap);
    auto errortestp = cafes::petsc::petsc_vec<dim>(dm_refine, error, 1, true);
    std::array<double, dim> hp_refine;
    hp_refine[0] = 2*h_refine[0];
    hp_refine[1] = 2*h_refine[1];
    cafes::algorithm::iterate(boxp, kernelL2x(errorpL2, errortestp, hp_refine));
    errorpL2 = std::sqrt(errorpL2);
    std::cout << "L2 error pressure : " << errorpL2 << std::endl;

    std::string stout3 = saverep+"L2errors_";
    std::string stmx = std::to_string(mx);
    stmx = std::string(3 - stmx.length(), '0').append(stmx);
    stout3.append(stmx);
    stout3.append(".txt");
    //const char * stw3 = stout3.c_str();
    std::ofstream ofs ;
    ofs.open(stout3, std::ofstream::out);
    ofs << st.ctx->h[0] << " " << erroruL2x << " " << erroruL2y << " " << erroruH1x << " " << erroruH1y << " " << errorpL2 <<  " " << s.kspiter << " " << s.kspresnorm << std::endl;
    ofs.close();

    // // OTHER WAY TO INTEGRATE
    // erroruL2 = 0;
    // for (std::size_t j=0; j< infov.my-1; j++)
    // {
    //     for (std::size_t i = 0; i < infov.mx-1; i++)
    //     {
    //         auto pos = cafes::geometry::position<int,dim>{i, j};
    //         auto ielem = cafes::fem::get_element(pos);
    //         for (std::size_t k=0; k<ielem.size(); k++)
    //         {
    //             auto u = erroru.at(ielem[k]);
    //             for (std::size_t d=0; d<dim; d++)
    //             {
    //                 erroruL2 += .25*u[d]*u[d]*h_refine[0]*h_refine[1];
    //             }
    //         }
            
    //     }
    // }
    // erroruL2 = std::sqrt(erroruL2);
    // cout << "L2 error velocity : " << erroruL2 << "\n";

    // ierr = cafes::io::read_solution_TXT("./Reference/Rsur5_nref25", solref, dmref, npoints); //256 cells in x and y in pressure
    
    
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}