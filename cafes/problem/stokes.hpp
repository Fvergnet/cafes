#ifndef PARTICLE_PROBLEM_STOKES_HPP_INCLUDED
#define PARTICLE_PROBLEM_STOKES_HPP_INCLUDED

#include <problem/context.hpp>
#include <problem/options.hpp>
#include <fem/matrixFree.hpp>
#include <fem/rhs.hpp>
#include <fem/mesh.hpp>
#include <petsc.h>
#include <iostream>
#include <algorithm>

namespace cafes
{
  namespace problem
  {

    #undef __FUNCT__
    #define __FUNCT__ "setPMMSolver"
    PetscErrorCode setPMMSolver(KSP ksp, DM dm){
      PetscErrorCode ierr;
      PetscFunctionBeginUser;
      PC pc, pc_i;
      KSP *sub_ksp;
      PetscInt MGlevels;
      DM dav;
      DMDALocalInfo  info;

      PetscFunctionBeginUser;

      ierr = DMCompositeGetEntries(dm, &dav, nullptr);CHKERRQ(ierr);
      ierr = DMDAGetLocalInfo(dav, &info);CHKERRQ(ierr);

      ierr = KSPSetType(ksp, KSPGCR);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PCSetType(pc, PCFIELDSPLIT);CHKERRQ(ierr);
      ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
      ierr = PCFieldSplitSetOffDiagUseAmat(pc, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER);CHKERRQ(ierr);
      ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, PETSC_NULL);CHKERRQ(ierr);
      ierr = KSPSetUp(ksp);CHKERRQ(ierr);

      ierr = PCFieldSplitGetSubKSP(pc, nullptr, &sub_ksp);CHKERRQ(ierr);

      /* Set MG solver on velocity field*/
      ierr = KSPGetPC(sub_ksp[0], &pc_i);CHKERRQ(ierr);
      // ierr = KSPSetDM(sub_ksp[0], dav);CHKERRQ(ierr);
      // ierr = KSPSetDMActive(sub_ksp[0], PETSC_FALSE);CHKERRQ(ierr);

      ierr = KSPSetType(sub_ksp[0], KSPCG);CHKERRQ(ierr);
      ierr = KSPSetTolerances(sub_ksp[0], 1e-2, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = PCSetType(pc_i, PCMG);CHKERRQ(ierr);

      auto i = (info.mx<info.my)? info.mx: info.my;
      i = (info.mz == 1 || i<info.mz)? i: info.mz;

      MGlevels = 1;
      while(i > 8){
        i >>= 1;
        MGlevels++;
      }

      ierr = PCMGSetLevels(pc_i, MGlevels, PETSC_NULL);CHKERRQ(ierr);

      /* Set Jacobi preconditionner on pressure field*/
      ierr = KSPSetType(sub_ksp[1], KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPSetTolerances(sub_ksp[1], 1e-1, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPGetPC(sub_ksp[1], &pc_i);CHKERRQ(ierr);
      ierr = PCSetType(pc_i, PCJACOBI);CHKERRQ(ierr);


      PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "createLevelMatrices"
    template<typename CTX>
    PetscErrorCode createLevelMatrices(KSP ksp, Mat Alevel, Mat Plevel, void *ctx)
    {
      DM             dm;
      PetscErrorCode ierr;
      CTX *ctx_ = (CTX*) ctx;
      int localsize, totalsize;
      PetscFunctionBeginUser;

      ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
      ctx_->dm = dm;

      ierr = fem::get_DM_sizes(dm, localsize, totalsize);CHKERRQ(ierr);

      ierr = MatSetSizes(Alevel, localsize, localsize, totalsize, totalsize);CHKERRQ(ierr);
      ierr = MatSetType(Alevel, MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetContext(Alevel, ctx);CHKERRQ(ierr);
      ierr = MatShellSetOperation(Alevel, MATOP_MULT, (void(*)())fem::diag_block_matrix<CTX>);CHKERRQ(ierr);
      ierr = MatShellSetOperation(Alevel, MATOP_GET_DIAGONAL, (void(*)())fem::diag_block_matrix_diag<CTX>);CHKERRQ(ierr);
      ierr = MatSetDM(Alevel, dm);CHKERRQ(ierr);
      ierr = MatSetFromOptions(Alevel);CHKERRQ(ierr);

      PetscFunctionReturn(0);
    }

    template<std::size_t Dimensions>
    struct stokes{
      using Ctx = context<Dimensions>;
      fem::rhs_conditions<Dimensions> rhsc_;
      options<Dimensions> opt{};
      Ctx *ctx;
      Vec sol;
      Vec rhs;
      Mat A;
      Mat P;
      KSP ksp;

      stokes(fem::dirichlet_conditions<Dimensions> bc, fem::rhs_conditions<Dimensions> rhsc={nullptr}){
        opt.process_options();

        DM mesh;
        fem::createMesh<Dimensions>(mesh, opt.mx, opt.xperiod);
        DMCreateGlobalVector(mesh, &sol);
        VecDuplicate(sol, &rhs);
        VecSet(rhs, 0.);

        std::array<double, Dimensions> hu;
        std::array<double, Dimensions> hp;
        for(std::size_t i = 0; i<Dimensions; ++i){
          hp[i] = opt.lx[i]/(opt.mx[i]-1);
          hu[i] = .5*hp[i];
        }

        // set Stokes matrix
        PetscErrorCode(*method)(DM, Vec, Vec, std::array<double, Dimensions> const&);
        if (opt.strain_tensor)
          method = fem::strain_tensor_mult;
        else
          method = fem::laplacian_mult;

        // fix this to avoid raw pointer !!
        rhsc_ = rhsc;
        ctx = new Ctx{mesh, hu, method};
        ctx->set_dirichlet_bc(bc);
        A = fem::make_matrix<Ctx>(ctx, fem::stokes_matrix<Ctx>);
        MatSetDM(A, mesh);
        MatSetFromOptions(A);

        // set preconditionner of Stokes matrix
        DM dav, dap;
        DMCompositeGetEntries(mesh, &dav, &dap);
        DMSetMatType(dav, MATSHELL);
        DMSetMatType(dap, MATSHELL);

        Ctx *slap = new Ctx{dav, hu, fem::laplacian_mult, fem::laplacian_mult_diag};
        slap->set_dirichlet_bc(bc);
        auto A11 = fem::make_matrix<Ctx>(slap);

        Ctx *smass = new Ctx{dap, hp, fem::mass_mult, fem::mass_mult_diag};
        auto A22 = fem::make_matrix<Ctx>(smass);
        
        Mat bA[2][2];
        bA[0][0] = A11; bA[0][1] = PETSC_NULL;
        bA[1][0] = PETSC_NULL; bA[1][1] = A22;
        MatCreateNest(PETSC_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL, &bA[0][0], &P);
        MatSetDM(P, ctx->dm);
        MatSetFromOptions(P);

      }

      #undef __FUNCT__
      #define __FUNCT__ "setup_RHS"
      PetscErrorCode setup_RHS(){
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        if (rhsc_.has_condition()){
          ierr = fem::set_rhs<Dimensions, 1>(ctx->dm, rhs, rhsc_, ctx->h);CHKERRQ(ierr);
        }
        
        DM dav;
        Vec rhsv;
        ierr = DMCompositeGetEntries(ctx->dm, &dav, nullptr);CHKERRQ(ierr);
        ierr = DMCompositeGetAccess(ctx->dm, rhs, &rhsv, nullptr);CHKERRQ(ierr);
        ierr = SetDirichletOnRHS(dav, ctx->bc_, rhsv, ctx->h);CHKERRQ(ierr);
        ierr = DMCompositeRestoreAccess(ctx->dm, rhs, &rhsv, nullptr);CHKERRQ(ierr);
        
        PetscFunctionReturn(0);
      }

      #undef __FUNCT__
      #define __FUNCT__ "setup_KSP"
      PetscErrorCode setup_KSP()
      {
        PetscErrorCode ierr;
        PC pc;
        PetscFunctionBeginUser;

        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
        ierr = KSPSetDM(ksp, ctx->dm);CHKERRQ(ierr);
        ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp, "stokes_");CHKERRQ(ierr);

        ierr = KSPSetOperators(ksp, A, P);CHKERRQ(ierr);

        if (opt.pmm){
          ierr = setPMMSolver(ksp, ctx->dm);CHKERRQ(ierr);
        }

        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ierr = KSPSetUp(ksp);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);

        PetscBool same;
        ierr = PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &same);

        // If we use fieldsplit for the preconditionner
        if (same) {
          KSP *subksp;
          DM dav, dap;
          ierr = PCFieldSplitGetSubKSP(pc, nullptr, &subksp);CHKERRQ(ierr);
          ierr = DMCompositeGetEntries(ctx->dm, &dav, &dap);CHKERRQ(ierr);

          ierr = KSPGetPC(subksp[0], &pc);CHKERRQ(ierr);
          ierr = KSPSetDM(subksp[0], dav);CHKERRQ(ierr);
          ierr = KSPSetDMActive(subksp[0], PETSC_FALSE);CHKERRQ(ierr);
          ierr = PetscObjectTypeCompare((PetscObject)pc, PCMG, &same);CHKERRQ(ierr);

          // if MG is set for fieldsplit_0
          if (same) {
            PetscInt MGlevels;
            KSP smoother;
            ierr = PCMGGetLevels(pc, &MGlevels);CHKERRQ(ierr);

            for(std::size_t i=0; i<MGlevels; ++i){
              auto mg_h{ctx->h};
              std::for_each(mg_h.begin(), mg_h.end(), [&](auto& x){x*=(1<<(MGlevels-1-i));});

              auto *mg_ctx = new Ctx{dav, mg_h, fem::laplacian_mult, fem::laplacian_mult_diag};
              mg_ctx->set_dirichlet_bc(ctx->bc_);

              PC pcsmoother;
              ierr = PCMGGetSmoother(pc, i, &smoother);CHKERRQ(ierr);

              ierr = KSPSetComputeOperators(smoother, createLevelMatrices<Ctx>, (void *) mg_ctx);CHKERRQ(ierr);
              ierr = KSPSetType(smoother, KSPCG);CHKERRQ(ierr);
              ierr = KSPGetPC(smoother, &pcsmoother);CHKERRQ(ierr);
              ierr = PCSetType(pcsmoother, PCJACOBI);CHKERRQ(ierr);
            }
  
            ierr = PCSetUp(pc);CHKERRQ(ierr);
          }
        }
        PetscFunctionReturn(0);
      }

      #undef __FUNCT__
      #define __FUNCT__ "solve"
      PetscErrorCode solve()
      {
        PetscErrorCode ierr;
        PC pc;
        PetscFunctionBeginUser;

        ierr = KSPSolve(ksp, rhs, sol);CHKERRQ(ierr);

        PetscFunctionReturn(0);
      }
    };

  } 
}

#endif