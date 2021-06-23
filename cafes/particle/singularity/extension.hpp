// Copyright (c) 2016, Loic Gouarin <loic.gouarin@math.u-psud.fr>
// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#ifndef CAFES_PARTICLE_SINGULARITY_EXTENSION_HPP_INCLUDED
#define CAFES_PARTICLE_SINGULARITY_EXTENSION_HPP_INCLUDED

#include <particle/particle.hpp>
#include <particle/singularity/singularity.hpp>
#include <particle/singularity/UandPNormal.hpp>
#include <particle/geometry/position.hpp>
#include <petsc/vec.hpp>
#include <particle/singularity/truncation.hpp>

#include <petsc.h>
#include <iostream>
#include <sstream>
#include <cmath>

namespace cafes
{
  namespace singularity
  {

    #define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF) \
    template<typename... Args> \
    inline auto highLevelF(Args&&... args) -> decltype(lowLevelF(std::forward<Args>(args)...)) \
    { \
        return lowLevelF(std::forward<Args>(args)...); \
    }

    #undef __FUNCT__
    #define __FUNCT__ "apply_extension_velocity"
    template<typename bfunctype, typename gradtype, std::size_t Dimensions>
    PetscErrorCode apply_extension_velocity(petsc::petsc_vec<2>& sol,
                                     const std::vector<std::array<int, Dimensions>> ielem,
                                     const double coef,
                                     const bfunctype bfunc,
                                     const gradtype gradUsingExtended,
                                     const double psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        for (std::size_t je=0; je<bfunc.size(); ++je)
        {
            auto u = sol.at(ielem[je]);

            for (std::size_t d1=0; d1<Dimensions; ++d1)
            {
                for (std::size_t d2=0; d2<Dimensions; ++d2)
                    u[d1] -= coef*gradUsingExtended[d1][d2]*bfunc[je][d2];
                u[d1] += coef*psingExtended*bfunc[je][d1];
            }
        }
        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "apply_extension_pressure"
    template<typename bfunctype, typename gradtype, std::size_t Dimensions>
    PetscErrorCode apply_extension_pressure(petsc::petsc_vec<2>& sol,
                                     const std::vector<std::array<int, Dimensions>> ielem,
                                     const double coef,
                                     const bfunctype bfunc,
                                     const gradtype gradUsingExtended,
                                     const double psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        for (std::size_t je=0; je<bfunc.size(); ++je)
        {
            auto u = sol.at(ielem[je]);
            u[0] += coef*(gradUsingExtended[0][0] + gradUsingExtended[1][1])*bfunc[je];
            u[0] += coef*1.e-6*psingExtended*bfunc[je];
        }
        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_with_chix"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_with_chix(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        
        double L = p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = L/2.;
        double a = L-eps;

        auto Using = sing.get_u_sing(pts);
        auto gradUsing = sing.get_grad_u_sing(pts);
        auto psing = sing.get_p_sing(pts);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps*eps);
        auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], a*a, eps*eps);

        gradUsingExtended[0][0] = gradUsing[0][0]*chix + Using[0]*dxchix;
        gradUsingExtended[0][1] = gradUsing[0][1]*chix;
        gradUsingExtended[1][0] = gradUsing[1][0]*chix + Using[1]*dxchix;
        gradUsingExtended[1][1] = gradUsing[1][1]*chix;
        
        psingExtended = psing*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_field_extension_with_chix"
    template<typename Shape, std::size_t Dimensions, typename velocitytype>
    PetscErrorCode get_field_extension_with_chix(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     velocitytype& UsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        
        double L = p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = L/2.;
        double a = L-eps;
        std::cout << "L = " << L << " ; eps = " << eps << " ; a = " << a << std::endl;

        auto Using = sing.get_u_sing(pts);
        auto psing = sing.get_p_sing(pts);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps*eps);

        UsingExtended[0] = Using[0]*chix;
        UsingExtended[1] = Using[1]*chix;
        
        psingExtended = psing*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_with_chir"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_with_chir(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        
        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double dx_radius = (std::abs(pts[0]-p1.center_[0])<=1e-14) ? 0. : (pts[0] - p1.center_[0])/radius;
        double dy_radius = (std::abs(pts[1]-p1.center_[1])<=1e-14) ? 0. : (pts[1] - p1.center_[1])/radius;
        
        double eps = p1.shape_factors_[0]/4.;
        double a = p1.shape_factors_[0] - 2*eps;

        auto Using = sing.get_u_sing(pts);
        auto gradUsing = sing.get_grad_u_sing(pts);
        auto psing = sing.get_p_sing(pts);
        auto chir = 1.-cafes::singularity::chiTrunc(radius, a*a, eps*eps);
        auto drchir = -1.*cafes::singularity::dchiTrunc(radius, a*a, eps*eps);

        gradUsingExtended[0][0] = gradUsing[0][0]*chir + Using[0]*drchir*dx_radius;
        gradUsingExtended[0][1] = gradUsing[0][1]*chir + Using[0]*drchir*dy_radius;
        gradUsingExtended[1][0] = gradUsing[1][0]*chir + Using[1]*drchir*dx_radius;
        gradUsingExtended[1][1] = gradUsing[1][1]*chir + Using[1]*drchir*dy_radius;
        
        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_old_Babic_field_extension"
    template<typename Shape, std::size_t Dimensions, typename velocitytype>
    PetscErrorCode get_old_Babic_field_extension(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     velocitytype& UsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double eps = p1.shape_factors_[0]/2.01;
        double a = 2*p1.shape_factors_[0]/3.;

        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double theta = std::atan2(pts[1]-p1.center_[1], pts[0]-p1.center_[0]);

        geometry::position<double, Dimensions> sympoint1 {p1.center_[0] + (2*p1.shape_factors_[0]-radius)*std::cos(theta),p1.center_[1] + (2*p1.shape_factors_[0]-radius)*std::sin(theta)};
        geometry::position<double, Dimensions> sympoint2 {p1.center_[0] + (1.5*p1.shape_factors_[0]-0.5*radius)*std::cos(theta),p1.center_[1] + (1.5*p1.shape_factors_[0]-0.5*radius)*std::sin(theta)};

        auto Using1 = sing.get_u_sing(sympoint1); 
        auto Using2 = sing.get_u_sing(sympoint2);
        auto psing = -3*sing.get_p_sing(sympoint1) + 4*sing.get_p_sing(sympoint2);
        auto chir = 1.-cafes::singularity::chiTrunc(radius, a*a, eps*eps);

        UsingExtended[0] = (-3*Using1[0] + 4*Using2[0])*chir;
        UsingExtended[1] = (-3*Using1[1] + 4*Using2[1])*chir;  
        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_Babic_field_extension"
    template<typename Shape, std::size_t Dimensions, typename velocitytype>
    PetscErrorCode get_Babic_field_extension(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     velocitytype& UsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double eps = p1.shape_factors_[0]/3.;
        double a = 2*p1.shape_factors_[0]/3.;
        double Babic1 = -1. * (sing.contact_length_ + 2.*p1.shape_factors_[0]) / sing.contact_length_;
        double Babic2 =  2. * (sing.contact_length_ +    p1.shape_factors_[0]) / sing.contact_length_;

        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double theta = std::atan2(pts[1]-p1.center_[1], pts[0]-p1.center_[0]);

        double R1 = p1.shape_factors_[0] +    sing.contact_length_ -    sing.contact_length_*radius/p1.shape_factors_[0];
        double R2 = p1.shape_factors_[0] + .5*sing.contact_length_ - .5*sing.contact_length_*radius/p1.shape_factors_[0];

        geometry::position<double, Dimensions> sympoint1 {p1.center_[0] + R1 * std::cos(theta), 
                                                          p1.center_[1] + R1 * std::sin(theta)};
        geometry::position<double, Dimensions> sympoint2 {p1.center_[0] + R2 * std::cos(theta),
                                                          p1.center_[1] + R2 * std::sin(theta)};

        auto Using1 = sing.get_u_sing(sympoint1); 
        auto Using2 = sing.get_u_sing(sympoint2);
        auto psing = Babic1*sing.get_p_sing(sympoint1) + Babic2*sing.get_p_sing(sympoint2);
        auto chir = 1.-cafes::singularity::extchiTrunc(radius, a, eps);

        UsingExtended[0] = (Babic1*Using1[0] + Babic2*Using2[0])*chir;
        UsingExtended[1] = (Babic1*Using1[1] + Babic2*Using2[1])*chir;  
        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_old_Babic_singularity_extension"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_old_Babic_singularity_extension(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        
        double eps = p1.shape_factors_[0]/2.01;
        double a = 2*p1.shape_factors_[0]/3.;

        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double dx_radius = (std::abs(pts[0]-p1.center_[0])<=1e-6) ? 0. : (pts[0] - p1.center_[0])/radius;
        double dy_radius = (std::abs(pts[1]-p1.center_[1])<=1e-6) ? 0. : (pts[1] - p1.center_[1])/radius;
        
        double theta = std::atan2(pts[1]-p1.center_[1], pts[0]-p1.center_[0]);
        double dx_theta = (std::abs(pts[1]-p1.center_[1])<=1e-6) ? 0. : (pts[1] - p1.center_[1])/(radius*radius);
        double dy_theta = (std::abs(pts[0]-p1.center_[0])<=1e-6) ? 0. : (pts[0] - p1.center_[0])/(radius*radius);

        geometry::position<double, Dimensions> sympoint1 {p1.center_[0] + (  2*p1.shape_factors_[0]-    radius)*std::cos(theta),p1.center_[1] + (  2*p1.shape_factors_[0]-    radius)*std::sin(theta)};
        geometry::position<double, Dimensions> sympoint2 {p1.center_[0] + (1.5*p1.shape_factors_[0]-0.5*radius)*std::cos(theta),p1.center_[1] + (1.5*p1.shape_factors_[0]-0.5*radius)*std::sin(theta)};

        auto Using1 = sing.get_u_sing(sympoint1); 
        auto Using2 = sing.get_u_sing(sympoint2);
        auto gradUsing1 = sing.get_grad_u_sing(sympoint1);
        auto gradUsing2 = sing.get_grad_u_sing(sympoint2);
        auto psing = -3*sing.get_p_sing(sympoint1) + 4*sing.get_p_sing(sympoint2);
        auto chir = 1.-cafes::singularity::chiTrunc(radius, a*a, eps*eps);
        auto drchir = -1.*cafes::singularity::dchiTrunc(radius, a*a, eps*eps);

        // // Gradient of Using1 and Using2 in polar coordinates
        // gradtype polarGradUsing1, polarGradUsing2;
        // polarGradUsing1[0][0] = -std::cos(theta)*gradUsing1[0][0] - std::sin(theta)*gradUsing1[0][1];
        // polarGradUsing1[1][0] = -std::cos(theta)*gradUsing1[1][0] - std::sin(theta)*gradUsing1[1][1];
        // polarGradUsing1[0][1] = -(2*p1.shape_factors_[0]-radius)*std::sin(theta)*gradUsing1[0][0] + (2*p1.shape_factors_[0]-radius)*std::cos(theta)*gradUsing1[0][1];
        // polarGradUsing1[1][1] = -(2*p1.shape_factors_[0]-radius)*std::sin(theta)*gradUsing1[1][0] + (2*p1.shape_factors_[0]-radius)*std::cos(theta)*gradUsing1[1][1];

        // polarGradUsing2[0][0] = -.5*std::cos(theta)*gradUsing2[0][0] - .5*std::sin(theta)*gradUsing2[0][1];
        // polarGradUsing2[1][0] = -.5*std::cos(theta)*gradUsing2[1][0] - .5*std::sin(theta)*gradUsing2[1][1];
        // polarGradUsing2[0][1] = -(1.5*p1.shape_factors_[0]-.5*radius)*std::sin(theta)*gradUsing2[0][0] + (1.5*p1.shape_factors_[0]-.5*radius)*std::cos(theta)*gradUsing2[0][1];
        // polarGradUsing2[1][1] = -(1.5*p1.shape_factors_[0]-.5*radius)*std::sin(theta)*gradUsing2[1][0] + (1.5*p1.shape_factors_[0]-.5*radius)*std::cos(theta)*gradUsing2[1][1];

        // // Gradient of the Babic extension in Cartesian coordinates
        // gradtype BabicGradUsing;
        // BabicGradUsing[0][0] = dx_radius*(3.*polarGradUsing1[0][0] - 2.*polarGradUsing2[0][0]) + dx_theta*(-3.*polarGradUsing1[0][1] + 4.*polarGradUsing2[0][1]);
        // BabicGradUsing[1][0] = dx_radius*(3.*polarGradUsing1[1][0] - 2.*polarGradUsing2[1][0]) + dx_theta*(-3.*polarGradUsing1[1][1] + 4.*polarGradUsing2[1][1]);
        // BabicGradUsing[0][1] = dy_radius*(3.*polarGradUsing1[0][0] - 2.*polarGradUsing2[0][0]) + dy_theta*(-3.*polarGradUsing1[0][1] + 4.*polarGradUsing2[0][1]);
        // BabicGradUsing[1][1] = dy_radius*(3.*polarGradUsing1[1][0] - 2.*polarGradUsing2[1][0]) + dy_theta*(-3.*polarGradUsing1[1][1] + 4.*polarGradUsing2[1][1]);

        // // Gradient of (Babic extension times chir) in Cartesian coordinates
        // gradUsingExtended[0][0] = BabicGradUsing[0][0]*chir + (-3*Using1[0] + 4*Using2[0])*dx_radius*drchir;
        // gradUsingExtended[1][0] = BabicGradUsing[1][0]*chir + (-3*Using1[1] + 4*Using2[1])*dx_radius*drchir;
        // gradUsingExtended[0][1] = BabicGradUsing[0][1]*chir + (-3*Using1[0] + 4*Using2[0])*dy_radius*drchir;
        // gradUsingExtended[1][1] = BabicGradUsing[1][1]*chir + (-3*Using1[1] + 4*Using2[1])*dy_radius*drchir;

        gradUsingExtended[0][0] = ( -3*(-dx_radius*std::cos(theta)    - (  2*p1.shape_factors_[0] -    radius)*dx_theta*std::sin(theta))*gradUsing1[0][0]   \
                                    -3*(-dx_radius*std::sin(theta)    + (  2*p1.shape_factors_[0] -    radius)*dx_theta*std::cos(theta))*gradUsing1[0][1]   \
                                    +4*(-.5*dx_radius*std::cos(theta) - (1.5*p1.shape_factors_[0] - .5*radius)*dx_theta*std::sin(theta))*gradUsing2[0][0]   \
                                    +4*(-.5*dx_radius*std::sin(theta) + (1.5*p1.shape_factors_[0] - .5*radius)*dx_theta*std::cos(theta))*gradUsing2[0][1] ) \
                                  *chir \
                                  + (-3*Using1[0] + 4*Using2[0])*dx_radius*drchir;

        gradUsingExtended[1][0] = ( -3*(-dx_radius*std::cos(theta)    - (  2*p1.shape_factors_[0] -    radius)*dx_theta*std::sin(theta))*gradUsing1[1][0]   \
                                    -3*(-dx_radius*std::sin(theta)    + (  2*p1.shape_factors_[0] -    radius)*dx_theta*std::cos(theta))*gradUsing1[1][1]   \
                                    +4*(-.5*dx_radius*std::cos(theta) - (1.5*p1.shape_factors_[0] - .5*radius)*dx_theta*std::sin(theta))*gradUsing2[1][0]   \
                                    +4*(-.5*dx_radius*std::sin(theta) + (1.5*p1.shape_factors_[0] - .5*radius)*dx_theta*std::cos(theta))*gradUsing2[1][1] ) \
                                  *chir \
                                  + (-3*Using1[1] + 4*Using2[1])*dx_radius*drchir;

        gradUsingExtended[0][1] = ( -3*(-dy_radius*std::cos(theta)    - (  2*p1.shape_factors_[0] -    radius)*dy_theta*std::sin(theta))*gradUsing1[0][0]   \
                                    -3*(-dy_radius*std::sin(theta)    + (  2*p1.shape_factors_[0] -    radius)*dy_theta*std::cos(theta))*gradUsing1[0][1]   \
                                    +4*(-.5*dy_radius*std::cos(theta) - (1.5*p1.shape_factors_[0] - .5*radius)*dy_theta*std::sin(theta))*gradUsing2[0][0]   \
                                    +4*(-.5*dy_radius*std::sin(theta) + (1.5*p1.shape_factors_[0] - .5*radius)*dy_theta*std::cos(theta))*gradUsing2[0][1] ) \
                                  *chir \
                                  + (-3*Using1[0] + 4*Using2[0])*dy_radius*drchir;

        gradUsingExtended[1][1] = ( -3*(-dy_radius*std::cos(theta)    - (  2*p1.shape_factors_[0] -    radius)*dy_theta*std::sin(theta))*gradUsing1[1][0]   \
                                    -3*(-dy_radius*std::sin(theta)    + (  2*p1.shape_factors_[0] -    radius)*dy_theta*std::cos(theta))*gradUsing1[1][1]   \
                                    +4*(-.5*dy_radius*std::cos(theta) - (1.5*p1.shape_factors_[0] - .5*radius)*dy_theta*std::sin(theta))*gradUsing2[1][0]   \
                                    +4*(-.5*dy_radius*std::sin(theta) + (1.5*p1.shape_factors_[0] - .5*radius)*dy_theta*std::cos(theta))*gradUsing2[1][1] ) \
                                  *chir \
                                  + (-3*Using1[1] + 4*Using2[1])*dy_radius*drchir;

        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "lcontraction"
    double lcontraction(double r, double Radius, double contact_length, int l)
    {
        double value = Radius + contact_length*(1.-r/Radius)/l;
        return value;
    }

    #undef __FUNCT__
    #define __FUNCT__ "drlcontraction"
    double drlcontraction(double Radius, double contact_length, int l)
    {
        double value = -contact_length/(Radius*l);
        return value;
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_Babic_coefficients"
    std::vector<double> get_Babic_coefficients(int const& order)
    {
        std::vector<double> coeffs(order,0.);
        if (order==2) coeffs = {-9., 10.};
        else if (order==3) coeffs = {58.5, -260.,   202.5};
        return coeffs;
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_Babic_singularity_extension"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_Babic_singularity_extension(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        int BabicOrder = 3;
        
        double eps = p1.shape_factors_[0]/4.;
        double a = 3*p1.shape_factors_[0]/4.;
        
        auto BabicCoeffs = get_Babic_coefficients(BabicOrder);
        double Babic1 = BabicCoeffs[0];//-1. * (sing.contact_length_ + 2.*p1.shape_factors_[0]) / sing.contact_length_;
        double Babic2 =  BabicCoeffs[1];//2. * (sing.contact_length_ +    p1.shape_factors_[0]) / sing.contact_length_;
        // std::cout << "Babic: " << Babic1 << " " << Babic2 << std::endl; 

        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double dx_radius = (std::abs(pts[0]-p1.center_[0])<=1e-6) ? 0. : (pts[0] - p1.center_[0])/radius;
        double dy_radius = (std::abs(pts[1]-p1.center_[1])<=1e-6) ? 0. : (pts[1] - p1.center_[1])/radius;
        
        double theta = std::atan2(pts[1]-p1.center_[1], pts[0]-p1.center_[0]);
        double dx_theta = (std::abs(pts[1]-p1.center_[1])<=1e-6) ? 0. : (pts[1] - p1.center_[1])/(radius*radius);
        double dy_theta = (std::abs(pts[0]-p1.center_[0])<=1e-6) ? 0. : (pts[0] - p1.center_[0])/(radius*radius);

        auto chir = 1.-cafes::singularity::extchiTrunc(radius, a, eps);
        auto drchir = -1.*cafes::singularity::dextchiTrunc(radius, a, eps);

        // std::vector<double> R(BabicOrder,0.);
        // std::vector<double> drR(BabicOrder,0.);
        // std::vector<double> Using(BabicOrder,0.);
        // std::vector<double> gradUsing(BabicOrder,0.);
        gradUsingExtended[0][0] = 0.;
        gradUsingExtended[0][1] = 0.;
        gradUsingExtended[1][0] = 0.;
        gradUsingExtended[1][1] = 0.;
        double psing = 0;
        for (std::size_t i=0; i<BabicOrder; ++i)
        {
            auto R = lcontraction(radius, p1.shape_factors_[0], sing.contact_length_, i+1);
            auto drR = drlcontraction(p1.shape_factors_[0], sing.contact_length_, i+1);
            geometry::position<double, Dimensions> sympoint {p1.center_[0] + R * std::cos(theta), 
                                                             p1.center_[1] + R * std::sin(theta)};
            auto Using = sing.get_u_sing(sympoint);
            auto gradUsing = sing.get_grad_u_sing(sympoint);

            psing += BabicCoeffs[i]*sing.get_p_sing(sympoint);
            gradUsingExtended[0][0] += ( BabicCoeffs[i]*(drR*dx_radius*std::cos(theta) - R*dx_theta*std::sin(theta))*gradUsing[0][0] \
                                        +BabicCoeffs[i]*(drR*dx_radius*std::sin(theta) + R*dx_theta*std::cos(theta))*gradUsing[0][1] ) \
                                       *chir \
                                       + BabicCoeffs[i]*Using[0]*dx_radius*drchir;
            
            gradUsingExtended[1][0] += (  BabicCoeffs[i]*(drR*dx_radius*std::cos(theta) - R*dx_theta*std::sin(theta))*gradUsing[1][0]   \
                                         +BabicCoeffs[i]*(drR*dx_radius*std::sin(theta) + R*dx_theta*std::cos(theta))*gradUsing[1][1] ) \
                                       *chir \
                                       + BabicCoeffs[i]*Using[1]*dx_radius*drchir;

            gradUsingExtended[0][1] += (  BabicCoeffs[i]*(drR*dy_radius*std::cos(theta) - R*dy_theta*std::sin(theta))*gradUsing[0][0]   \
                                         +BabicCoeffs[i]*(drR*dy_radius*std::sin(theta) + R*dy_theta*std::cos(theta))*gradUsing[0][1] ) \
                                       *chir \
                                       + BabicCoeffs[i]*Using[0]*dy_radius*drchir;

            gradUsingExtended[1][1] += ( BabicCoeffs[i]*(drR*dy_radius*std::cos(theta) - R*dy_theta*std::sin(theta))*gradUsing[1][0]   \
                                       +BabicCoeffs[i]*(drR*dy_radius*std::sin(theta) + R*dy_theta*std::cos(theta))*gradUsing[1][1] ) \
                                      *chir \
                                      + BabicCoeffs[i]*Using[1]*dy_radius*drchir;
        }

        // double R1 = p1.shape_factors_[0] +    sing.contact_length_ -    sing.contact_length_*radius/p1.shape_factors_[0];
        // double R2 = p1.shape_factors_[0] + .5*sing.contact_length_ - .5*sing.contact_length_*radius/p1.shape_factors_[0];
        // double drR1 = -    sing.contact_length_ / p1.shape_factors_[0];
        // double drR2 = - .5*sing.contact_length_ / p1.shape_factors_[0];

        // geometry::position<double, Dimensions> sympoint1 {p1.center_[0] + R1 * std::cos(theta), 
        //                                                   p1.center_[1] + R1 * std::sin(theta)};
        // geometry::position<double, Dimensions> sympoint2 {p1.center_[0] + R2 * std::cos(theta),
        //                                                   p1.center_[1] + R2 * std::sin(theta)};

        // auto Using1 = sing.get_u_sing(sympoint1); 
        // auto Using2 = sing.get_u_sing(sympoint2);
        // auto gradUsing1 = sing.get_grad_u_sing(sympoint1);
        // auto gradUsing2 = sing.get_grad_u_sing(sympoint2);
        // auto psing = Babic1*sing.get_p_sing(sympoint1) + Babic2*sing.get_p_sing(sympoint2);
        

        // gradUsingExtended[0][0] = (Babic1*gradUsing1[0][0] + Babic2*gradUsing2[0][0])*chir;
        // gradUsingExtended[0][1] = (Babic1*gradUsing1[0][1] + Babic2*gradUsing2[0][1])*chir;
        // gradUsingExtended[1][0] = (Babic1*gradUsing1[1][0] + Babic2*gradUsing2[1][0])*chir;
        // gradUsingExtended[1][1] = (Babic1*gradUsing1[1][1] + Babic2*gradUsing2[1][1])*chir;

        // gradUsingExtended[0][0] = (  Babic1*(drR1*dx_radius*std::cos(theta) - R1*dx_theta*std::sin(theta))*gradUsing1[0][0]   \
        //                             +Babic1*(drR1*dx_radius*std::sin(theta) + R1*dx_theta*std::cos(theta))*gradUsing1[0][1]   \
        //                             +Babic2*(drR2*dx_radius*std::cos(theta) - R2*dx_theta*std::sin(theta))*gradUsing2[0][0]   \
        //                             +Babic2*(drR2*dx_radius*std::sin(theta) + R2*dx_theta*std::cos(theta))*gradUsing2[0][1] ) \
        //                           *chir \
        //                           + (Babic1*Using1[0] + Babic2*Using2[0])*dx_radius*drchir;

        // gradUsingExtended[1][0] = (  Babic1*(drR1*dx_radius*std::cos(theta) - R1*dx_theta*std::sin(theta))*gradUsing1[1][0]   \
        //                             +Babic1*(drR1*dx_radius*std::sin(theta) + R1*dx_theta*std::cos(theta))*gradUsing1[1][1]   \
        //                             +Babic2*(drR2*dx_radius*std::cos(theta) - R2*dx_theta*std::sin(theta))*gradUsing2[1][0]   \
        //                             +Babic2*(drR2*dx_radius*std::sin(theta) + R2*dx_theta*std::cos(theta))*gradUsing2[1][1] ) \
        //                           *chir \
        //                           + (Babic1*Using1[1] + Babic2*Using2[1])*dx_radius*drchir;

        // gradUsingExtended[0][1] = (  Babic1*(drR1*dy_radius*std::cos(theta) - R1*dy_theta*std::sin(theta))*gradUsing1[0][0]   \
        //                             +Babic1*(drR1*dy_radius*std::sin(theta) + R1*dy_theta*std::cos(theta))*gradUsing1[0][1]   \
        //                             +Babic2*(drR2*dy_radius*std::cos(theta) - R2*dy_theta*std::sin(theta))*gradUsing2[0][0]   \
        //                             +Babic2*(drR2*dy_radius*std::sin(theta) + R2*dy_theta*std::cos(theta))*gradUsing2[0][1] ) \
        //                           *chir \
        //                           + (Babic1*Using1[0] + Babic2*Using2[0])*dy_radius*drchir;

        // gradUsingExtended[1][1] = (  Babic1*(drR1*dy_radius*std::cos(theta) - R1*dy_theta*std::sin(theta))*gradUsing1[1][0]   \
        //                             +Babic1*(drR1*dy_radius*std::sin(theta) + R1*dy_theta*std::cos(theta))*gradUsing1[1][1]   \
        //                             +Babic2*(drR2*dy_radius*std::cos(theta) - R2*dy_theta*std::sin(theta))*gradUsing2[1][0]   \
        //                             +Babic2*(drR2*dy_radius*std::sin(theta) + R2*dy_theta*std::cos(theta))*gradUsing2[1][1] ) \
        //                           *chir \
        //                           + (Babic1*Using1[1] + Babic2*Using2[1])*dy_radius*drchir;

        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_with_chix_chir"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_with_chix_chir(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double Lx = p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double epsx = Lx/2.;
        double ax = Lx-epsx;
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], ax*ax, epsx*epsx);
        auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], ax*ax, epsx*epsx);
        
        double xx = ax;
        double sign = (pts[0]>=p1.center_[0]) ? 1. : -1.;
        double radius = std::sqrt( (pts[0]- (p1.center_[0] + sign*xx))*(pts[0]- (p1.center_[0] + sign*xx)) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        double dx_radius = (std::abs(pts[0]-(p1.center_[0] + sign*xx))<=1e-14) ? 0. : (pts[0] - (p1.center_[0] + sign*xx))/radius;
        double dy_radius = (std::abs(pts[1]-p1.center_[1])<=1e-14) ? 0. : (pts[1] - p1.center_[1])/radius;

        double epsr = (p1.shape_factors_[0]-xx)/2.;
        double ar = (p1.shape_factors_[0]-xx) - epsr;
        auto chir = 1.-cafes::singularity::chiTrunc(radius, ar*ar, epsr*epsr);
        auto drchir = -1.*cafes::singularity::dchiTrunc(radius, ar*ar, epsr*epsr);

        auto Using = sing.get_u_sing(pts);
        auto gradUsing = sing.get_grad_u_sing(pts);
        auto psing = sing.get_p_sing(pts);

        gradUsingExtended[0][0] = gradUsing[0][0]*chir*chix + Using[0]*drchir*dx_radius*chix + Using[0]*chir*dxchix;
        gradUsingExtended[0][1] = gradUsing[0][1]*chir*chix + Using[0]*drchir*dy_radius*chix;
        gradUsingExtended[1][0] = gradUsing[1][0]*chir*chix + Using[1]*drchir*dx_radius*chix + Using[1]*chir*dxchix;
        gradUsingExtended[1][1] = gradUsing[1][1]*chir*chix + Using[1]*drchir*dy_radius*chix;
        
        psingExtended = psing*chir*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_field_extension_with_chix_chir"
    template<typename Shape, std::size_t Dimensions, typename velocitytype>
    PetscErrorCode get_field_extension_with_chix_chir(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     velocitytype& UsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double Lx = p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double epsx = Lx/2.;
        double ax = Lx-epsx;
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], ax*ax, epsx*epsx);
        
        double xx = ax;
        double sign = (pts[0]>=p1.center_[0]) ? 1. : -1.;
        double radius = std::sqrt( (pts[0]- (p1.center_[0] + sign*xx))*(pts[0]- (p1.center_[0] + sign*xx)) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        // double dx_radius = (std::abs(pts[0]-(p1.center_[0] + sign*xx))<=1e-14) ? 0. : (pts[0] - (p1.center_[0] + sign*xx))/radius;
        // double dy_radius = (std::abs(pts[1]-p1.center_[1])<=1e-14) ? 0. : (pts[1] - p1.center_[1])/radius;

        double epsr = (p1.shape_factors_[0]-xx)/2.;
        double ar = (p1.shape_factors_[0]-xx) - epsr;
        auto chir = 1.-cafes::singularity::chiTrunc(radius, ar*ar, epsr*epsr);

        auto Using = sing.get_u_sing(pts);
        auto psing = sing.get_p_sing(pts);

        UsingExtended[0] = Using[0]*chir*chix;
        UsingExtended[1] = Using[1]*chir*chix;  
        psingExtended = psing*chir*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_force_with_chix"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_force_with_chix(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double a = .5*p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = a*a;

        // auto Using = sing.get_u_sing(pts);
        auto gradUsing = sing.get_grad_u_sing(pts);
        auto psing = sing.get_p_sing(pts);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps);
        // auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], a*a, eps);

        gradUsingExtended[0][0] = gradUsing[0][0]*chix;// + Using[0]*dxchix;
        gradUsingExtended[0][1] = gradUsing[0][1]*chix;
        gradUsingExtended[1][0] = gradUsing[1][0]*chix;// + Using[1]*dxchix;
        gradUsingExtended[1][1] = gradUsing[1][1]*chix;
        
        psingExtended = psing*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_from_border_in_y"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_from_border_in_y(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double sign = (pts[0] >= p1.center_[0]) ? 1. : -1.;
        geometry::position<double, 2> pts_border = {p1.center_[0] + sign * std::sqrt(p1.shape_factors_[0]*p1.shape_factors_[0] - (pts[1] - p1.center_[1])*(pts[1] - p1.center_[1])), pts[1]};
        double dy_pts_border_X = (std::abs(pts[1]-p1.center_[1])<=1.e-14) ? 0. : -1.*sign*(pts[1]-p1.center_[1])/std::sqrt(p1.shape_factors_[0]*p1.shape_factors_[0] - (pts[1] - p1.center_[1])*(pts[1] - p1.center_[1]));
        

        double a = .5*p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = a*a;

        auto Using = sing.get_u_sing(pts_border);
        auto gradUsing = sing.get_grad_u_sing(pts_border);
        auto psing = sing.get_p_sing(pts_border);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps);
        auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], a*a, eps);

        gradUsingExtended[0][0] = Using[0]*dxchix;
        gradUsingExtended[0][1] = (gradUsing[0][0]*dy_pts_border_X + gradUsing[0][1])*chix;
        gradUsingExtended[1][0] = Using[1]*dxchix;
        gradUsingExtended[1][1] = (gradUsing[1][0]*dy_pts_border_X + gradUsing[1][1])*chix;
        
        psingExtended = psing*chix;

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_from_border_in_radius"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_from_border_in_radius(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double theta = (abs(pts[0]-p1.center_[0])<1e-14 && abs(pts[1]-p1.center_[1])<1e-14) ? 0. : std::atan2(pts[1]-p1.center_[1],pts[0]-p1.center_[0]);
        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        geometry::position<double, 2> pts_border = { p1.center_[0] + p1.shape_factors_[0]*std::cos(theta), p1.center_[1] + p1.shape_factors_[0]*std::sin(theta) };

        double dx_radius = (radius <= 1e-14) ? 0. : (pts[0]-p1.center_[0])/radius;
        double dy_radius = (radius <= 1e-14) ? 0. : (pts[1]-p1.center_[1])/radius;

        double dtheta_pts_border_X = -p1.shape_factors_[0]*std::sin(theta);
        double dtheta_pts_border_Y = p1.shape_factors_[0]*std::cos(theta);

        auto Using = sing.get_u_sing(pts_border);
        auto gradUsing = sing.get_grad_u_sing(pts_border);
        auto psing = sing.get_p_sing(pts_border);

        gradUsingExtended[0][0] = (Using[0]*dx_radius - dy_radius*(gradUsing[0][0]*dtheta_pts_border_X + gradUsing[0][1]*dtheta_pts_border_Y))/p1.shape_factors_[0];
        gradUsingExtended[0][1] = (Using[0]*dy_radius + dx_radius*(gradUsing[0][0]*dtheta_pts_border_X + gradUsing[0][1]*dtheta_pts_border_Y))/p1.shape_factors_[0];
        gradUsingExtended[1][0] = (Using[1]*dx_radius - dy_radius*(gradUsing[1][0]*dtheta_pts_border_X + gradUsing[1][1]*dtheta_pts_border_Y))/p1.shape_factors_[0];
        gradUsingExtended[1][1] = (Using[1]*dy_radius + dx_radius*(gradUsing[1][0]*dtheta_pts_border_X + gradUsing[1][1]*dtheta_pts_border_Y))/p1.shape_factors_[0];
        
        psingExtended = psing*radius/p1.shape_factors_[0];

        PetscFunctionReturn(0);
    }

    #undef __FUNCT__
    #define __FUNCT__ "get_singularity_extension_force_from_border_in_radius"
    template<typename Shape, std::size_t Dimensions, typename gradtype>
    PetscErrorCode get_singularity_extension_force_from_border_in_radius(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     geometry::position<double, Dimensions> pts,
                                     gradtype& gradUsingExtended,
                                     double& psingExtended)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;

        double theta = (abs(pts[0]-p1.center_[0])<1e-14 && abs(pts[1]-p1.center_[1])<1e-14) ? 0. : std::atan2(pts[1]-p1.center_[1],pts[0]-p1.center_[0]);
        double radius = std::sqrt( (pts[0]-p1.center_[0])*(pts[0]-p1.center_[0]) + (pts[1]-p1.center_[1])*(pts[1]-p1.center_[1]) );
        // geometry::position<double, 2> pts_border = { p1.center_[0] + p1.shape_factors_[0]*std::cos(theta), p1.center_[1] + p1.shape_factors_[0]*std::sin(theta) };

        auto gradUsing = sing.get_grad_u_sing(pts);
        auto psing = sing.get_p_sing(pts);

        double a = .5*p1.shape_factors_[0];
        double eps = a/2;
        auto chir = 1.-cafes::singularity::chiTrunc(radius, a*a, eps*eps);

        gradUsingExtended[0][0] = gradUsing[0][0]*chir;
        gradUsingExtended[0][1] = gradUsing[0][1]*chir;
        gradUsingExtended[1][0] = gradUsing[1][0]*chir;
        gradUsingExtended[1][1] = gradUsing[1][1]*chir;
        
        psingExtended = psing*chir;

        PetscFunctionReturn(0);
    }

    ALIAS_TEMPLATE_FUNCTION(get_singularity_extension, get_Babic_singularity_extension);

    #undef __FUNCT__
    #define __FUNCT__ "extension_in_particle"
    template<typename Shape, std::size_t Dimensions>
    PetscErrorCode extension_in_particle(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     petsc::petsc_vec<2>& sol,
                                     geometry::position<double, Dimensions> pts,
                                     std::vector<std::array<int, 2>> ielem,
                                     std::array<double, 2> const& h, std::array<double, 2> const& hs, std::size_t is, std::size_t js,
                                     double coef)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        
        geometry::position<double, 2> pts_loc = {is*hs[0], js*hs[1]};
        auto bfunc = fem::P2_integration_grad(pts_loc, h);

        std::array< std::array<double, Dimensions>, Dimensions > gradUsingExtended;
        double psingExtended;
        get_singularity_extension(sing, p1, pts, gradUsingExtended, psingExtended);

        apply_extension_velocity(sol, ielem, coef, bfunc, gradUsingExtended, psingExtended);
        
        PetscFunctionReturn(0);
    } // END EXTENSION_WITH_CHIX FUNCTION FOR VELOCITY

    #undef __FUNCT__
    #define __FUNCT__ "extension_in_particle"
    template<typename Shape, std::size_t Dimensions>
    PetscErrorCode extension_in_particle(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     petsc::petsc_vec<2>& sol,
                                     geometry::position<double, Dimensions> pts,
                                     std::vector<std::array<int, 2>> ielem,
                                     std::array<double, 2> const& h, std::array<double, 2> const& hs, std::size_t is, std::size_t js,
                                     double coef,
                                     std::string type)
    {
        geometry::position<double, 2> pts_loc = {is*hs[0], js*hs[1]};
        auto bfunc = fem::P1_integration_sing(pts_loc, h);

        std::array< std::array<double, Dimensions>, Dimensions > gradUsingExtended;
        double psingExtended;
        get_singularity_extension(sing, p1, pts, gradUsingExtended, psingExtended);

        apply_extension_pressure(sol, ielem, coef, bfunc, gradUsingExtended, psingExtended);
        
        PetscFunctionReturn(0);
    } // END EXTENSION_WITH_CHIX FUNCTION FOR PRESSURE

    #undef __FUNCT__
    #define __FUNCT__ "extension_from_border_in_y"
    template<typename Shape, std::size_t Dimensions>
    PetscErrorCode extension_from_border_in_y(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     petsc::petsc_vec<2>& sol,
                                     geometry::position<double, Dimensions> pts,
                                     std::vector<std::array<int, 2>> ielem,
                                     std::array<double, 2> const& h, std::array<double, 2> const& hs, std::size_t is, std::size_t js,
                                     double coef)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        geometry::position<double, 2> pts_polar = {sqrt( (pts[0]- p1.center_[0])*(pts[0]- p1.center_[0]) + (pts[1]- p1.center_[1])*(pts[1]- p1.center_[1]) ), atan2(pts[1]- p1.center_[1], pts[0]- p1.center_[0])};
        geometry::position<double, 2> pts_loc = {is*hs[0], js*hs[1]};
        auto bfunc = fem::P2_integration_grad(pts_loc, h);
        double a = .5*p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = a*a;

        double theta = std::asin(pts[1]/p1.shape_factors_[0]);
        geometry::position<double, 2> pts_border = {p1.shape_factors_[0]*std::cos(theta), pts[1]};
        double dtheta_pts_border_X = -1.*p1.shape_factors_[0]*std::sin(theta);
        double dy_theta = 1./(p1.shape_factors_[0]*std::sqrt(1-pts[0]*pts[0]/(p1.shape_factors_[0]*p1.shape_factors_[0])));

        auto Using = sing.get_u_sing(pts_border);
        auto gradUsing = sing.get_grad_u_sing(pts_border);
        auto psing = sing.get_p_sing(pts_border);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps);
        auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], a*a, eps);

        std::array< std::array<double, Dimensions>, Dimensions > gradUsingExtended;
        gradUsingExtended[0][0] = Using[0]*dxchix;
        gradUsingExtended[0][1] = (gradUsing[0][0]*dtheta_pts_border_X*dy_theta + gradUsing[0][1])*chix;
        gradUsingExtended[1][0] = Using[1]*dxchix;
        gradUsingExtended[1][1] = (gradUsing[1][0]*dtheta_pts_border_X*dy_theta + gradUsing[1][1])*chix;
        
        for (std::size_t je=0; je<bfunc.size(); ++je)
        {
            auto u = sol.at(ielem[je]);

            for (std::size_t d1=0; d1<Dimensions; ++d1)
            {
                for (std::size_t d2=0; d2<Dimensions; ++d2)
                    u[d1] -= coef*gradUsingExtended[d1][d2]*bfunc[je][d2];
                u[d1] += coef*psing*bfunc[je][d1]*chix;
            }
        }
        PetscFunctionReturn(0);
    } // END EXTENSION_FROM_BORDER_IN_Y FUNCTION FOR VELOCITY

    #undef __FUNCT__
    #define __FUNCT__ "extension_from_border_in_y"
    template<typename Shape, std::size_t Dimensions>
    PetscErrorCode extension_from_border_in_y(singularity<Shape, 2> sing, 
                                     particle<Shape> const& p1,
                                     petsc::petsc_vec<2>& sol,
                                     geometry::position<double, Dimensions> pts,
                                     std::vector<std::array<int, 2>> ielem,
                                     std::array<double, 2> const& h, std::array<double, 2> const& hs, std::size_t is, std::size_t js,
                                     double coef,
                                     std::string type)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        geometry::position<double, 2> pts_polar = {sqrt( (pts[0]- p1.center_[0])*(pts[0]- p1.center_[0]) + (pts[1]- p1.center_[1])*(pts[1]- p1.center_[1]) ), atan2(pts[1]- p1.center_[1], pts[0]- p1.center_[0])};
        geometry::position<double, 2> pts_loc = {is*hs[0], js*hs[1]};
        auto bfunc = fem::P1_integration_sing(pts_loc, h);
        double a = .5*p1.shape_factors_[0]*cos(asin(sing.cutoff_dist_/p1.shape_factors_[0]));
        double eps = a*a;

        double theta = std::asin(pts[1]/p1.shape_factors_[0]);
        geometry::position<double, 2> pts_border = {p1.shape_factors_[0]*std::cos(theta), pts[1]};
        double dtheta_pts_border_X = -1.*p1.shape_factors_[0]*std::sin(theta);
        double dy_theta = 1./(p1.shape_factors_[0]*std::sqrt(1-pts[0]*pts[0]/(p1.shape_factors_[0]*p1.shape_factors_[0])));

        auto Using = sing.get_u_sing(pts_border);
        auto gradUsing = sing.get_grad_u_sing(pts_border);
        auto psing = sing.get_p_sing(pts_border);
        auto chix = 1.-cafes::singularity::chiTrunc(pts[0]-p1.center_[0], a*a, eps);
        auto dxchix = -1.*cafes::singularity::dchiTrunc(pts[0]-p1.center_[0], a*a, eps);

        std::array< std::array<double, Dimensions>, Dimensions > gradUsingExtended;
        gradUsingExtended[0][0] = Using[0]*dxchix;
        gradUsingExtended[0][1] = (gradUsing[0][0]*dtheta_pts_border_X*dy_theta + gradUsing[0][1])*chix;
        gradUsingExtended[1][0] = Using[1]*dxchix;
        gradUsingExtended[1][1] = (gradUsing[1][0]*dtheta_pts_border_X*dy_theta + gradUsing[1][1])*chix;
        
        for (std::size_t je=0; je<bfunc.size(); ++je)
        {
            auto u = sol.at(ielem[je]);
            u[0] += coef*(gradUsingExtended[0][0] + gradUsingExtended[1][1])*bfunc[je];
            u[0] += coef*1.e-6*psing*bfunc[je]*chix;

        }
        PetscFunctionReturn(0);
    } // END EXTENSION_FROM_BORDER_IN_Y FUNCTION FOR PRESSURE

  } // END SINGULARITY NAMESPACE
} // END CAFES NAMESPACE

#endif
