// Copyright (c) 2016, Loic Gouarin <loic.gouarin@math.u-psud.fr>
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef CAFES_FEM_MATELEM_HPP_INCLUDED
#define CAFES_FEM_MATELEM_HPP_INCLUDED

#include <array>
#include <petsc.h>

using array_2d = std::array<std::array<double, 4>, 4>;
using array_2d_p = std::array<std::array<double, 9>, 4>;

using array_3d = std::array<std::array<double, 8>, 8>;
using array_3d_p = std::array<std::array<double, 27>, 8>;

/* Elementary matrices for the 2D problem */

/* \int dx(\phi_i) dx(\phi_j) */
constexpr array_2d matElem2d_dxudxu{{{{1. / 3, -1. / 3, 1. / 6, -1. / 6}},
                                     {{-1. / 3, 1. / 3, -1. / 6, 1. / 6}},
                                     {{1. / 6, -1. / 6, 1. / 3, -1. / 3}},
                                     {{-1. / 6, 1. / 6, -1. / 3, 1. / 3}}}};

/* \int dy(\phi_i) dy(\phi_j) */
constexpr array_2d matElem2d_dyudyu{{{{1. / 3, 1. / 6, -1. / 3, -1. / 6}},
                                     {{1. / 6, 1. / 3, -1. / 6, -1. / 3}},
                                     {{-1. / 3, -1. / 6, 1. / 3, 1. / 6}},
                                     {{-1. / 6, -1. / 3, 1. / 6, 1. / 3}}}};

/* \int dx(\phi_i) dy(\phi_j) */
constexpr array_2d matElem2d_dxudyu{{{{0.25, -0.25, 0.25, -0.25}},
                                     {{0.25, -0.25, 0.25, -0.25}},
                                     {{-0.25, 0.25, -0.25, 0.25}},
                                     {{-0.25, 0.25, -0.25, 0.25}}}};

/* \int phi_i \phi_j */
constexpr array_2d matElemMass2d{{{{1. / 9, 1. / 18, 1. / 18, 1. / 36}},
                                  {{1. / 18, 1. / 9, 1. / 36, 1. / 18}},
                                  {{1. / 18, 1. / 36, 1. / 9, 1. / 18}},
                                  {{1. / 36, 1. / 18, 1. / 18, 1. / 9}}}};

constexpr array_2d_p matElem2d_pdxu{
    {{{15. / 48, -10. / 48, -5. / 48, 18. / 48, -12. / 48, -6. / 48, 3. / 48,
       -2. / 48, -1. / 48}},
     {{5. / 48, 10. / 48, -15. / 48, 6. / 48, 12. / 48, -18. / 48, 1. / 48,
       2. / 48, -3. / 48}},
     {{3. / 48, -2. / 48, -1. / 48, 18. / 48, -12. / 48, -6. / 48, 15. / 48,
       -10. / 48, -5. / 48}},
     {{1. / 48, 2. / 48, -3. / 48, 6. / 48, 12. / 48, -18. / 48, 5. / 48,
       10. / 48, -15. / 48}}}};

constexpr array_2d_p matElem2d_pdyu{
    {{{15. / 48, 18. / 48, 3. / 48, -10. / 48, -12. / 48, -2. / 48, -5. / 48,
       -6. / 48, -1. / 48}},
     {{3. / 48, 18. / 48, 15. / 48, -2. / 48, -12. / 48, -10. / 48, -1. / 48,
       -6. / 48, -5. / 48}},
     {{5. / 48, 6. / 48, 1. / 48, 10. / 48, 12. / 48, 2. / 48, -15. / 48,
       -18. / 48, -3. / 48}},
     {{1. / 48, 6. / 48, 5. / 48, 2. / 48, 12. / 48, 10. / 48, -3. / 48,
       -18. / 48, -15. / 48}}}};

// Elementary matrices for the 3D problem
constexpr array_3d matElem3d_dxudxu{{{{1. / 9, -1. / 9, 1. / 18, -1. / 18,
                                       1. / 18, -1. / 18, 1. / 36, -1. / 36}},
                                     {{-1. / 9, 1. / 9, -1. / 18, 1. / 18,
                                       -1. / 18, 1. / 18, -1. / 36, 1. / 36}},
                                     {{1. / 18, -1. / 18, 1. / 9, -1. / 9,
                                       1. / 36, -1. / 36, 1. / 18, -1. / 18}},
                                     {{-1. / 18, 1. / 18, -1. / 9, 1. / 9,
                                       -1. / 36, 1. / 36, -1. / 18, 1. / 18}},
                                     {{1. / 18, -1. / 18, 1. / 36, -1. / 36,
                                       1. / 9, -1. / 9, 1. / 18, -1. / 18}},
                                     {{-1. / 18, 1. / 18, -1. / 36, 1. / 36,
                                       -1. / 9, 1. / 9, -1. / 18, 1. / 18}},
                                     {{1. / 36, -1. / 36, 1. / 18, -1. / 18,
                                       1. / 18, -1. / 18, 1. / 9, -1. / 9}},
                                     {{-1. / 36, 1. / 36, -1. / 18, 1. / 18,
                                       -1. / 18, 1. / 18, -1. / 9, 1. / 9}}}};

constexpr array_3d matElem3d_dyudyu{{{{1. / 9, 1. / 18, -1. / 9, -1. / 18,
                                       1. / 18, 1. / 36, -1. / 18, -1. / 36}},
                                     {{1. / 18, 1. / 9, -1. / 18, -1. / 9,
                                       1. / 36, 1. / 18, -1. / 36, -1. / 18}},
                                     {{-1. / 9, -1. / 18, 1. / 9, 1. / 18,
                                       -1. / 18, -1. / 36, 1. / 18, 1. / 36}},
                                     {{-1. / 18, -1. / 9, 1. / 18, 1. / 9,
                                       -1. / 36, -1. / 18, 1. / 36, 1. / 18}},
                                     {{1. / 18, 1. / 36, -1. / 18, -1. / 36,
                                       1. / 9, 1. / 18, -1. / 9, -1. / 18}},
                                     {{1. / 36, 1. / 18, -1. / 36, -1. / 18,
                                       1. / 18, 1. / 9, -1. / 18, -1. / 9}},
                                     {{-1. / 18, -1. / 36, 1. / 18, 1. / 36,
                                       -1. / 9, -1. / 18, 1. / 9, 1. / 18}},
                                     {{-1. / 36, -1. / 18, 1. / 36, 1. / 18,
                                       -1. / 18, -1. / 9, 1. / 18, 1. / 9}}}};

constexpr array_3d matElem3d_dzudzu{{{{1. / 9, 1. / 18, 1. / 18, 1. / 36,
                                       -1. / 9, -1. / 18, -1. / 18, -1. / 36}},
                                     {{1. / 18, 1. / 9, 1. / 36, 1. / 18,
                                       -1. / 18, -1. / 9, -1. / 36, -1. / 18}},
                                     {{1. / 18, 1. / 36, 1. / 9, 1. / 18,
                                       -1. / 18, -1. / 36, -1. / 9, -1. / 18}},
                                     {{1. / 36, 1. / 18, 1. / 18, 1. / 9,
                                       -1. / 36, -1. / 18, -1. / 18, -1. / 9}},
                                     {{-1. / 9, -1. / 18, -1. / 18, -1. / 36,
                                       1. / 9, 1. / 18, 1. / 18, 1. / 36}},
                                     {{-1. / 18, -1. / 9, -1. / 36, -1. / 18,
                                       1. / 18, 1. / 9, 1. / 36, 1. / 18}},
                                     {{-1. / 18, -1. / 36, -1. / 9, -1. / 18,
                                       1. / 18, 1. / 36, 1. / 9, 1. / 18}},
                                     {{-1. / 36, -1. / 18, -1. / 18, -1. / 9,
                                       1. / 36, 1. / 18, 1. / 18, 1. / 9}}}};

constexpr array_3d matElem3d_dxudyu{{{{2. / 24, -2. / 24, 2. / 24, -2. / 24,
                                       1. / 24, -1. / 24, 1. / 24, -1. / 24}},
                                     {{2. / 24, -2. / 24, 2. / 24, -2. / 24,
                                       1. / 24, -1. / 24, 1. / 24, -1. / 24}},
                                     {{-2. / 24, 2. / 24, -2. / 24, 2. / 24,
                                       -1. / 24, 1. / 24, -1. / 24, 1. / 24}},
                                     {{-2. / 24, 2. / 24, -2. / 24, 2. / 24,
                                       -1. / 24, 1. / 24, -1. / 24, 1. / 24}},
                                     {{1. / 24, -1. / 24, 1. / 24, -1. / 24,
                                       2. / 24, -2. / 24, 2. / 24, -2. / 24}},
                                     {{1. / 24, -1. / 24, 1. / 24, -1. / 24,
                                       2. / 24, -2. / 24, 2. / 24, -2. / 24}},
                                     {{-1. / 24, 1. / 24, -1. / 24, 1. / 24,
                                       -2. / 24, 2. / 24, -2. / 24, 2. / 24}},
                                     {{-1. / 24, 1. / 24, -1. / 24, 1. / 24,
                                       -2. / 24, 2. / 24, -2. / 24, 2. / 24}}}};

constexpr array_3d matElem3d_dyudzu{{{{2. / 24, 1. / 24, -2. / 24, -1. / 24,
                                       2. / 24, 1. / 24, -2. / 24, -1. / 24}},
                                     {{1. / 24, 2. / 24, -1. / 24, -2. / 24,
                                       1. / 24, 2. / 24, -1. / 24, -2. / 24}},
                                     {{2. / 24, 1. / 24, -2. / 24, -1. / 24,
                                       2. / 24, 1. / 24, -2. / 24, -1. / 24}},
                                     {{1. / 24, 2. / 24, -1. / 24, -2. / 24,
                                       1. / 24, 2. / 24, -1. / 24, -2. / 24}},
                                     {{-2. / 24, -1. / 24, 2. / 24, 1. / 24,
                                       -2. / 24, -1. / 24, 2. / 24, 1. / 24}},
                                     {{-1. / 24, -2. / 24, 1. / 24, 2. / 24,
                                       -1. / 24, -2. / 24, 1. / 24, 2. / 24}},
                                     {{-2. / 24, -1. / 24, 2. / 24, 1. / 24,
                                       -2. / 24, -1. / 24, 2. / 24, 1. / 24}},
                                     {{-1. / 24, -2. / 24, 1. / 24, 2. / 24,
                                       -1. / 24, -2. / 24, 1. / 24, 2. / 24}}}};

constexpr array_3d matElem3d_dxudzu{{{{2. / 24, -2. / 24, 1. / 24, -1. / 24,
                                       2. / 24, -2. / 24, 1. / 24, -1. / 24}},
                                     {{2. / 24, -2. / 24, 1. / 24, -1. / 24,
                                       2. / 24, -2. / 24, 1. / 24, -1. / 24}},
                                     {{1. / 24, -1. / 24, 2. / 24, -2. / 24,
                                       1. / 24, -1. / 24, 2. / 24, -2. / 24}},
                                     {{1. / 24, -1. / 24, 2. / 24, -2. / 24,
                                       1. / 24, -1. / 24, 2. / 24, -2. / 24}},
                                     {{-2. / 24, 2. / 24, -1. / 24, 1. / 24,
                                       -2. / 24, 2. / 24, -1. / 24, 1. / 24}},
                                     {{-2. / 24, 2. / 24, -1. / 24, 1. / 24,
                                       -2. / 24, 2. / 24, -1. / 24, 1. / 24}},
                                     {{-1. / 24, 1. / 24, -2. / 24, 2. / 24,
                                       -1. / 24, 1. / 24, -2. / 24, 2. / 24}},
                                     {{-1. / 24, 1. / 24, -2. / 24, 2. / 24,
                                       -1. / 24, 1. / 24, -2. / 24, 2. / 24}}}};

constexpr array_3d matElemMass3d{{{{1. / 27, 1. / 54, 1. / 54, 1. / 108,
                                    1. / 54, 1. / 108, 1. / 108, 1. / 216}},
                                  {{1. / 54, 1. / 27, 1. / 108, 1. / 54,
                                    1. / 108, 1. / 54, 1. / 216, 1. / 108}},
                                  {{1. / 54, 1. / 108, 1. / 27, 1. / 54,
                                    1. / 108, 1. / 216, 1. / 54, 1. / 108}},
                                  {{1. / 108, 1. / 54, 1. / 54, 1. / 27,
                                    1. / 216, 1. / 108, 1. / 108, 1. / 54}},
                                  {{1. / 54, 1. / 108, 1. / 108, 1. / 216,
                                    1. / 27, 1. / 54, 1. / 54, 1. / 108}},
                                  {{1. / 108, 1. / 54, 1. / 216, 1. / 108,
                                    1. / 54, 1. / 27, 1. / 108, 1. / 54}},
                                  {{1. / 108, 1. / 216, 1. / 54, 1. / 108,
                                    1. / 54, 1. / 108, 1. / 27, 1. / 54}},
                                  {{1. / 216, 1. / 108, 1. / 108, 1. / 54,
                                    1. / 108, 1. / 54, 1. / 54, 1. / 27}}}};

constexpr array_3d_p matElem3d_pdxu{
    {{{25. / 192, -25. / 288, -25. / 576, 5. / 32, -5. / 48, -5. / 96,
       5. / 192,  -5. / 288,  -5. / 576,  5. / 32, -5. / 48, -5. / 96,
       3. / 16,   -1. / 8,    -1. / 16,   1. / 32, -1. / 48, -1. / 96,
       5. / 192,  -5. / 288,  -5. / 576,  1. / 32, -1. / 48, -1. / 96,
       1. / 192,  -1. / 288,  -1. / 576}},
     {{25. / 576, 25. / 288, -25. / 192, 5. / 96, 5. / 48, -5. / 32,
       5. / 576,  5. / 288,  -5. / 192,  5. / 96, 5. / 48, -5. / 32,
       1. / 16,   1. / 8,    -3. / 16,   1. / 96, 1. / 48, -1. / 32,
       5. / 576,  5. / 288,  -5. / 192,  1. / 96, 1. / 48, -1. / 32,
       1. / 576,  1. / 288,  -1. / 192}},
     {{5. / 192,  -5. / 288,  -5. / 576,  5. / 32, -5. / 48, -5. / 96,
       25. / 192, -25. / 288, -25. / 576, 1. / 32, -1. / 48, -1. / 96,
       3. / 16,   -1. / 8,    -1. / 16,   5. / 32, -5. / 48, -5. / 96,
       1. / 192,  -1. / 288,  -1. / 576,  1. / 32, -1. / 48, -1. / 96,
       5. / 192,  -5. / 288,  -5. / 576}},
     {{5. / 576,  5. / 288,  -5. / 192,  5. / 96, 5. / 48, -5. / 32,
       25. / 576, 25. / 288, -25. / 192, 1. / 96, 1. / 48, -1. / 32,
       1. / 16,   1. / 8,    -3. / 16,   5. / 96, 5. / 48, -5. / 32,
       1. / 576,  1. / 288,  -1. / 192,  1. / 96, 1. / 48, -1. / 32,
       5. / 576,  5. / 288,  -5. / 192}},
     {{5. / 192,  -5. / 288,  -5. / 576,  1. / 32, -1. / 48, -1. / 96,
       1. / 192,  -1. / 288,  -1. / 576,  5. / 32, -5. / 48, -5. / 96,
       3. / 16,   -1. / 8,    -1. / 16,   1. / 32, -1. / 48, -1. / 96,
       25. / 192, -25. / 288, -25. / 576, 5. / 32, -5. / 48, -5. / 96,
       5. / 192,  -5. / 288,  -5. / 576}},
     {{5. / 576,  5. / 288,  -5. / 192,  1. / 96, 1. / 48, -1. / 32,
       1. / 576,  1. / 288,  -1. / 192,  5. / 96, 5. / 48, -5. / 32,
       1. / 16,   1. / 8,    -3. / 16,   1. / 96, 1. / 48, -1. / 32,
       25. / 576, 25. / 288, -25. / 192, 5. / 96, 5. / 48, -5. / 32,
       5. / 576,  5. / 288,  -5. / 192}},
     {{1. / 192,  -1. / 288,  -1. / 576, 1. / 32, -1. / 48, -1. / 96,
       5. / 192,  -5. / 288,  -5. / 576, 1. / 32, -1. / 48, -1. / 96,
       3. / 16,   -1. / 8,    -1. / 16,  5. / 32, -5. / 48, -5. / 96,
       5. / 192,  -5. / 288,  -5. / 576, 5. / 32, -5. / 48, -5. / 96,
       25. / 192, -25. / 288, -25. / 576}},
     {{1. / 576,  1. / 288,  -1. / 192, 1. / 96, 1. / 48, -1. / 32,
       5. / 576,  5. / 288,  -5. / 192, 1. / 96, 1. / 48, -1. / 32,
       1. / 16,   1. / 8,    -3. / 16,  5. / 96, 5. / 48, -5. / 32,
       5. / 576,  5. / 288,  -5. / 192, 5. / 96, 5. / 48, -5. / 32,
       25. / 576, 25. / 288, -25. / 192}}}};

constexpr array_3d_p matElem3d_pdyu{
    {{{25. / 192,  5. / 32,  5. / 192,  -25. / 288, -5. / 48, -5. / 288,
       -25. / 576, -5. / 96, -5. / 576, 5. / 32,    3. / 16,  1. / 32,
       -5. / 48,   -1. / 8,  -1. / 48,  -5. / 96,   -1. / 16, -1. / 96,
       5. / 192,   1. / 32,  1. / 192,  -5. / 288,  -1. / 48, -1. / 288,
       -5. / 576,  -1. / 96, -1. / 576}},
     {{5. / 192,  5. / 32,  25. / 192,  -5. / 288, -5. / 48, -25. / 288,
       -5. / 576, -5. / 96, -25. / 576, 1. / 32,   3. / 16,  5. / 32,
       -1. / 48,  -1. / 8,  -5. / 48,   -1. / 96,  -1. / 16, -5. / 96,
       1. / 192,  1. / 32,  5. / 192,   -1. / 288, -1. / 48, -5. / 288,
       -1. / 576, -1. / 96, -5. / 576}},
     {{25. / 576,  5. / 96,  5. / 576,  25. / 288, 5. / 48,  5. / 288,
       -25. / 192, -5. / 32, -5. / 192, 5. / 96,   1. / 16,  1. / 96,
       5. / 48,    1. / 8,   1. / 48,   -5. / 32,  -3. / 16, -1. / 32,
       5. / 576,   1. / 96,  1. / 576,  5. / 288,  1. / 48,  1. / 288,
       -5. / 192,  -1. / 32, -1. / 192}},
     {{5. / 576,  5. / 96,  25. / 576,  5. / 288, 5. / 48,  25. / 288,
       -5. / 192, -5. / 32, -25. / 192, 1. / 96,  1. / 16,  5. / 96,
       1. / 48,   1. / 8,   5. / 48,    -1. / 32, -3. / 16, -5. / 32,
       1. / 576,  1. / 96,  5. / 576,   1. / 288, 1. / 48,  5. / 288,
       -1. / 192, -1. / 32, -5. / 192}},
     {{5. / 192,   1. / 32,  1. / 192,  -5. / 288,  -1. / 48, -1. / 288,
       -5. / 576,  -1. / 96, -1. / 576, 5. / 32,    3. / 16,  1. / 32,
       -5. / 48,   -1. / 8,  -1. / 48,  -5. / 96,   -1. / 16, -1. / 96,
       25. / 192,  5. / 32,  5. / 192,  -25. / 288, -5. / 48, -5. / 288,
       -25. / 576, -5. / 96, -5. / 576}},
     {{1. / 192,  1. / 32,  5. / 192,  -1. / 288, -1. / 48, -5. / 288,
       -1. / 576, -1. / 96, -5. / 576, 1. / 32,   3. / 16,  5. / 32,
       -1. / 48,  -1. / 8,  -5. / 48,  -1. / 96,  -1. / 16, -5. / 96,
       5. / 192,  5. / 32,  25. / 192, -5. / 288, -5. / 48, -25. / 288,
       -5. / 576, -5. / 96, -25. / 576}},
     {{5. / 576,   1. / 96,  1. / 576,  5. / 288,  1. / 48,  1. / 288,
       -5. / 192,  -1. / 32, -1. / 192, 5. / 96,   1. / 16,  1. / 96,
       5. / 48,    1. / 8,   1. / 48,   -5. / 32,  -3. / 16, -1. / 32,
       25. / 576,  5. / 96,  5. / 576,  25. / 288, 5. / 48,  5. / 288,
       -25. / 192, -5. / 32, -5. / 192}},
     {{1. / 576,  1. / 96,  5. / 576,  1. / 288, 1. / 48,  5. / 288,
       -1. / 192, -1. / 32, -5. / 192, 1. / 96,  1. / 16,  5. / 96,
       1. / 48,   1. / 8,   5. / 48,   -1. / 32, -3. / 16, -5. / 32,
       5. / 576,  5. / 96,  25. / 576, 5. / 288, 5. / 48,  25. / 288,
       -5. / 192, -5. / 32, -25. / 192}}}};

constexpr array_3d_p matElem3d_pdzu{
    {{{25. / 192,  5. / 32,  5. / 192,  5. / 32,    3. / 16,  1. / 32,
       5. / 192,   1. / 32,  1. / 192,  -25. / 288, -5. / 48, -5. / 288,
       -5. / 48,   -1. / 8,  -1. / 48,  -5. / 288,  -1. / 48, -1. / 288,
       -25. / 576, -5. / 96, -5. / 576, -5. / 96,   -1. / 16, -1. / 96,
       -5. / 576,  -1. / 96, -1. / 576}},
     {{5. / 192,  5. / 32,  25. / 192,  1. / 32,   3. / 16,  5. / 32,
       1. / 192,  1. / 32,  5. / 192,   -5. / 288, -5. / 48, -25. / 288,
       -1. / 48,  -1. / 8,  -5. / 48,   -1. / 288, -1. / 48, -5. / 288,
       -5. / 576, -5. / 96, -25. / 576, -1. / 96,  -1. / 16, -5. / 96,
       -1. / 576, -1. / 96, -5. / 576}},
     {{5. / 192,   1. / 32,  1. / 192,  5. / 32,    3. / 16,  1. / 32,
       25. / 192,  5. / 32,  5. / 192,  -5. / 288,  -1. / 48, -1. / 288,
       -5. / 48,   -1. / 8,  -1. / 48,  -25. / 288, -5. / 48, -5. / 288,
       -5. / 576,  -1. / 96, -1. / 576, -5. / 96,   -1. / 16, -1. / 96,
       -25. / 576, -5. / 96, -5. / 576}},
     {{1. / 192,  1. / 32,  5. / 192,  1. / 32,   3. / 16,  5. / 32,
       5. / 192,  5. / 32,  25. / 192, -1. / 288, -1. / 48, -5. / 288,
       -1. / 48,  -1. / 8,  -5. / 48,  -5. / 288, -5. / 48, -25. / 288,
       -1. / 576, -1. / 96, -5. / 576, -1. / 96,  -1. / 16, -5. / 96,
       -5. / 576, -5. / 96, -25. / 576}},
     {{25. / 576,  5. / 96,  5. / 576,  5. / 96,   1. / 16,  1. / 96,
       5. / 576,   1. / 96,  1. / 576,  25. / 288, 5. / 48,  5. / 288,
       5. / 48,    1. / 8,   1. / 48,   5. / 288,  1. / 48,  1. / 288,
       -25. / 192, -5. / 32, -5. / 192, -5. / 32,  -3. / 16, -1. / 32,
       -5. / 192,  -1. / 32, -1. / 192}},
     {{5. / 576,  5. / 96,  25. / 576,  1. / 96,  1. / 16,  5. / 96,
       1. / 576,  1. / 96,  5. / 576,   5. / 288, 5. / 48,  25. / 288,
       1. / 48,   1. / 8,   5. / 48,    1. / 288, 1. / 48,  5. / 288,
       -5. / 192, -5. / 32, -25. / 192, -1. / 32, -3. / 16, -5. / 32,
       -1. / 192, -1. / 32, -5. / 192}},
     {{5. / 576,   1. / 96,  1. / 576,  5. / 96,   1. / 16,  1. / 96,
       25. / 576,  5. / 96,  5. / 576,  5. / 288,  1. / 48,  1. / 288,
       5. / 48,    1. / 8,   1. / 48,   25. / 288, 5. / 48,  5. / 288,
       -5. / 192,  -1. / 32, -1. / 192, -5. / 32,  -3. / 16, -1. / 32,
       -25. / 192, -5. / 32, -5. / 192}},
     {{1. / 576,  1. / 96,  5. / 576,  1. / 96,  1. / 16,  5. / 96,
       5. / 576,  5. / 96,  25. / 576, 1. / 288, 1. / 48,  5. / 288,
       1. / 48,   1. / 8,   5. / 48,   5. / 288, 5. / 48,  25. / 288,
       -1. / 192, -1. / 32, -5. / 192, -1. / 32, -3. / 16, -5. / 32,
       -5. / 192, -5. / 32, -25. / 192}}}};

auto getMatElemLaplacian(std::array<double, 2> const &h)
{
    array_2d MatElem;
    double hxy = h[0] / h[1], hyx = h[1] / h[0];

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            MatElem[i][j] =
                hyx * matElem2d_dxudxu[i][j] + hxy * matElem2d_dyudyu[i][j];

    return MatElem;
}

auto getMatElemLaplacianA(std::array<double, 2> const &h)
{
    std::array<double, 16> MatElem;
    double hxy = h[0] / h[1], hyx = h[1] / h[0];

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            MatElem[i + j * 4] =
                hyx * matElem2d_dxudxu[i][j] + hxy * matElem2d_dyudyu[i][j];

    return MatElem;
}

auto getMatElemStrainTensor(std::array<double, 2> const &h)
{
    std::array<std::array<std::array<std::array<double, 2>, 2>, 4>, 4> MatElem;
    double hxy = h[0] / h[1], hyx = h[1] / h[0];

    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            MatElem[i][j][0][0] =
                2 * hyx * matElem2d_dxudxu[i][j] + hxy * matElem2d_dyudyu[i][j];
            MatElem[i][j][0][1] = matElem2d_dxudyu[i][j];
            MatElem[i][j][1][1] =
                hyx * matElem2d_dxudxu[i][j] + 2 * hxy * matElem2d_dyudyu[i][j];
        }
    }

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            MatElem[j][i][1][0] = MatElem[i][j][0][1];

    return MatElem;
}

auto getMatElemStrainTensorA(std::array<double, 2> const &h)
{
    std::array<double, 64> MatElem;
    double hxy = h[0] / h[1], hyx = h[1] / h[0];

    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            MatElem[i + j * 8] =
                2 * hyx * matElem2d_dxudxu[i][j] + hxy * matElem2d_dyudyu[i][j];
            MatElem[i + 4 + j * 8] = matElem2d_dxudyu[i][j];

            MatElem[i + (j + 4) * 8] = matElem2d_dxudyu[i][j];
            MatElem[i + 4 + (j + 4) * 8] =
                hyx * matElem2d_dxudxu[i][j] + 2 * hxy * matElem2d_dyudyu[i][j];
        }
    }
    return MatElem;
}

auto getMatElemPressure(std::array<double, 2> const &h)
{
    // return the integral of p div u
    std::array<std::array<std::array<double, 2>, 9>, 4> MatElem;

    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 9; ++j)
        {
            MatElem[i][j][0] = -h[1] * matElem2d_pdxu[i][j];
            MatElem[i][j][1] = -h[0] * matElem2d_pdyu[i][j];
        }
    }

    return MatElem;
}

auto getMatElemPressureA(std::array<double, 2> const &h)
{
    // return the integral of p div u
    std::array<double, 72> MatElem;

    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 9; ++j)
        {
            MatElem[18 * i + j] = -h[1] * matElem2d_pdxu[i][j];
            MatElem[18 * i + j + 9] = -h[0] * matElem2d_pdyu[i][j];
        }
    }

    return MatElem;
}

auto getMatElemPressureAT(std::array<double, 2> const &h)
{
    // return the integral of p div u
    std::array<double, 72> MatElem;

    for (std::size_t j = 0; j < 9; ++j)
    {
        for (std::size_t i = 0; i < 4; ++i)
        {
            MatElem[i + 4 * j] = h[1] * matElem2d_pdxu[i][j];
            MatElem[i + 4 * (j + 9)] = h[0] * matElem2d_pdyu[i][j];
        }
    }

    return MatElem;
}

auto getMatElemMass(std::array<double, 2> const &h)
{
    array_2d MatElem;

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            MatElem[i][j] = h[0] * h[1] * matElemMass2d[i][j];

    return MatElem;
}

auto getMatElemMassA(std::array<double, 2> const &h)
{
    std::array<double, 16> MatElem;

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            MatElem[i + 4 * j] = h[0] * h[1] * matElemMass2d[i][j];

    return MatElem;
}

auto getMatElemLaplacian(std::array<double, 3> const &h)
{
    array_3d MatElem;
    double hxy = h[0] * h[1], hxz = h[0] * h[2], hyz = h[1] * h[2];

    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            MatElem[i][j] = hyz / h[0] * matElem3d_dxudxu[i][j] +
                            hxz / h[1] * matElem3d_dyudyu[i][j] +
                            hxy / h[2] * matElem3d_dzudzu[i][j];

    return MatElem;
}

auto getMatElemStrainTensor(std::array<double, 3> const &h)
{
    std::array<std::array<std::array<std::array<double, 3>, 3>, 8>, 8> MatElem;
    double hxy = h[0] * h[1], hxz = h[0] * h[2], hyz = h[1] * h[2];

    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 8; ++j)
        {
            MatElem[i][j][0][0] = 2 * hyz / h[0] * matElem3d_dxudxu[i][j] +
                                  hxz / h[1] * matElem3d_dyudyu[i][j] +
                                  hxy / h[2] * matElem3d_dzudzu[i][j];
            MatElem[i][j][1][1] = hyz / h[0] * matElem3d_dxudxu[i][j] +
                                  2 * hxz / h[1] * matElem3d_dyudyu[i][j] +
                                  hxy / h[2] * matElem3d_dzudzu[i][j];
            MatElem[i][j][2][2] = hyz / h[0] * matElem3d_dxudxu[i][j] +
                                  hxz / h[1] * matElem3d_dyudyu[i][j] +
                                  2 * hxy / h[2] * matElem3d_dzudzu[i][j];
            MatElem[i][j][0][1] = h[2] * matElem3d_dxudyu[i][j];
            MatElem[i][j][0][2] = h[1] * matElem3d_dxudzu[i][j];
            MatElem[i][j][1][2] = h[0] * matElem3d_dyudzu[i][j];
        }
    }

    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
        {
            MatElem[j][i][1][0] = MatElem[i][j][0][1];
            MatElem[j][i][2][0] = MatElem[i][j][0][2];
            MatElem[j][i][2][1] = MatElem[i][j][1][2];
        }

    return MatElem;
}

auto getMatElemPressure(std::array<double, 3> const &h)
{
    std::array<std::array<std::array<double, 3>, 27>, 8> MatElem;

    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 27; ++j)
        {
            MatElem[i][j][0] = -h[1] * h[2] * matElem3d_pdxu[i][j];
            MatElem[i][j][1] = -h[0] * h[2] * matElem3d_pdyu[i][j];
            MatElem[i][j][2] = -h[0] * h[1] * matElem3d_pdzu[i][j];
        }
    }

    return MatElem;
}

auto getMatElemMass(std::array<double, 3> const &h)
{
    array_3d MatElem;

    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            MatElem[i][j] = h[0] * h[1] * h[2] * matElemMass3d[i][j];

    return MatElem;
}
#endif