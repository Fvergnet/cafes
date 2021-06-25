#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cafes.hpp>
// #include <algorithm/iterate.hpp>
// #include <particle/geometry/position.hpp>
// #include <particle/geometry/circle.hpp>
// #include <particle/singularity/singularity.hpp>
// #include <particle/singularity/extension.hpp>
// #include "../../cafes/singularity.hpp"
// 
// #include "../../cpp/truncation.hpp"

// #include "../../cpp/circle.hpp"

// template<typename T, std::size_t Dimensions>
// struct position : std::array<double, 2>
// {
//   using parent = std::array<double, 2>;
//   using parent::operator[];
//   using parent::begin;
//   using parent::end;
//   using parent::fill;
//   using parent::data;
//   position(double x, double y) : parent{{x,y}} {}

//   // double getx()
//   // {
//   //   return (*this)[0];
//   // }
// };

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "Babic Python interface";
  m.attr("__version__") = "0.1";

  // Create truncation submodule [truncation]
  m.def("extchiTrunc", &cafes::singularity::extchiTrunc, "Truncation function to use in Babic extension");
  m.def("dextchiTrunc", &cafes::singularity::dextchiTrunc, "Derivative of truncation function to use in Babic extension");

  // Create position object
  using position = cafes::geometry::position<double,2>;
  py::class_<position>(m,"position")
    .def(py::init<double, double>())
    .def("get",[](const position &p, int &i) {
            return p[i];
        });

  // Create velocity object
  using velocity = cafes::physics::velocity<2>;
  py::class_<velocity>(m,"velocity")
    .def(py::init<double, double>());

  // Create circle object
  using circle = cafes::geometry::circle<>;
  py::class_<circle>(m,"circle")
    .def(py::init<position const&, double const&>());

  // Create particle object
  using particle = cafes::particle<circle>;
  using angular_velocity_type = std::conditional<circle::dimension_type::value==2,
                                                               double, 
                                                               cafes::geometry::vector<double, 3>>::type;
  py::class_<particle>(m, "particle")
    .def(py::init<circle const&, velocity const&, angular_velocity_type const&>())
    .def("get_surface",[](particle& self, int const& N) {
      std::vector<std::array<double,2>> surf;
      double dtheta = 2*std::atan(1.0)*4/N;
      double theta=0;
      for (std::size_t i=0; i<N+1; ++i) {
        surf.push_back( {{ self.center_[0] + self.shape_factors_[0]*std::cos(theta), self.center_[1] + self.shape_factors_[0]*std::sin(theta) }} );
        theta += dtheta;
      }
      return surf;
    });

  // Create singularity class
  using sing = cafes::singularity::singularity<circle,2>;
  py::class_<sing>(m, "singularity")
    .def(py::init<cafes::particle<circle> const &, cafes::particle<circle> const &, double, double, double>())
    .def("get_p_sing",[](sing& self, position pos) {
        return self.get_p_sing(pos, std::integral_constant<int, 2>{});
    })
    .def("get_u_sing",[](sing& self, position pos) {
        return self.get_u_sing(pos, std::integral_constant<int, 2>{});
    })
    .def("get_grad_u_sing",[](sing& self, position pos) {
        return self.get_grad_u_sing(pos);
    });

  // Functiona to get babic etension
  m.def("babic_extension", &cafes::singularity::babic_extension);
  m.def("get_Babic_field_extension", [](sing s, particle p, position pts) {
    double pressure = 0;
    velocity vel {0,0};
    cafes::singularity::get_Babic_field_extension<circle, 2, velocity>(s, p, pts, vel, pressure);
    std::array<double,3> out {pressure, vel[0], vel[1]};
    return out;
  });
  m.def("get_Babic_singularity_extension", [](sing s, particle p, position pts) {
    double pressure = 0;
    std::array< std::array<double, 2>, 2> vel {{ {0,0},{0,0} }};
    cafes::singularity::get_Babic_singularity_extension<circle, 2>(s, p, pts, vel, pressure);
    std::array<double,5> out {pressure, vel[0][0], vel[0][1], vel[1][0], vel[1][1]};
    return out;
  });
}