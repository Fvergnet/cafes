#include <cafes.hpp>
#include <petsc.h>
#include <petsc/vec.hpp>
#include <fstream>
#include <type_traits>

template<typename T>
struct InterpolationContext
{
    DM dm;
    std::array<double,2> h;
    const T& p1;
    const T& p2;
    bool compute_singularity;
};

template<class T>
auto make_interpolation_context(DM dm_, const std::array<double,2>& h_, const T& p1_, const T& p2_, bool compute_singularity_)
{
    return InterpolationContext<T>{dm_, h_, p1_, p2_, compute_singularity_};
}

auto get_element_from_point(std::vector<double> pt, std::array<double,2> h, int order)
{
    cafes::geometry::position<int, 2> pts_i;
    for (std::size_t i=0; i<pt.size(); i++)
    {
        int index = pt[i]/h[i];
        int N = 1./h[i];
        if (index==N) index-=1;
        pts_i[i] = index;
        // std::cout << pt[i] << " " << h[i] << " " << index << " ";
    }
    // std::cout << " \n";
    auto ielem = cafes::fem::get_element(order*pts_i, order);
    return ielem;
}

auto get_local_coordinate(std::vector<double> pt, std::array<double,2> h)
{
    cafes::geometry::position<double, 2> pt_loc;
    for (std::size_t i=0; i<pt.size(); i++)
    {
        int index = pt[i]/h[i];
        int N = 1./h[i];
        if (index==N) index-=1;
        pt_loc[i] = pt[i] - index*h[i];
    }
    return pt_loc;
}

auto loadmesh(std::string file)
{
    std::ifstream infile(file);
    std::vector<std::vector<double>> dofs_coordinates;
    std::string str, s;

    while (std::getline(infile, str))
    {
        std::stringstream ss(str);
        std::vector<double> line;
        while (std::getline(ss, s, ' '))
        {
            line.push_back(std::stod(s));
        }
        // std::cout << line[0] << " " << line[1] << std::endl;
        dofs_coordinates.push_back(line);
    }
    return dofs_coordinates;
}

class FenicsFunction
{
    public:
    int velocity_size, pressure_size, dim;
    std::vector<std::vector<double>> velocity_coordinates, pressure_coordinates, velocity;
    std::vector<double> pressure;
    FenicsFunction(std::string velocity_mesh, std::string pressure_mesh){
        velocity_coordinates = loadmesh(velocity_mesh);
        pressure_coordinates = loadmesh(pressure_mesh);
        velocity_size = velocity_coordinates.size();
        pressure_size = pressure_coordinates.size();
        dim = velocity_coordinates[0].size();
        velocity.resize(velocity_size);
        pressure.resize(pressure_size);    
        for (std::size_t i=0; i<velocity_size; ++i)
        {
          velocity[i].resize(dim);
        }
    }

    template<typename CTX, typename T>
    auto interpolate(CTX ctx, T From)
    {
        PetscErrorCode ierr;
        PetscFunctionBeginUser;
        std::array<double,2> hu = {{.5*ctx.h[0], .5*ctx.h[1]}};
        auto From_velocity = cafes::petsc::petsc_vec<2>(ctx.dm, From, 0);
        auto From_pressure = cafes::petsc::petsc_vec<2>(ctx.dm, From, 1);
        ierr = From_velocity.global_to_local(INSERT_VALUES);CHKERRQ(ierr);
        ierr = From_pressure.global_to_local(INSERT_VALUES);CHKERRQ(ierr);

        using part_type = typename std::remove_reference<decltype(ctx.p1)>::type;
        using shape_type = typename part_type::shape_type;
        auto sing = cafes::singularity::singularity<shape_type, 2>(ctx.p1, ctx.p2, ctx.h[0]);

        // interpolate velocity
        for (std::size_t ipt=0; ipt<velocity_size; ipt++)
        {
            auto ielem = get_element_from_point(velocity_coordinates[ipt], ctx.h, 2);
            auto pt_loc = get_local_coordinate(velocity_coordinates[ipt], ctx.h);
            auto bfunc = cafes::fem::P2_integration(pt_loc, hu);
            
            if (ctx.compute_singularity)
            {
                cafes::geometry::position<double, 2> pt{velocity_coordinates[ipt][0], velocity_coordinates[ipt][1]};
                auto u_sing = sing.get_u_sing(pt);
                for (std::size_t d=0; d<dim; d++)
                {
                    velocity[ipt][d] += u_sing[d];
                }
            }
            
            for (std::size_t je=0; je<ielem.size(); je++)
            {
                auto u = From_velocity.at(ielem[je]);
                for (std::size_t d=0; d<dim; d++)
                {
                    velocity[ipt][d] += u[d]*bfunc[je];
                }
            }
        }

        // interpolate pressure
        for (std::size_t ipt=0; ipt<pressure_size; ipt++)
        {
            auto ielem = get_element_from_point(pressure_coordinates[ipt], ctx.h, 1);
            auto pt_loc = get_local_coordinate(pressure_coordinates[ipt], ctx.h);
            auto bfunc = cafes::fem::P1_integration(pt_loc, ctx.h);

            if (ctx.compute_singularity)
            {
                cafes::geometry::position<double, 2> pt{pressure_coordinates[ipt][0], pressure_coordinates[ipt][1]};
                auto p_sing = sing.get_p_sing(pt);
                pressure[ipt] += p_sing;
            }

            double mean = 0;
            int N = 1./ctx.h[0];
            double coef = .25*ctx.h[0]*ctx.h[1];
            for (std::size_t j=0; j<N; j++)
            {
                for (std::size_t i=0; i<N; i++)
                {
                    cafes::geometry::position<int, 2> pts_i = {i, j};
                    auto ielem = cafes::fem::get_element(pts_i,1);
                    for (std::size_t je=0; je<ielem.size(); je++)
                    {
                        auto u = From_pressure.at(ielem[je]);
                        mean += coef*u[0];
                    }
                    
                }
            }

            for (std::size_t je=0; je<ielem.size(); je++)
            { 
                auto u = From_pressure.at(ielem[je]);
                pressure[ipt] += (u[0]-mean)*bfunc[je];
                // std::cout << u[0] << std::endl;
            }
            // std::cout << pressure[ipt] << " " << p_sing << std::endl;
        }

        PetscFunctionReturn(0);
    }

    void save(std::string file)
    {
        // save velocity
        std::ofstream velocity_file;
        velocity_file.open(file + "_velocity.txt");
        for (std::size_t i=0; i<velocity_size; i++)
        {
            // velocity_file << velocity_coordinates[i][0] << " " << velocity_coordinates[i][1] << " ";
            for (std::size_t d=0; d<dim; d++)
            {
                velocity_file << std::scientific << velocity[i][d] << " ";
            }
            velocity_file << "\n";
        }
        velocity_file.close();

        // save pressure
        std::ofstream pressure_file;
        pressure_file.open(file + "_pressure.txt");
        for (std::size_t i=0; i<pressure_size; i++)
        {
            pressure_file << std::scientific << pressure[i] << "\n";
        }
        pressure_file.close();

    }

};