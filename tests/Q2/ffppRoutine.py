import os

class ffppRoutine:
    def __init__(self, Nborder, Nbetweenparticles, radius, distance):
        self.Nborder = Nborder
        self.N = Nbetweenparticles
        self.radius = radius
        self.distance = distance

    def freefem_code(self, filename="reference"):
        ffmesh='''// Parameters
int Nborder = %d;
int N = %d;
real distance = %f;
real radius = %f;

// Construct and save mesh
border bottom(t=0,1){x=t; y=0; label=1;};
border right(t=0,1){x=1.; y=t; label=1;};
border top(t=1,0){x=t; y=1.; label=1;};
border left(t=1,0){x=0.; y=t; label=1;};
border circle1(t=0,2*pi){x=0.5-distance/2.-radius + radius*cos(t); y = 0.5 + radius*sin(t); label=2;};
border circle2(t=0,2*pi){x=0.5+distance/2.+radius + radius*cos(t); y = 0.5 + radius*sin(t); label=3;};

int Ncircle = max(2*pi*radius*N/distance, 2*pi*radius*Nborder);

mesh Th = buildmesh(bottom(Nborder) + right(Nborder) + top(Nborder) + left(Nborder) + circle1(-Ncircle) + circle2(-Ncircle));

savemesh(Th,"mesh_distance_is_radius_over_"+int(radius/distance)+".msh");

// Fespace
fespace Uh(Th, P2);
Uh ux, uy;
Uh vx, vy;
fespace Ph(Th, P1);
Ph p, q;

// Save velocity and pressure dofs
Ph xp=x, yp=y;
ofstream filedofspressure("%s_mesh_distance_is_radius_over_"+int(radius/distance)+"_pressure.txt");
for (int i=0; i<Ph.ndof; i++)
{
    filedofspressure.scientific << xp[][i] << " " << yp[][i] << endl;
}

Uh xu=x, yu=y;
ofstream filedofsvelocity("%s_mesh_distance_is_radius_over_"+int(radius/distance)+"_velocity.txt");
for (int i=0; i<Uh.ndof; i++)
{
    filedofsvelocity.scientific << xu[][i] << " " << yu[][i] << endl;
}

// Solve Stokes problem
solve stokes ([ux, uy, p], [vx, vy, q])
    = int2d(Th)(
          dx(ux)*dx(vx)
        + dy(ux)*dy(vx)
        + dx(uy)*dx(vy)
        + dy(uy)*dy(vy)
        - p*(dx(vx) + dy(vy))
        - q*(dx(ux) + dy(uy))
        - 1e-10*p*q
    )
    + on(1, ux=0, uy=0)
    + on(2, ux=1, uy=0)
    + on(3, ux=-1, uy=0)
    ;

// Save solution
ofstream solutionpressure("%s_solution_distance_is_radius_over_"+int(radius/distance)+"_pressure.txt");
for (int i=0; i<Ph.ndof; i++)
{
    solutionpressure.scientific << p[][i] << endl;
}

ofstream solutionvelocity("%s_solution_distance_is_radius_over_"+int(radius/distance)+"_velocity.txt");
for (int i=0; i<Uh.ndof; i++)
{
    solutionvelocity.scientific << ux[][i] << " " << uy[][i] << endl;
}
''' % (self.Nborder, self.N, self.distance, self.radius, filename, filename, filename, filename)
        return ffmesh

    def compute_solution(self, repertory):
        os.system('mkdir {}'.format(repertory))
        ffmesh = self.freefem_code()
        with open(repertory+"/compute_solution.edp","w") as file:
            file.write(ffmesh)
        os.chdir('{}'.format(repertory))
        os.system("FreeFem++ compute_solution.edp")

    def compute_errors_code(self, repertory, filemesh, filereference, filescafes):
        ffcode='''
mesh Th = readmesh("%s.msh");

// Fespace
fespace Uh(Th, P2);
Uh uxref, uyref, ux, uy;
fespace Ph(Th, P1);
Ph pref, p;

// Read reference solution
ifstream solutionpressure("%s_pressure.txt");
for (int i=0; i<Ph.ndof; i++)
{
    solutionpressure >> pref[][i];
}

ifstream solutionvelocity("%s_velocity.txt");
for (int i=0; i<Uh.ndof; i++)
{
    solutionvelocity >> uxref[][i] >> uyref[][i];
}

// Read Cafes solutions
int size = %d;
string[int] cafesfiles = ["%s"];
ofstream errors("%s/errors.txt");
for (int f=0; f<size; f++)
{
    cout << "file : " << cafesfiles[f]<< endl;
    ifstream solutionpressurecafes(cafesfiles[f]+"_pressure.txt");
    for (int i=0; i<Ph.ndof; i++)
    {
        solutionpressurecafes >> p[][i];
    }

    ifstream solutionvelocitycafes(cafesfiles[f]+"_velocity.txt");
    for (int i=0; i<Uh.ndof; i++)
    {
        solutionvelocitycafes >> ux[][i] >> uy[][i];
    }

    // Compute errors
    Ph perr = pref-p;
    Uh uxerr = uxref-ux, uyerr = uyref - uy; 
    real l2p = sqrt(int2d(Th)(perr*perr));
    real l2x = sqrt(int2d(Th)(uxerr*uxerr));
    real l2y = sqrt(int2d(Th)(uyerr*uyerr));
    real h1x = sqrt(int2d(Th)(dx(uxerr)*dx(uxerr) + dy(uxerr)*dy(uxerr)));
    real h1y = sqrt(int2d(Th)(dx(uyerr)*dx(uyerr) + dy(uyerr)*dy(uyerr)));

    errors << l2p << " " << l2x << " " << l2y << " " << h1x << " " << h1y << endl;
}
''' % (filemesh, filereference, filereference, len(filescafes), '","'.join(filescafes), repertory)
        return ffcode

    def compute_errors(self, repertory, referencemesh, referencesolution, cafesfiles):
        ffcode = self.compute_errors_code(repertory, referencemesh, referencesolution, cafesfiles)
        with open('{}/compute_errors.edp'.format(repertory),'w') as file:
            file.write(ffcode)
        os.chdir('{}'.format(repertory))
        os.system("FreeFem++ compute_errors.edp")



if __name__ == "__main__":
    test = ffppRoutine(100,10,0.1,0.1/4.)
    test.compute_solution("tresse")
    # test.compute_errors("test")