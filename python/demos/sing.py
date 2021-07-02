from context import cafes
import numpy as np
import matplotlib.pyplot as plt

order = 1

r1 = 0.1
r2 = 0.1
d = 0.025
x1 = 0.5-d/2-r1
y1=0.5
x2 = 0.5+d/2+r2
y2=0.5
pos1 = cafes.cpp.position(x1, y1)
pos2 = cafes.cpp.position(x2, y2)
c1 = cafes.cpp.circle(pos1,r1)
c2 = cafes.cpp.circle(pos2,r2)
v1 = cafes.cpp.velocity(1,0)
v2 = cafes.cpp.velocity(-1,0)
p1 = cafes.cpp.particle(c1,v1,0.)
p2 = cafes.cpp.particle(c2,v2,0.)
surf1 = np.array(p1.get_surface(100))
surf2 = np.array(p2.get_surface(100))

sing = cafes.cpp.singularity(p1,p2,0.01,4,1./5)

eps = r1/3;
l = 2*r1/3;


N = 100
# x=np.linspace(0,1,N)
# y=x.copy()
x = np.linspace(pos1.get(0),pos2.get(0),N)
y =  np.linspace(pos1.get(1)-r1, pos1.get(1)+r1,N)
x,y = np.meshgrid(x,y)
P = np.zeros(x.shape)
Ux = P.copy()
Uy = P.copy()

Pcpp = P.copy()
Uxcpp = P.copy()
Uycpp = P.copy()

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if (x[i,j]-x1)**2 + (y[i,j]-y1)**2>=r1**2 and (x[i,j]-x2)**2 + (y[i,j]-y2)**2>=r2**2:
            P[i,j] = sing.get_p_sing(cafes.cpp.position(x[i,j],y[i,j]))
            tmp = sing.get_u_sing(cafes.cpp.position(x[i,j],y[i,j]))
            Ux[i,j] = tmp[0]
            Uy[i,j] = tmp[1]
            # cpp
            Pcpp[i,j] = sing.get_p_sing(cafes.cpp.position(x[i,j],y[i,j]))
            tmp = sing.get_u_sing(cafes.cpp.position(x[i,j],y[i,j]))
            Uxcpp[i,j] = tmp[0]
            Uycpp[i,j] = tmp[1]
        elif (x[i,j]-x1)**2 + (y[i,j]-y1)**2<r1**2:
            r = np.sqrt((x1-x[i,j])**2 + (y1-y[i,j])**2)
            theta = np.arctan2(y[i,j]-y1,x[i,j]-x1)
            fp = lambda r: sing.get_p_sing(cafes.cpp.position(x1 + r*np.cos(theta), y1 + r*np.sin(theta)))
            fux = lambda r: sing.get_u_sing(cafes.cpp.position(x1 + r*np.cos(theta), y1 + r*np.sin(theta)))[0]
            fuy = lambda r: sing.get_u_sing(cafes.cpp.position(x1 + r*np.cos(theta), y1 + r*np.sin(theta)))[1]
            chi = (1-cafes.cpp.extchiTrunc(r,l,eps))
            P[i,j] = cafes.babic_extension(fp,r,0.,r1,r1+d,order)*chi
            Ux[i,j] = cafes.babic_extension(fux,r,0.,r1,r1+d,order)*chi
            Uy[i,j] = cafes.babic_extension(fuy,r,0.,r1,r1+d,order)*chi
            Pcpp[i,j], Uxcpp[i,j], Uycpp[i,j] = cafes.get_Babic_field_extension(sing,p1,cafes.cpp.position(x[i,j],y[i,j]))
        elif (x[i,j]-x2)**2 + (y[i,j]-y2)**2<r2**2:
            r = np.sqrt((x2-x[i,j])**2 + (y2-y[i,j])**2)
            theta = np.arctan2(y[i,j]-y2,x[i,j]-x2)
            fp = lambda r: sing.get_p_sing(cafes.cpp.position(x2 + r*np.cos(theta), y2 + r*np.sin(theta)))
            fux = lambda r: sing.get_u_sing(cafes.cpp.position(x2 + r*np.cos(theta), y2 + r*np.sin(theta)))[0]
            fuy = lambda r: sing.get_u_sing(cafes.cpp.position(x2 + r*np.cos(theta), y2 + r*np.sin(theta)))[1]
            chi = (1-cafes.cpp.extchiTrunc(r,l,eps))
            P[i,j] = cafes.babic_extension(fp,r,0.,r2,r2+d,order)*chi
            Ux[i,j] = cafes.babic_extension(fux,r,0.,r2,r2+d,order)*chi
            Uy[i,j] = cafes.babic_extension(fuy,r,0.,r2,r2+d,order)*chi
            Pcpp[i,j], Uxcpp[i,j], Uycpp[i,j] = cafes.get_Babic_field_extension(sing,p2,cafes.cpp.position(x[i,j],y[i,j]))

plt.figure(0)
plt.axis('equal')
plt.contourf(x,y,P-Pcpp,20)
plt.plot(surf1[:,0], surf1[:,1])
plt.plot(surf2[:,0], surf2[:,1])
plt.colorbar()

plt.figure(1)
plt.axis('equal')
plt.contourf(x,y,Ux-Uxcpp,20)
plt.plot(surf1[:,0], surf1[:,1])
plt.plot(surf2[:,0], surf2[:,1])
plt.colorbar()

plt.figure(2)
plt.axis('equal')
plt.contourf(x,y,Uy-Uycpp,20)
plt.plot(surf1[:,0], surf1[:,1])
plt.plot(surf2[:,0], surf2[:,1])
plt.colorbar()

plt.show()