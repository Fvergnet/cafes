from context import cafes
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 4.5*x**2 + 2.+x -6. + np.cos(x)

def df(x):
    return 3*x**2 - 9*x + 2. - np.sin(x)

a=-2.
b=5.
c=7.

N = 100
x1 = np.linspace(a,b,N)
x2 = np.linspace(b,c,N)

# for f
g0 = np.zeros(x1.size)
g1 = np.zeros(x1.size)
g2 = np.zeros(x1.size)
g0cpp = g0.copy()
g1cpp = g0.copy()
g2cpp = g0.copy()

# For df
dg0 = np.zeros(x1.size)
dg1 = np.zeros(x1.size)
dg2 = np.zeros(x1.size)
dg0cpp = g0.copy()
dg1cpp = g0.copy()
dg2cpp = g0.copy()

# For derivative of extension of f
deg0 = np.zeros(x1.size)
deg1 = np.zeros(x1.size)
deg2 = np.zeros(x1.size)
deg0cpp = g0.copy()
deg1cpp = g0.copy()
deg2cpp = g0.copy()

for i in range(g0.size):
    # For f
    g0[i] = cafes.babic_extension(f,x1[i],a,b,c,0)
    g1[i] = cafes.babic_extension(f,x1[i],a,b,c,1)
    g2[i] = cafes.babic_extension(f,x1[i],a,b,c,2)
    g0cpp[i] = cafes.cpp_babic_extension(f,x1[i],a,b,c,0)
    g1cpp[i] = cafes.cpp_babic_extension(f,x1[i],a,b,c,1)
    g2cpp[i] = cafes.cpp_babic_extension(f,x1[i],a,b,c,2)

    # For df
    dg0[i] = cafes.babic_extension(df,x1[i],a,b,c,0)
    dg1[i] = cafes.babic_extension(df,x1[i],a,b,c,1)
    dg2[i] = cafes.babic_extension(df,x1[i],a,b,c,2)
    dg0cpp[i] = cafes.cpp_babic_extension(df,x1[i],a,b,c,0)
    dg1cpp[i] = cafes.cpp_babic_extension(df,x1[i],a,b,c,1)
    dg2cpp[i] = cafes.cpp_babic_extension(df,x1[i],a,b,c,2)

    # For d ext(f)
    deg0[i] = cafes.babic_grad_extension(df,x1[i],a,b,c,0)
    deg1[i] = cafes.babic_grad_extension(df,x1[i],a,b,c,1)
    deg2[i] = cafes.babic_grad_extension(df,x1[i],a,b,c,2)
    # deg0cpp[i] = cafes.cpp_grad_babic_extension(df,x1[i],a,b,c,0)
    # deg1cpp[i] = cafes.cpp_grad_babic_extension(df,x1[i],a,b,c,1)
    # deg2cpp[i] = cafes.cpp_grad_babic_extension(df,x1[i],a,b,c,2)

plt.figure()
plt.plot(x2,f(x2),'r')
plt.plot(x1,g0,label="extension order 0")
plt.plot(x1,g1,label="extension order 1")
plt.plot(x1,g2,label="extension order 2")
plt.plot(x1,g0cpp,'+', label="cpp extension order 0")
plt.plot(x1,g1cpp,'+', label="cpp extension order 1")
plt.plot(x1,g2cpp,'+', label="cpp extension order 2")
plt.legend()
plt.title("extension of f")

plt.figure()
plt.plot(x2,df(x2),'r')
plt.plot(x1,dg0,label="extension order 0")
plt.plot(x1,dg1,label="extension order 1")
plt.plot(x1,dg2,label="extension order 2")
plt.plot(x1,dg0cpp,'+', label="cpp extension order 0")
plt.plot(x1,dg1cpp,'+', label="cpp extension order 1")
plt.plot(x1,dg2cpp,'+', label="cpp extension order 2")
plt.legend()
plt.title("extension of df")

plt.figure()
plt.plot(x2,df(x2),'r')
plt.plot(x1,deg0,label="extension order 0")
plt.plot(x1,deg1,label="extension order 1")
plt.plot(x1,deg2,label="extension order 2")
# plt.plot(x1,deg0cpp,'+', label="cpp extension order 0")
# plt.plot(x1,deg1cpp,'+', label="cpp extension order 1")
# plt.plot(x1,deg2cpp,'+', label="cpp extension order 2")
plt.legend()
plt.title("derivative of extension of f")

plt.show()
