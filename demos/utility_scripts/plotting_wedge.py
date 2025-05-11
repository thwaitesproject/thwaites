from thwaites import *
from math import pi
import numpy as np
from firedrake.petsc import PETSc
from firedrake import FacetNormal

#from firedrake import *
L = 5000


dz = 5#float(L)/float(nz)
mesh = Mesh("/data/thwaites/meshes/mesh/ice_shelf_cavity_mesh.msh")#ice_shelf_mesh5.msh") #SquareMesh(nx, nz, L) #Mesh("ice_shelf_mesh3.msh")#


import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

file_location1 = '/data/thwaites/9.12.18.iSWING_3_eq_param_dt_864_t120_delT_3.0_reduced_mu_kappa/'


#sal = Function(V, name="A")

#temp = Function(V)

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V,W])


K = FunctionSpace(mesh,"DG",1)    # temperature space
S = FunctionSpace(mesh,"DG",1)    # salinity space

#M = FunctionSpace(mesh,"DG",1)      # melt space - dont really need but might be easier for plotting in paraview


m = Function(M)
u_, p_ = m.subfunctions


temp = Function(K, name="Temperature")
sal = Function(S, name="Salinity")



# Normal code here
with DumbCheckpoint(file_location1+"dump.h5", mode=FILE_READ) as chk:
    # Checkpoint file open for reading and writing

    chk.load(temp)
    chk.load(sal)
    chk.load(u_,name="Velocity")

# Checkpoint file closed, continue with normal code

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

try:
    plot(temp,contour=True)
    plt.title("Temperature at time ...")
    plt.show()
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

