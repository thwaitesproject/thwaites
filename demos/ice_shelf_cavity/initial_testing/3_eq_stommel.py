# Buoyancy driven overturning circulation
# implement a bc which is equivalent to ocean - ice boundary simplified heat flow in ice
# i.e no advection but do have diffusion of heat into ice so Q^T_I = -rho_icpK(TI-Tb)/h
# Two equation formulation so assume instataneous diffusion of salinity. -> working towards three equation
# T_b = aS_b + b + cP  # salinity/pressure dependent melting point of water.
# w_b = T_m - (aS_b+b-cP).C_p_m . gamma_T / L_f  # melt rate of water see end of notebook 1. (Holland and Jenkins 1999)
#
from thwaites import *
from math import pi
import numpy as np
from firedrake.petsc import PETSc
from firedrake import FacetNormal


L = 5000 
nx = 100
nz = 100

dz = 5#float(L)/float(nz)
mesh = Mesh("/data/will/thwaites/meshes/mesh.msh")#ice_shelf_mesh5.msh") #SquareMesh(nx, nz, L) #Mesh("ice_shelf_mesh3.msh")#


import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)



#mesh = Mesh("./meshes/square_mesh_test3.msh")
# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space
K = FunctionSpace(mesh,"DG",1)    # temperature space
S = FunctionSpace(mesh,"DG",1)    # salinity space

#M = FunctionSpace(mesh,"DG",1)      # melt space - dont really need but might be easier for plotting in paraview


m = Function(M)
u_, p_ = m.split()
rho = Function(Q)

temp = Function(K)
sal = Function(S)

melt = Function(Q)
Q_mixed = Function(Q)
Q_ice = Function(Q)
Q_latent = Function(Q)
Tb = Function(Q)

full_pressure = Function(M.sub(1))



# We set up the initial velocity field using a simple analytic expression. ::


def ice_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

def cavity_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

x,z = SpatialCoordinate(mesh)

h_ice = ice_thickness(x,0.0,999.0,5000.0,900.0)
h_cav = cavity_thickness(x,0.0,1.0,5000.0,100.0)

z = z -h_cav

u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa_temp = Constant(1)
kappa_sal = Constant(1)
#mu = Constant(1e-3)
mu = Constant(1)  # kinematic



temp_init = Constant(0.37)   # Tinit =Tres deg C from ben's thesis page 54
temp.interpolate(temp_init)


sal_init = Constant(34.4) # stationay flow S = Srest#set this to slightly less than Sres. horizontal pressure gradient increases as depth increases. pressure greater outside domain so drives flow in!
sal.interpolate(sal_init)

melt_init = Constant(0)
melt.interpolate(melt_init)

Q_mixed_init = Constant(0)
Q_mixed.interpolate(Q_mixed_init)

Q_ice_init = Constant(0)
Q_ice.interpolate(Q_ice_init)

Q_latent_init = Constant(0)
Q_latent.interpolate(Q_latent_init)

Tb_init = Constant(0)
Tb.interpolate(Tb_init)



full_pressure_init = Constant(0)
full_pressure.interpolate(full_pressure_init)

folder = "/data/will/thwaites/27.11.18.ice_shelf_cavity_circulation_3_eq_param1/"
# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)
d_file = File(folder+"density.pvd")
d_file.write(rho)
t_file = File(folder+"temperature.pvd")
t_file.write(temp)
s_file = File(folder+"salinity.pvd")
s_file.write(sal)
m_file = File(folder+"melt.pvd")
m_file.write(melt)

Q_mixed_file = File(folder+"Qmixed.pvd")
Q_mixed_file.write(Q_mixed)

full_pressure_file = File(folder+"full_pressure.pvd")
full_pressure_file.write(full_pressure)



# time period and time step
T = 3600*24*30
dt = 100

#T = 5.
#dt = T/5000.




u_test, p_test = TestFunctions(M)


# i think stephan changed this from test to testspace in the last git pull...
mom_eq = MomentumEquation(M.sub(0), M.sub(0))


cty_eq = ContinuityEquation(M.sub(1), M.sub(1))


#rho_test = TestFunction(Q)
#rho_eq = ScalarAdvectionDiffusionEquation(temp_test, Q)

temp_test = TestFunction(K)
#temp_eq = ScalarAdvectionDiffusionEquation(temp_test, K)
temp_eq = ScalarAdvectionDiffusionEquation(K, K)


sal_test = TestFunction(S)
#sal_eq = ScalarAdvectionDiffusionEquation(sal_test, S)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)


u, p = split(m)


#From Ben's thesis page 31

Tref= 0.0
Sref=34.8 #PSU
beta_temp = 3.87*10E-5 #5.0E-3
beta_sal=7.86*10E-4
g = 9.81
mom_source = as_vector((0, -g))*(-beta_temp*(temp-Tref)+beta_sal*(sal-Sref))  # momentum source: the buoyancy term boussinesq approx


rho0=1025  # Holland and jenkins 1997) used later to recalculate the full pressure for meltrate without hydrostatic term



# sourcefor scalar equations at open boundary. linearly relaxes to T/Srestore
absorption_factor =2.0E-4

#Trestore = 0.37  # degC  n.b delta T = 3degC  (ben's thesis page
Trestore = 0.37#0.37

Srestore = 34.5 #PSUi

interp_source_temp = Function(K)
interp_source_sal = Function(S)

source_temp = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04))*Trestore, 0.0)
source_sal = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04))*Srestore, 0.0)

interp_source_temp.interpolate(source_temp)
source_temp_file = File(folder+"source_temp.pvd")
source_temp_file.write(interp_source_temp)


interp_source_sal.interpolate(source_sal)

#x_vector, y_vector = interpolate(x[0], Function(DG_2d)).dat.data, interpolate(x[1], Function(DG_2d)).dat.data

#absorption term
interp_absorp_temp = Function(K)
interp_absorp_sal = Function(S)

absorp_temp = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04)), 0.0)
absorp_sal = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04)), 0.0)

interp_absorp_temp.interpolate(absorp_temp)

absorb_temp_file = File(folder+"absorb_temp.pvd")
absorb_temp_file.write(interp_absorp_temp)

interp_absorp_sal.interpolate(absorp_sal)


Q_s = Function(K)
Q_s_init = Constant(0.0)
Q_s.interpolate(Q_s_init)

Qs_file = File(folder+"Q_s.pvd")
Qs_file.write(Q_s)


up_fields = {'viscosity': mu, 'source': mom_source}
#rho_fields = {'diffusivity': kappa, 'velocity': u}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'source': interp_source_temp, 'absorption coefficient': interp_absorp_temp}

sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'source': interp_source_sal, 'absorption coefficient': interp_absorp_sal}

mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_atol': 2.5e-6
}


def two_eq_param_meltrate(S, T, Pin):
    a = -5.73E-2  # salinity coefficient of freezing eqation
    b = 9.39E-2  # constant coeff of freezing equation
    c = -7.53E-8  # pressure coeff of freezing equation

    c_p_m = 3974.  # specific heat capacity of mixed layer
    c_p_i = 2009.
    gammaT = 1E-5  # roughly thermal exchange velocity
    Lf = 3.34E5  # latent heat of fusion

    k_i = 1.14E-6
    # Sb = np.array(S.at(coordinates))
    # Tm = np.array(T.at(coordinates))
    # P = np.array(Pin.at(coordinates))

    rho_ice = 920.0
    # h_ice = 1000. # in m
    T_ice = -25.0

    P_ice = rho_ice * g * h_ice  # hydrostatic pressure just from ice
    Pfull = rho0 * (Pin - g * z) + P_ice  # # this is just true for top boundary - as z =0 need Pfull = rho0 (Pin-gz)+P_ice

    dz_calc = dz * 1.0  # 5.0

    Tb = conditional(z > 0 - dz_calc, a * S + b + c * Pfull, 0.0)
    # Q_ice = conditional(z > 0-dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick!
    Q_ice = Constant(0.0)
    Q_mixed = conditional(z > 0 - dz_calc, -rho0 * (Tb - T) * c_p_m * gammaT, 0.0)
    Q_latent = conditional(z > 0 - dz_calc, Q_ice - Q_mixed, 0.0)
    wb = conditional(z > 0 - dz_calc, -Q_latent / (Lf * rho0), 0.0)

    Q_mixed_bc = conditional(z > 0 - dz_calc, -(wb + gammaT) * (Tb - T),
                             0.0)  # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
    Q_s = wb * S

    return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s, wb, Tb, Pfull  # these are all still expressions

def three_eq_param_meltrate_without_Qice(S, T, Pin):
    # solve with 3 equation param and Qice = 0 

    a = -5.73E-2  # salinity coefficient of freezing eqation
    b = 9.39E-2   # constant coeff of freezing equation
    c = -7.53E-8    # pressure coeff of freezing equation

    c_p_m = 3974. # specific heat capacity of mixed layer
    c_p_i = 2009.
    gammaT = 1E-4 # roughly thermal exchange velocity
    gammaS = 5.05E-7
    Lf = 3.34E5  # latent heat of fusion

    k_i = 1.14E-6
    #Sb = np.array(S.at(coordinates))
    #Tm = np.array(T.at(coordinates))
    #P = np.array(Pin.at(coordinates))

    rho_ice = 920.0
    #h_ice = 1000. # in m
    T_ice = -25.0


    P_ice = rho_ice*g*h_ice  # hydrostatic pressure just from ice
    Pfull= rho0*(Pin - g*z)+P_ice#

    dz_calc = dz*1.0#5.0
    
    Aa = c_p_m*gammaT*a

    Bb = c_p_m*gammaT*b - c_p_m*gammaT*T + c_p_m*c*Pfull*gammaT - gammaS*Lf

    Cc = gammaS*Lf*S

    soln1 = (-Bb + pow(Bb**2 - 4.0*Aa*Cc,0.5))/(2.0*Aa)   # this is one value of loc_Sb
    soln2 = (-Bb - pow(Bb**2 - 4.0*Aa*Cc,0.5))/(2.0*Aa)

    loc_Sb = conditional(soln1 > 0.0, soln1, soln2)

    '''try:
        p = int(typ[1:])
        if p < 1:
            raise ValueError
    except ValueError:
        raise ValueError("Don't know how to interpret %s-norm" % norm_type)'''

    # need to add in an exception if the salinity falls below zero. ...
    '''error = conditional(if i in loc_Sb.dat.data < 0.0:
        print("Melt interface, loc_Sb: ",  i)
        print("Melt interface, Aa: ",  Aa)
        print("Melt interface, Bb: ",  Bb)
        print("Melt interface, Cc: ",  Cc)
        print("Melt interface, T: ",  T)
        print("Melt interface, S: ",  S)
        print("Melt interface, P: ",  Pfull)
        print("Melt interface, fv: n/a")  # add in friction velocity

        raise Exception("Melt interface, Sb is negative. The range of Salinity is not right.")'''




    loc_Tb = conditional(z > 0-dz_calc, a*loc_Sb + b + c*Pfull,0.0)
    #Q_ice = conditional(z > 0-dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick! 
    Q_ice = Constant(0.0)
    Q_mixed = conditional(z > 0-dz_calc,-rho0*(loc_Tb-T)*c_p_m*gammaT,0.0)
    Q_latent = conditional(z > 0-dz_calc, Q_ice-Q_mixed,0.0)
    wb = conditional(z > 0-dz_calc, -Q_latent/(Lf*rho0), 0.0)

    Q_mixed_bc = conditional(z > 0 - dz_calc, -(wb+gammaT)*(loc_Tb - T), 0.0) # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
    Q_s_bc = conditional(z > 0 - dz_calc, -(wb + gammaS) * (loc_Sb - S), 0.0)

    return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s_bc, wb, Tb, Pfull  # these are all still expressions


def three_eq_param_meltrate(S, T, Pin):
    # solve with three equation param and diffusion of heat through ice

    a = -5.73E-2  # salinity coefficient of freezing eqation
    b = 9.39E-2  # constant coeff of freezing equation
    c = -7.53E-8  # pressure coeff of freezing equation

    c_p_m = 3974.  # specific heat capacity of mixed layer
    c_p_i = 2009.
    gammaT = 1E-4  # roughly thermal exchange velocity
    gammaS = 5.05E-7
    Lf = 3.34E5  # latent heat of fusion

    k_i = 1.14E-6


    rho_ice = 920.0
    # h_ice = 1000. # in m
    T_ice = -25.0

    P_ice = rho_ice * g * h_ice  # hydrostatic pressure just from ice
    Pfull = rho0 * (Pin - g * z) + P_ice  #

    dz_calc = dz * 1.0  # 5.0

    Aa = c_p_m*gammaT*a - c_p_i*gammaS*a

    Bb = c_p_i*gammaS*T_ice - c_p_i*gammaS*(b + c*Pfull) + c_p_i*gammaS*S*a
    Bb = Bb + c_p_m*gammaT*(b + c*Pfull) - c_p_m*gammaT*T - gammaS*Lf

    Cc = c_p_i*gammaS*S*(b+c*Pfull) - c_p_m*gammaS*S*T_ice + gammaS*Lf*S

    soln1 = (-Bb + pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)  # this is one value of loc_Sb
    soln2 = (-Bb - pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)

    loc_Sb = conditional(soln1 > 0.0, soln1, soln2)

    '''try:
        p = int(typ[1:])
        if p < 1:
            raise ValueError
    except ValueError:
        raise ValueError("Don't know how to interpret %s-norm" % norm_type)'''

    # need to add in an exception if the salinity falls below zero. ...
    '''error = conditional(if i in loc_Sb.dat.data < 0.0:
        print("Melt interface, loc_Sb: ",  i)
        print("Melt interface, Aa: ",  Aa)
        print("Melt interface, Bb: ",  Bb)
        print("Melt interface, Cc: ",  Cc)
        print("Melt interface, T: ",  T)
        print("Melt interface, S: ",  S)
        print("Melt interface, P: ",  Pfull)
        print("Melt interface, fv: n/a")  # add in friction velocity

        raise Exception("Melt interface, Sb is negative. The range of Salinity is not right.")'''

    loc_Tb = conditional(z > 0 - dz_calc, a * loc_Sb + b + c * Pfull, 0.0)
    # Q_ice = conditional(z > 0-dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick!
       
    wb = conditional(z > 0 - dz_calc, -gammaS * (loc_Sb - S) / loc_Sb, 0.0)
    Q_mixed = conditional(z > 0 - dz_calc, -rho0 * (loc_Tb - T) * c_p_m * gammaT, 0.0)
    Q_ice = conditional(z > 0 - dz_calc, -rho0 * (T_ice - loc_Tb) * c_p_i * wb, 0.0)
    
    Q_latent = conditional(z > 0 - dz_calc, Q_ice - Q_mixed, 0.0)
    




    Q_mixed_bc = conditional(z > 0 - dz_calc, -(wb + gammaT) * (loc_Tb - T),
                             0.0)  # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
    Q_s_bc = conditional(z > 0 - dz_calc, -(wb + gammaS) * (loc_Sb - S), 0.0)

    return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s_bc, wb, Tb, Pfull  # these are all still expressions



q_ice, q_mixed, q_mixed_real, q_latent, q_s, wb, t_b, pfull = three_eq_param_meltrate(sal, temp, p_)






# assign values of these expressions to functions. each time update the only things that change are T and S so these alter the expression and give new value for functions.
Q_ice.interpolate(q_ice)
Q_mixed.interpolate(q_mixed)

Q_latent.interpolate(q_latent)
Q_s.interpolate(q_s)
melt.interpolate(wb)
Tb.interpolate(t_b)
full_pressure.interpolate(pfull)

# this is where write first file for above functions.
m_file = File(folder+"melt.pvd")
m_file.write(melt)

Q_mixed_file = File(folder+"Qmixed.pvd")
Q_mixed_file.write(Q_mixed)

full_pressure_file = File(folder+"full_pressure.pvd")
full_pressure_file.write(full_pressure)

Qs_file = File(folder+"Q_s.pvd")
Qs_file.write(Q_s)

Q_ice_file = File(folder+"Q_ice.pvd")
Q_ice_file.write(Q_ice)

temp_bcs = {2: {'q': Trestore}, 1: {'flux': -Q_mixed}}
temp_solver_parameters = mumps_solver_parameters


sal_bcs ={2: {'q': Srestore}, 1: {'flux': -Q_s}} # {1: {'q': sal_ice}}
sal_solver_parameters = mumps_solver_parameters


n = 100
dx = 5000. / float(n)
x1 = np.array([i * dx for i in range(n)])

#y = ice_thickness(x1, 0.0, 999.0, 5000.0, 900.0)
shelf_boundary_points = []
for i in range(n):
    x_i = i * dx
    y_i = cavity_thickness(x_i, 0.0, 1.0, 5000.0, 100.0) - 0.01
    shelf_boundary_points.append([x_i, y_i])

shelf_boundary_points = np.array(shelf_boundary_points)

q_vector = Q_mixed.at(shelf_boundary_points)
if mesh.comm.rank ==0:
    print(q_vector)
    print(len(q_vector))
    import matplotlib.pyplot as plt
    plt.plot(x1,q_vector)
    plt.savefig("q_mixed0.png")
    #Q_ice_vector = Q_ice([shelf_boundary_points])
    #Q_latent_vector = Q_latent([shelf_boundary_points])
    #Tb_vector = Tb([shelf_boundary_points])
    #Melt_vector = melt([shelf_boundary_points])

'''print("Shelf boundary \n")
print(shelf_boundary_points)
print(Q_ice_vector)
print(len(Q_ice_vector))


print("rank 0 ")

print()
if mesh.comm.rank ==0:
    print("Shelf boundary \n")
    print(shelf_boundary_points)
    print(Q_ice_vector)
    print(len(Q_ice_vector))
'''

no_normal_flow = {'un': 0.}

n = FacetNormal(mesh)
stress_open_boundary = {'stress': -n*-g*(-beta_temp*(Trestore-Tref)+beta_sal*(Srestore-Sref))*z}  # p = -g*(alpha(delT)+beta(delS))*z and -n because pressure into the domain! nah its the way stephan defines sigma...

#up_bcs = {1: no_normal_flow, 2: {}, 3: no_normal_flow, 4: no_normal_flow}  # case1: don't specify stress on rhs boundary
up_bcs = {1: no_normal_flow, 2: stress_open_boundary, 3: no_normal_flow, 4: no_normal_flow}

up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)




t = 0.0
step = 0

output_dt = 100# 3600*24
output_step = output_dt/dt

pressure_ice = 920.*9.81*h_ice

T=dt*10
while t < T - 0.5*dt:


    up_timestepper.advance(t)
    #rho_timestepper.advance(t)
    temp_timestepper.advance(t)
    sal_timestepper.advance(t)
    # find melt rate...


    melt.interpolate(wb)
    Q_mixed.interpolate(q_mixed)
    Q_ice.interpolate(q_ice)
    Q_latent.interpolate(q_latent)
    Tb.interpolate(t_b)
    full_pressure.interpolate(pfull)
    Q_s.interpolate(q_s)
    



    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        #p_file.write(p_)
        #d_file.write(rho)
        t_file.write(temp)
        s_file.write(sal)
        m_file.write(melt)
        Q_mixed_file.write(Q_mixed)
        absorb_temp_file.write(interp_absorp_temp)
        source_temp_file.write(interp_source_temp)
        full_pressure_file.write(full_pressure)
        Qs_file.write(Q_s)
        Q_ice_file.write(Q_ice)
        PETSc.Sys.Print("t=", t)

'''
n = 100
dx = 5000./float(n)
x1 = np.array([i*dx for i in range(n)])

y =ice_thickness(x,0.0,999.0,5000.0,900.0)
shelf_boundary_points =[]
for i in range(n):
    x = i*dx
    y = cavity_thickness(x,0.0,1.0,5000.0,100.0)-0.01
    shelf_boundary_points.append([x,y]) 
if mesh.comm.rank ==0:
    Q_mixed_vector = Q_mixed([points])
    Q_ice_vector = Q_ice([
    Q_latent_vector = Q_latent.dat.data
    Tb_vector = Tb.dat.data
    Melt_vector = melt([points])
     


'''

'''

DG_2d = FunctionSpace(mesh, 'DG', 1)
#x = SpatialCoordinate(mesh2d)
x_vector, z_vector = interpolate(x, Function(DG_2d)).dat.data, interpolate(z, Function(DG_2d)).dat.data
print("x",x_vector)

print("z",z_vector)

Q_mixed_vector = Q_mixed.dat.data
Q_ice_vector = Q_ice.dat.data
Q_latent_vector = Q_latent.dat.data
Tb_vector = Tb.dat.data
Melt_vector = melt.dat.data
Ice_height_vector = interpolate(h0+m_h*x, Function(DG_2d)).dat.data

x_top =[]
Q_mixed_top=[]
Q_ice_top =[]
Q_latent_top=[]
Tb_top =[]
Melt_top = []
Ice_height_top =[]

for i in range(len(z_vector)):
    if z_vector[i] ==0:
        x_top.append(float(x_vector[i]))
        Q_mixed_top.append(float(Q_mixed_vector[i]))
        Q_ice_top.append(float(Q_ice_vector[i]))
        Q_latent_top.append(float(Q_latent_vector[i]))
        Tb_top.append(float(Tb_vector[i]))
        Melt_top.append(float(Melt_vector[i]))
        Ice_height_top.append(float(Ice_height_vector[i]))

print(len(x_top))
print(len(Q_mixed_top))

from firedrake import plot

try:
  import matplotlib.pyplot as plt
  import numpy as np
except:
  warning("Matplotlib not imported")

try:
    x_top=np.array(x_top)
    Q_mixed_top=np.array(Q_mixed_top)*1025.*3974
    
    plt.figure(1)
    
    plt.subplot(3,1,1)
    plt.plot(x_top,Ice_height_top)
    plt.ylabel("Ice thickness /m")

    plt.subplot(3,1,2)
    plt.plot(x_top,Q_mixed_top,label="Qm")
    plt.plot(x_top,Q_ice_top,label="Qi")
    plt.plot(x_top,Q_latent_top,label="Qlat=Qi-Qm")
    plt.legend()
    #plt.ylabel("Heat flux through the top boundary / W/m^2")
    
    plt.grid() 
    plt.ylabel("Heat flux through top boundary / W/m^2")
    
    plt.subplot(3,1,3)
    plt.plot(x_top,np.array(Melt_top)*(3600*24.*365),label="wb = -Qlat/(rho_m*Lf)")
    plt.legend()
    plt.ylabel("Melt rate / m/yr")
    plt.xlabel("Distance along top boundary / m")
    plt.grid()

    plt.figure(2)
    plt.grid()
    plt.plot(x_top,Q_mixed_top)
  
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)


try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
 ''' 
