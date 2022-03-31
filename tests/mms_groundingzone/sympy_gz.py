from sympy import *
import numpy as np
x, y, t, kappa, mu_h, mu_v, p, H2, depth, L, g, beta_temp, beta_sal, T_ref, S_ref = symbols('x y t kappa mu_h mu_v p H2 depth L g beta_temp beta_sal T_ref S_ref')

print("-----------")

arg = - np.pi / H2 * (y + depth - H2)
u =  x / L * cos(arg)
v = (H2 / np.pi) * sin(arg) / L
#u =  pow(x,2) / L**2 * cos(arg)
#v = 2*x *(H2 / np.pi) * sin(arg) / L**2
#p = cos(np.pi / L * (x + L)) - cos(- np.pi / H2 * (y + depth - H2))
#p = x**2 / (2*L) * sin(arg)  + H2 / (2 * np.pi * L) * y
#u = sin(np.pi * x / (2*L)) * sin(-2 * np.pi / H2 * (y + depth - H2)) 
#v = - np.pi/(2*L) * cos(np.pi * x / (2*L)) * (H2/(2*np.pi) *cos(-2 * np.pi / H2 * (y + depth - H2)) - H2/(2*np.pi)) 
#p = -2 *L / np.pi * cos(np.pi * x / (2*L)) * (H2/(2*np.pi)*sin(-2 * np.pi / H2 * (y + depth - H2))-y*H2/(2*np.pi))
#pint = x / L * sin(arg) * sin(-2 * np.pi / H2 * (y + depth - H2)) 
#p = pint -  sin(-2 * np.pi / H2 * (y + depth - H2)) * sin(-2 * np.pi / H2 * (y + depth - H2))

p = cos(np.pi * x / L) * cos(arg) #cos(arg) * cos(np.pi * x / L)

#p = 2.0 
print("dudx", diff(u,x) )
print("dvdy", diff(v,y) )
print("div vel", diff(u,x) + diff(v,y))
print("div vel", simplify(diff(u,x) + diff(v,y)))

print("dpdx", diff(p,x) )
print("dpdy", diff(p,y) )


print("u at -900m (x=0): ", u.subs([(depth, 1000), (H2, 100), (y, -900), (L,100), (x, 0)]))
print("u at -925m (x=0): ", u.subs([(depth, 1000), (H2, 100), (y, -925), (L,100), (x, 0)]))
print("u at -950m (x=0): ", u.subs([(depth, 1000), (H2, 100), (y, -950), (L,100), (x, 0)]))
print("u at -975m (x=0): ", u.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 0)]))
print("u at -1000m (x=0): ", u.subs([(depth, 1000), (H2, 100), (y, -1000), (L,100), (x, 0)]))

print("u at -900m (x=50): ", u.subs([(depth, 1000), (H2, 100), (y, -900), (L,100), (x, 50)]))
print("u at -925m (x=50): ", u.subs([(depth, 1000), (H2, 100), (y, -925), (L,100), (x, 50)]))
print("u at -950m (x=50): ", u.subs([(depth, 1000), (H2, 100), (y, -950), (L,100), (x, 50)]))
print("u at -975m (x=50): ", u.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 50)]))
print("u at -1000m (x=50): ", u.subs([(depth, 1000), (H2, 100), (y, -1000), (L,100), (x, 50)]))

print("u at -900m (x=100): ", u.subs([(depth, 1000), (H2, 100), (y, -900), (L,100), (x, 100)]))
print("u at -925m (x=100): ", u.subs([(depth, 1000), (H2, 100), (y, -925), (L,100), (x, 100)]))
print("u at -950m (x=100): ", u.subs([(depth, 1000), (H2, 100), (y, -950), (L,100), (x, 100)]))
print("u at -975m (x=100): ", u.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 100)]))
print("u at -1000m (x=100): ", u.subs([(depth, 1000), (H2, 100), (y, -1000), (L,100), (x, 100)]))

print("v at -900m: ", v.subs([(depth, 1000), (H2, 100), (y, -900), (L,100)]))
print("v at -950m: ", v.subs([(depth, 1000), (H2, 100), (y, -950), (L,100)]))
print("v at -1000m: ", v.subs([(depth, 1000), (H2, 100), (y, -1000), (L,100)]))

print("p at -925m (x=25): ", p.subs([(depth, 1000), (H2, 100), (y, -925), (L,100), (x, 25)]))
print("p at -925m (x=75): ", p.subs([(depth, 1000), (H2, 100), (y, -925), (L,100), (x, 75)]))
print("p at -975m (x=25): ", p.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 25)]))
print("p at -975m (x=75): ", p.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 75)]))
print("p at -975m (x=100): ", p.subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 100)]))


print("dp/dx at (x=0): ", diff(p,x).subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 0)]))
print("dp/dx at (x=100): ", diff(p,x).subs([(depth, 1000), (H2, 100), (y, -975), (L,100), (x, 100)]))


print("dp/dy at (y=-1000): ", diff(p,y).subs([(depth, 1000), (H2, 100), (y, 1000), (L,100), (x, 0)]))
print("dp/dy at (y=-900): ", diff(p,y).subs([(depth, 1000), (H2, 100), (y, -900), (L,100), (x, 100)]))

# y = -900 T = -0.5
# y = -910 T = 0.0
# y = -1000 T = 1
depths = [-900, -910, -1000]
temperature = [-0.5, 0.0, 1]
p_temperature = np.polyfit(depths, temperature, 2) 
print("p = ", p_temperature)

salinity = [33.8,34.0, 34.5]
p_salt = np.polyfit(depths, salinity, 2) 
print("p salt = ", p_salt)
print(type(p_salt[0]))


T =  0.1*sin(4*np.pi*x/L) + p_temperature[0]*Pow(y,2) + p_temperature[1]*y + p_temperature[2]
S = 0.01* 34.5 * cos(4*np.pi*x/L)  +  p_salt[0]*Pow(y,2) + p_salt[1]*y + p_salt[2]



print("temp at -975m (x=100): ", T.subs([(depth, 1000), (H2, 100), (y, -900), (L,100), (x, 100)]))

u_source = diff(u, t) + u * diff(u, x) + v * diff(u, y)  - mu_h * diff(u, x, 2)  - mu_v * diff(u, y, 2)  + diff(p, x)
v_source = diff(v, t) + u * diff(v, x) + v * diff(v, y)  - mu_h * diff(v, x, 2)  - mu_v * diff(v, y, 2) + diff(p,y) #+ g*(-beta_temp*(T - T_ref) + beta_sal * (S - S_ref)) #+ diff(p, y)

print("u_source:", u_source)
print("v_source:", v_source)


T_source = diff(T, t) + u * diff(T, x) + v * diff(T, y)  - kappa * (diff(T, x, 2)  + diff(T, y, 2))
S_source = diff(S, t) + u * diff(S, x) + v * diff(S, y)  - kappa * (diff(S, x, 2)  + diff(S, y, 2))
print("T_source:", T_source)
print("S_source:", S_source)
print()


#print("old sources for just temp/sal")


#T =  sin(25*x*y) - 2*y / sqrt(x)
#S =  30 + cos(25*x*y) - 2*y / sqrt(x)
#
#T_source = diff(T, t) + u * diff(T, x) + v * diff(T, y)  - kappa * (diff(T, x, 2)  + diff(T, y, 2))
#S_source = diff(S, t) + u * diff(S, x) + v * diff(S, y)  - kappa * (diff(S, x, 2)  + diff(S, y, 2))
#print("T_source:", T_source)
#print("S_source:", S_source)

print("-----------")

a = -5.73E-2  # salinity coefficient of freezing equation
b = 8.32E-2  # constant coefficient of freezing equation
c = -7.53E-8  # pressure coefficient of freezing equation

c_p_m = 3974.  # specific heat capacity of mixed layer, J / kg /K
c_p_i = 2000.  # Specific heat capacity of ice, J/kg/K

gammaT = 1.0E-4  # thermal exchange velocity, m/s
gammaS = 5.05E-7  # salt exchange velocity, m/s

GammaT = 1.1E-2  # dimensionless GammaT (capital). gammaT = GammaT.u* from jenkins et al 2010
GammaS = GammaT / 35.0  # dimensionless GammaS (capital). gammaS = GammaS.u* from jenkins et al 2010
C_d = 2.5E-3  # Drag coefficient for ice, dimensionless

Lf = 3.34E5  # Latent heat of fusion of ice, J/kg

k_i = 1.14E-6  # Molecular thermal conductivity of ice, m^2/s

rho_ice = 920.0  # Reference density of ice, kg/m^3
T_ice = -20.0  # Temperature of far field ice, degC

g = 9.81  # gravitational acceleration, m/s^2
rho0 = 1028.  # Reference density of sea water, kg/m^3

Pr = 13.8  # Prandtl number, dimnesionless
Sc = 2432  # Schmidt number, dimensionless
zeta_N = 0.052  # Stability constant, dimensionless
eta_star = 1.0  # Stability parameter, dimensionless. This is default in MTIgcm. Should this be
# N.b MITgcm sets etastar = 1 so that the first part of Gamma_turb term below is constant.
# eta_star = (1 + (zeta_N * u_star) / (f * L0 * Rc))^-1/2  (Eq 18)
# In H&J99  eta_star is set to 1 when the Obukhov length is negative (i.e the buoyancy flux
# is destabilising. This occurs during freezing. (Melting is stabilising)
# So it does seem a bit odd because then eta_star is tuned to freezing conditions
# need to work this out...?
k = 0.4  # Von Karman's constant, dimensionless
nu = 1.95e-6  # Kinematic viscosity of seawater, m^2/s

P_hydrostatic = -rho0 * g * y
P_full = P_hydrostatic
def twoeqmelt():
            
    S = 35.0
    Tb = a * S + b + c * P_full

    Q_ice = 0.0

    Q_mixed = -rho0 * c_p_m * gammaT * (Tb - T)
    Q_latent = Q_ice - Q_mixed
    wb = -Q_latent / (Lf * rho0)
    T_flux_bc = -(wb + gammaT) * (Tb - T)


    print("Tfluxbc", T_flux_bc)


#freezing point Tb = a * Sb + b + c * P_full
# cons salt  QsI - QsM = 0
# cons heat QtI - QtM = Qlatent

#freezing_point = Tb - ( a * Sb + b + c * P_full)
#heat_conservation = wb *Lf - c_p_m*gammaT*(T - Tb) # without heat flow into ice = 0
#salt_conservation = wb*Sb - gammaS*(S-Sb)
#print(nonlinsolve([freezing_point, heat_conservation, salt_conservation], (Tb, wb, Sb)))

from thwaites import ThreeEqMeltRateParam

#mp = ThreeEqMeltRateParam(S.evalf(subs={x: 0.6, y:-0.1}), T.evalf(subs={x: 0.6, y:-0.1}), 0, -0.1) #, velocity=pow(dot(v, v) + pow(u, 2), 0.5), f=f)
mp = ThreeEqMeltRateParam(S, T, 0, y) #, velocity=pow(dot(v, v) + pow(u, 2), 0.5), f=f)

print("Tfluxbc \n",mp.T_flux_bc)
print("\nSfluxbc \n",mp.S_flux_bc)
print("\nMelt \n", mp.wb)


u1 = sin(x*x + y*y) 
v1 = cos(x*x + y*y)


print("div vel", diff(u1,x) + diff(v1,y))
