from sympy import *

x, y, t, kappa, = symbols('x y t kappa')
a = Integral(cos(x)*exp(x),x)
print(Eq(a,a.doit()))

print("")

q_an = sin(25*x*y) - 2*y / sqrt(x)
print(q_an)

qan_diffx = diff(q_an, x)
print("dx: ", qan_diffx) 

print("-----------")
u = sin(5*(x*x+y*y))
v = cos(3*(x*x-y*y))

print("div vel", diff(u,x) + diff(v,y))
print("div vel", simplify(diff(u,x) + diff(v,y)))

T =  sin(25*x*y) - 2*y / sqrt(x)
S =  30 + cos(25*x*y) - 2*y / sqrt(x)

T_source = diff(T, t) + u * diff(T, x) + v * diff(T, y)  - kappa * (diff(T, x, 2)  + diff(T, y, 2))
S_source = diff(S, t) + u * diff(S, x) + v * diff(S, y)  - kappa * (diff(S, x, 2)  + diff(S, y, 2))
print("T_source:", T_source)
print("S_source:", S_source)

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
