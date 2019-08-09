# script for calculating height of ice that will give a pressure that leads to zero melt! 
# just a quadratic equaion

from math import sqrt
g = 9.81  # gravitational acceleration

Tice = -25.0  # temp of far field ice
Tm = -0.5  # boundary layer temp


rho_ice = 920.  # density of ice
cI = 2009.  # specific heat capacity mixed layer
kappaI = 1.14E-6  # thermal diffusivity of ice

rho_water=1025
cW = 3974
thermal_exchange_vel = 1E-4


I = rho_ice*cI*kappaI
W = rho_water*cW*thermal_exchange_vel


## Tb = a_tSb + b_t + c_tP
Sb = 34.5

a_t = -5.73E-2  # salinity coefficient of freezing eqation

b_t = 9.39E-2
c_t = -7.53E-8


#ax^2 + bx +c
a = W*c_t*rho_ice*g

b = I*c_t*rho_ice*g + W*a_t*Sb + W*b_t - W*Tm

c = I*a_t*Sb + I*b_t - I*Tice

print("b^2-4ac = ",b**2 -4.*a*c)
print(sqrt(b**2 -4.*a*c))
h1 = (-b + sqrt(b**2 -4.*a*c))/(2.*a)
h2 = (-b - sqrt(b**2 -4.*a*c))/(2.*a)
print("The solutions are ", h1 ,"and", h2)

