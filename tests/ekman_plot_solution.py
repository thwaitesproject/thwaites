# script for plotting ekman solution. adapted from fluidity test case.
from numpy import sqrt,exp, pi, cos, sin, where

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd

ekman_pd = pd.read_csv("/data/ekman2.5D/2.12.19.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.T10days.dtoutput3600/ekman_profile.csv")
ekman_pd_final = pd.read_csv("/data/ekman2.5D/2.12.19.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.Tfrom_spinup788400.dtoutput60/ekman_profile.csv")
z = ekman_pd['Z_profile']





rho_air=1.3
rho_water=1025.34
C_s=1.4e-3
nu_H=100
nu_V=1.4e-2
u10=0
v10=10
f=1.032e-4
# derived quantities:
BigTau=rho_air/rho_water*C_s*sqrt(u10**2+v10**2)*v10
print('BigTau:', BigTau)
D=pi*sqrt(2*nu_V/f)
print('D:', D)
u0=BigTau*D/sqrt(2.0)/pi/nu_V
print('u0:', u0)

# only look at point (x,y)=(0.0, 0.0)
#ind=where( (abs(xyz[:,0]-50)<1.0) & (abs(xyz[:,1]-50)<1.0) )


u_ex=cos(pi/4.0+pi*z/D)
u_ex=u0*exp(pi*z/D)*cos(pi/4.0+pi*z/D)
v_ex=u0*exp(pi*z/D)*sin(pi/4.0+pi*z/D)


plt.xlabel('U-component / m/s')
plt.ylabel('V-component / m/s')
plt.plot(u_ex, v_ex, '-o')
U = ekman_pd['U_t_219']
V = ekman_pd['V_t_219']
Uf = ekman_pd_final['U_t_51']
Vf = ekman_pd_final['V_t_51']
#plt.plot(U, V, '.k')
plt.plot(Uf, Vf, '.g')
plt.legend(['analytical solution', 'Thwaites result'], loc='upper left')
plt.show()

#n = 24
#for i in range(1, n):
    #t_str = str(i)
    #U = ekman_pd['U_t_' + t_str]
    #V = ekman_pd['V_t_' + t_str]
    #plot(U, V, '.k')

#show()


#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#
#fig, ax = plt.subplots()
#
#line, = ax.plot(u_ex, v_ex)
#line, = ax.plot(ekman_pd['U_t_0.0'], ekman_pd['V_t_0.0'])
#
#plt.xlabel('U-component / m/s')
#plt.ylabel('V-component / m/s')
##plt.legend(['analytical solution', 'Thwaites result'], loc='upper left')
#
#
#def init():  # only required for blitting to give a clean slate.
#    line.set_xdata([np.nan] * len(u_ex))
#    line.set_ydata([np.nan] * len(u_ex))
#    return line,
#
#
#def animate(i):
#    i +=1
#    line.set_xdata(ekman_pd['U_t_' + str(i)])
#    line.set_ydata(ekman_pd['V_t_' + str(i)])  # update the data.
#    return line,
#
#
##ani = animation.FuncAnimation(fig, animate, frames=23, init_func=init, interval=10, blit=True, save_count=10)
#
## To save the animation, use e.g.
##
##ani.save("movie.mp4")
##
## or
##
## from matplotlib.animation import FFMpegWriter
## writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
## ani.save("movie.mp4", writer=writer)
#
#plt.show()