# script for plotting ekman solution. adapted from fluidity test case.
from numpy import sqrt,exp, pi, cos, sin, where

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd

ekman_pd = pd.read_csv("/data/ekman2.5D/19.04.21.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un_ut_0.dtoutput60/ekman_profile.csv") 
#"/data/ekman2.5D/2.12.19.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.T10days.dtoutput3600/ekman_profile.csv")
ekman_pd_final = pd.read_csv("/data/ekman2.5D/19.04.21.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.Tfrom_spinup.dtoutput60/ekman_profile.csv")
z = ekman_pd['Z_profile']

ekman_pd_extruded_direct = pd.read_csv("/data/ekman2.5D/extruded_meshes/20.04.21_2.5d_ekman+_dt60.0_dtOut3600.0_T864000_ip3_Muh1.0_Muv0.014withoutpnullspace_withoutg_mumps/ekman_profile.csv")
ekman_pd_extruded_directmuh100 = pd.read_csv("/data/ekman2.5D/extruded_meshes/20.04.21_2.5d_ekman+_dt60.0_dtOut3600.0_T864000_ip3_Muh100.0_Muv0.014withoutpnullspace_withoutg_mumps/ekman_profile.csv")
ekman_pd_extruded_iterusolve = pd.read_csv("/data/ekman2.5D/extruded_meshes/20.04.21_2.5d_ekman+_dt60.0_dtOut3600.0_T864000_ip3_Muh100.0_Muv0.014usolve_gmres/ekman_profile.csv")
ekman_pd_nog = pd.read_csv("/data/ekman2.5D/20.04.21.ekman2.5D.dt60.T10day.dtout1hour.tau1.775E-4.layers50.ip50.with_coriolis.bottom_unut0._no_g/ekman_profile.csv")
ekman_pd_mumps = pd.read_csv("/data/ekman2.5D/20.04.21.ekman2.5D.dt60.T10day.dtout1hour.tau1.775E-4.layers50.ip50.with_coriolis.bottom_unut0._no_g_defaultip_qdeg10_extramumps/ekman_profile.csv")

ekman_pd_mumps_noicntl = pd.read_csv("/data/ekman2.5D/20.04.21.ekman2.5D.dt60.T10day.dtout1hour.tau1.775E-4.layers50.ip50.with_coriolis.bottom_unut0._no_g_defaultip_qdeg10_extramumps_defatol/ekman_profile.csv")
ekman_pd_extruded_lumping_closed_null = pd.read_csv("/data/ekman2.5D/extruded_meshes/22.04.21_2.5d_ekman+_dt60.0_dtOut3600.0_T864000_ip3_Muh100.0_Muv0.014_iter_usolve_pressurecorriter_closetop/ekman_profile.csv")
ekman3d_pd = pd.read_csv("/data/ekman3D/extruded_meshes/28.04.21_3d_ekman_dt300.0_dtOut3600.0_T864000_ip2_Muh1.0_Muv0.014fromdump7.8days/ekman_profile.csv")

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
U = ekman_pd['U_t_152']
V = ekman_pd['V_t_152']
Unog = ekman_pd_nog['U_t_152']
Vnog = ekman_pd_nog['V_t_152']
Umumps152 = ekman_pd_mumps['U_t_152']
Vmumps152 = ekman_pd_mumps['V_t_152']
Umumps = ekman_pd_mumps['U_t_240']
Vmumps = ekman_pd_mumps['V_t_240']
Umumpsnoicntl = ekman_pd_mumps_noicntl['U_t_240']
Vmumpsnoicntl = ekman_pd_mumps_noicntl['V_t_240']
Uf = ekman_pd_final['U_t_12']
Vf = ekman_pd_final['V_t_12']

Uext = ekman_pd_extruded_direct['U_t_14400']
Vext = ekman_pd_extruded_direct['V_t_14400']
Uextlumpclosednull = ekman_pd_extruded_lumping_closed_null['U_t_14400']
Vextlumpclosednull = ekman_pd_extruded_lumping_closed_null['V_t_14400']
Uextiterusolve = ekman_pd_extruded_iterusolve['U_t_14400']
Vextiterusolve = ekman_pd_extruded_iterusolve['V_t_14400']
Uextmuh100 = ekman_pd_extruded_directmuh100['U_t_14400']
Vextmuh100 = ekman_pd_extruded_directmuh100['V_t_14400']
U3d = ekman3d_pd['U_t_2004']
V3d = ekman3d_pd['V_t_2004']
plt.plot(Uextmuh100, Vextmuh100, '-oy')
plt.plot(U3d, V3d, '.r')
plt.legend(['analytical solution', '2.5d' ,'3d'], loc='upper left')
plt.savefig("28.04.21_ekman3d.png")
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
