# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing stratified according to ocean0 isomip.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain, vertical is 10x weaker.
from thwaites import *
from thwaites.utility import get_top_boundary, cavity_thickness
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from MITgcmutils import mds

##########
firedrake_folder = "/data/2.5d_mitgcm_comparison/06.05.20_3_eq_param_ufricHJ99_dt30.0_dtOutput3600.0_T43200.0_ip50.0_tres86400.0constant_Kh0.001_Kv0.0001_structured_dy50_dz1_no_limiter_closed_no_TS_diric_freeslip_rhs_iterative_pressure_fix_coriolis/"
mitgcm_folder = "/data/mitgcm_runs/ben_FRISP_wout_coriolis_T100.0days_original_Ks_check/run_dt30s_gamma_fric_snapshot_50years_1hour_output/"
firedrake_df = pd.read_hdf(firedrake_folder+"matplotlib_arrays.h5", key="0")

output_folder = "/data/mitgcm_comparison_videos/2.5d_12hours_06.05.20_3_eq_param_ufricHJ99_dt30.0_dtOutput3600.0_T43200.0_ip50.0_tres86400.0constant_Kh0.001_Kv0.0001_structured_dy50_dz1_no_limiter_closed_no_TS_diric_freeslip_rhs_iterative_pressure_fix_coriolis/"
##########

# Get firedrake coordinates
y_array_fd = firedrake_df['y_array']
z_array_fd = firedrake_df['z_array']

# Get mitgcm coordinates
YC = mds.rdmds(mitgcm_folder + 'YC')
RC = mds.rdmds(mitgcm_folder + 'RC')

# number of cells mitgcm run
nx = 1
ny = 200
nz = 102

# Reorganize RC and YC into matrices matching the shape of the T matrix
y_array_mit = np.empty((ny, nz))
z_array_mit = np.empty((ny, nz))

for j in range(nz):
    y_array_mit[:, j] = YC[:, 0]
for j in range(ny):
    z_array_mit[j, :] = RC[:, 0, 0]

##########


triangulation_fd = np.vstack(np.split(np.arange(y_array_fd.shape[0]), y_array_fd.shape[0] / 3))

PLOT_MESH_FD = False

if PLOT_MESH_FD:

    # Plot firedrake mesh 
    fontsize = 11
    fig_ratio = (np.amax(z_array_fd) - np.amin(z_array_fd)) / (np.amax(y_array_fd) - np.amin(y_array_fd))

    print("aspect_ratio", fig_ratio)
    fig = plt.figure(figsize=(20, 10*fig_ratio*20))
    
    plt.triplot(y_array_fd, z_array_fd, triangulation_fd, lw=0.2, color="black", alpha=0.5)
    plt.xlabel("Distance from grounding line / m", fontsize=fontsize)
    plt.ylabel("Depth / m", fontsize=fontsize)
    plt.savefig(output_folder + "firedrake_mesh.png")
    plt.close()

##########

nz = 5
dy = 50.0
ip_factor = Constant(10.)
dt_fd = 30.0
restoring_time = 600.
dt_mit = 30.0
##########

#  Generate mesh
L = 10E3
H1 = 2.
H2 = 100.
#dy = 50.0
ny = round(L/dy)
#nz = 50
dz = H2/nz
# shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
water_depth = 600.0


# momentum source: the buoyancy term Boussinesq approx. From mitgcm default
T_ref = Constant(1.0)
S_ref = Constant(34.4)
beta_temp = Constant(2.0E-4)
beta_sal = Constant(7.4E-4)
g = Constant(9.81)


def top_boundary_to_csv(boundary_points, df, t_str):
    df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
    df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
    df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
    df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
    df['Melt_t' + t_str] = melt.at(boundary_points)
    df['Tb_t_' + t_str] = Tb.at(boundary_points)
    df['P_t_' + t_str] = full_pressure.at(boundary_points)
    df['Sal_t_' + t_str] = sal.at(boundary_points)
    df['Temp_t_' + t_str] = temp.at(boundary_points)
    df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

    if mesh.comm.rank == 0:
        top_boundary_mp.to_csv(folder+"top_boundary_data.csv")

##########

def depth_profile_to_csv(profile, df, depth, t_str):
    #df['U_t_' + t_str] = u.at(profile)
    vw = np.array(v_.at(profile))
    vv = vw[:, 0]
    ww = vw[:, 1]
    df['V_t_' + t_str] = vv
    df['W_t_' + t_str] = ww
    if mesh.comm.rank == 0:
        df.to_csv(folder+depth+"_profile.csv")


def MITgcm_Thwaites_comparison_plot(t, fields, same_colour_scale=True):
    fig = plt.figure(figsize=(20, 10))
    model = ["Firedrake", "MITgcm"]
    ax = []
    ax.append(plt.subplot(2, 1, 1))
    ax.append(plt.subplot(2, 1, 2))
    
    # Add time to figure
    t_hours = t % 24
    t_days = int(t/24)
    ax[0].set_title(str(t_days) + " days " + str(t_hours) + " hours", fontsize=44)

    for i in range(2):
        # Set up labels and axes
        ax.append(plt.subplot(2, 1, i+1))
        ax[i].set_xlim((0.0, 10.0))
        ax[i].set_ylim((-600, -500))
        ax[i].set_xlabel('Distance from grounding line / km', fontsize=22)
        ax[i].set_ylabel('Depth / m', fontsize=22)
        ax[i].text(0.5, -520, model[i], bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10}, fontsize=22)

    if same_colour_scale:
        cplotFD = ax[0].tricontourf(1e-3 * y_array_fd, z_array_fd, triangulation_fd, fields[0], fields[2], cmap='bwr',extend='both')
        cplotMIT = ax[1].contourf(1e-3 * y_array_mit, z_array_mit, fields[1], cplotFD.levels, cmap='bwr',extend='both')
    else:
        cplotFD = ax[0].tricontourf(1e-3 * y_array_fd, z_array_fd, triangulation_fd, fields[0], 30, cmap='coolwarm')
        cplotMIT = ax[1].contourf(1e-3 * y_array_mit, z_array_mit, fields[1], 30, cmap='PRGn')
    
    # Add colour bar
    cbar0 = plt.colorbar(cplotFD, ax=ax[0], shrink=1.0)
    cbar0.set_label(fields[3], fontsize=22)
    cbar1 = plt.colorbar(cplotMIT, ax=ax[1], shrink=1.0)
    cbar1.set_label(fields[3], fontsize=22)
    
    # Save and close figure
    if t < 10:
        plt.savefig(output_folder + fields[4] + "0" + str(t) + "hours.png")
    else:
        plt.savefig(output_folder + fields[4] + str(t) + "hours.png")
    
    plt.close()
    return cplotFD.levels 



# Read in firedrake arrays at 3 hours to use for first 6 hours color levels
u_fd_3hours = firedrake_df['u_array_3hours']
v_fd_3hours = firedrake_df['v_array_3hours']
w_fd_3hours = firedrake_df['w_array_3hours']
vel_mag_fd_3hours = firedrake_df['vel_mag_array_3hours']
temp_fd_3hours = firedrake_df['temp_array_3hours']
sal_fd_3hours = firedrake_df['sal_array_3hours']
rho_fd_3hours = firedrake_df['rho_array_3hours']

# v and w use symmetric scale bar about v =0 m/s
u_levels = np.linspace(-np.abs(u_fd_3hours).max(), np.abs(u_fd_3hours).max(), 25)
v_levels = np.linspace(-np.abs(v_fd_3hours).max(), np.abs(v_fd_3hours).max(), 25)
w_levels = np.linspace(-np.abs(w_fd_3hours).max(), np.abs(w_fd_3hours).max(), 25)
vel_mag_levels = np.linspace(vel_mag_fd_3hours.min(), vel_mag_fd_3hours.max(), 25)
temp_levels = np.linspace(temp_fd_3hours.min(), temp_fd_3hours.max(), 25)
sal_levels = np.linspace(sal_fd_3hours.min(), sal_fd_3hours.max(), 25)
rho_levels = np.linspace(rho_fd_3hours.min(), rho_fd_3hours.max(), 25)


T_end_days = 50
T_end_hours = int(T_end_days * 24)



hb = [-600.0, -600.0]  # bed depth at GL and open boundary

xind = 0  # x-index at which to plot
yind = 49
zind = 14
yind_profiles = [120]#, 80]#, 40]
# aspect_ratio = 0.025
# vector_skip = 4
# vector_scale = 0.001
plot_ice_profile = True
yb = [0.05, 10.0]  # y endpoints (in km)
hi = [-598.0, -500.0]  # ice-ocean boundary depth at GL, open boundary
z_top = -498.0
colors = ['#ADD8E6', '#87A96B', '#FF4F00']
# =============================================================================

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

for t in range(1,13): #T_end_hours+1):
    print('Hour ' + str(t) + ' of ' + str(T_end_hours) + ' hours')
    
    # Get mitgcm results
    iter_mit = int(t * 3600 / dt_mit)
    print(iter_mit)
    diag_U = mds.rdmds(mitgcm_folder + 'diag_U', itrs=iter_mit)
    U = diag_U[0, :, :, :]
    V = diag_U[1, :, :, :]
    W = diag_U[2, :, :, :]
    Umag = (U ** 2 + V ** 2 + W ** 2) ** 0.5
    PhiVel = diag_U[3, :, :]  # Horiz. vel. potential
    PsiVel = diag_U[4, :, :]  # Horiz. vel. streamfunction

    diag_Tracers = mds.rdmds(mitgcm_folder + 'diag_Tracers', itrs=iter_mit)
    Theta = diag_Tracers[0, :, :, :]
    Sal = diag_Tracers[1, :, :, :]

    diag_Shelfice = mds.rdmds(mitgcm_folder + 'diag_Shelfice', itrs=iter_mit)

    SHIfwFlx = diag_Shelfice[0, :, :]
    SHI_Tfrz = diag_Shelfice[11, :, :]
    MR = diag_Shelfice[10, :, :]

    # Create mask for contour plot
    HI = hi[0] + 1e-3 * y_array_mit * (hi[1] - hi[0]) / (yb[1] - yb[0])
    HB = hb[0] + 1e-3 * y_array_mit * (hb[1] - hb[0]) / (yb[1] - yb[0])
    mask = (z_array_mit > HI) + (z_array_mit < HB)  # mask = areas we DON'T plot
    temp_mit = np.ma.array(Theta[:, :, xind].T, mask=mask)
    sal_mit = np.ma.array(Sal[:, :, xind].T, mask=mask)
    u_mit = np.ma.array(U[:, :, xind].T, mask=mask)
    v_mit = np.ma.array(V[:, :, xind].T, mask=mask)
    w_mit = np.ma.array(W[:, :, xind].T, mask=mask)
    vel_mag_mit = np.ma.array(Umag[:,:,xind].T, mask=mask)

    # Read in firedrake arrays
    u_fd = firedrake_df['u_array_'+str(t)+'hours']
    v_fd = firedrake_df['v_array_'+str(t)+'hours']
    w_fd = firedrake_df['w_array_'+str(t)+'hours']
    vel_mag_fd = firedrake_df['vel_mag_array_'+str(t)+'hours']
    temp_fd = firedrake_df['temp_array_'+str(t)+'hours']
    sal_fd = firedrake_df['sal_array_'+str(t)+'hours']
    rho_fd = firedrake_df['rho_array_'+str(t)+'hours']

    if t % 12 ==0:
        # Update colour bar levels
        u_levels = np.linspace(-np.abs(u_fd).max(), np.abs(u_fd).max(), 25)
        v_levels = np.linspace(-np.abs(v_fd).max(), np.abs(v_fd).max(), 25)
        w_levels = np.linspace(-np.abs(w_fd).max(), np.abs(w_fd).max(), 25)
        vel_mag_levels = np.linspace(vel_mag_fd.min(), vel_mag_fd.max(), 25)
        temp_levels = np.linspace(temp_fd.min(), temp_fd.max(), 25)
        sal_levels = np.linspace(sal_fd.min(), sal_fd.max(), 25)
        rho_levels = np.linspace(rho_fd.min(), rho_fd.max(), 25)
    
    # fields for plotting function
    # 0) firedrake array, 1) mitgcm array, 2) Colour bar level, 3) Colour bar label, 4) filename prefix
    temperature_plot = [temp_fd, temp_mit, temp_levels, "Temperature / $^{\circ}$C", "temperature_"]
    salinity_plot = [sal_fd, sal_mit, sal_levels, "Salinity / PSU", "salinity_"]
    u_plot = [u_fd, u_mit, v_levels, "u (zonal) / m/s", "u_velocity_"]
    v_plot = [v_fd, v_mit, v_levels, "v (meridional) / m/s", "v_velocity_"]
    w_plot = [w_fd, w_mit, w_levels, "w (vertical) / m/s", "w_velocity_"]
    mag_vel_plot = [vel_mag_fd, vel_mag_mit, vel_mag_levels, "|u| / m/s", "mag_velocity_"]
    
    # Create comparison plots
    MITgcm_Thwaites_comparison_plot(t, temperature_plot)
    MITgcm_Thwaites_comparison_plot(t, salinity_plot)
    MITgcm_Thwaites_comparison_plot(t, u_plot)
    MITgcm_Thwaites_comparison_plot(t, v_plot)
    MITgcm_Thwaites_comparison_plot(t, w_plot)

'''            
    year_in_seconds = 3600 * 24 * 365.25
    top_boundary_firedrake = pd.read_csv(firedrake_folder+"top_boundary_data.csv")
    n = 100
    dx = 10000. / float(n)
    x1 = np.array([i * dx for i in range(n)])

    # Initialize plotting environment
    fig = plt.figure(figsize=(16, 7))
    ax_vy = plt.subplot()


    #ax_vy.set_xlim((0.0, 8.0))
    #ax_vy.set_ylim((0.0, 7.0))
    ax_vy.set_xlabel(r'Distance from the grounding line (km)', fontsize=22)
    ax_vy.set_ylabel(r'Melt rate (m a$^{-1}$)', fontsize=22)
    ax_vy.plot(1e-3 * YC, MR * year_in_seconds, lw=1.5, label=r'MITgcm')


    ax_vy.plot(1e-3 * x1, top_boundary_firedrake['Melt_t115200'] * year_in_seconds, label=r'Firedrake')

    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 40 days')
    plt.savefig(output_folder + "melt_rate_960_hours.png")
'''
'''
      year_in_seconds = 3600 * 24 * 365.25
      top_boundary_firedrake = pd.read_csv("/data/2d_mitgcm_comparison/24.01.20_3EqParam_dt60.0_dtOutput172800.0_T8640000.0_ip50.0_tres600.0_Kh5.0_Kv2.0openocean_dy50.0_nz5/top_boundary_data.csv")
      profile_4km_firedrake = pd.read_csv(
          "/data/2d_mitgcm_comparison/24.01.20_3EqParam_dt60.0_dtOutput172800.0_T8640000.0_ip50.0_tres600.0_Kh5.0_Kv2.0openocean_dy50.0_nz5/6km_profile.csv")
      n = 100
      dx = 10000. / float(n)
      x1 = np.array([i * dx for i in range(n)])

      # Initialize plotting environment
      fig = plt.figure(figsize=(16, 7))
      ax_vy = plt.subplot()


      #ax_vy.set_xlim((0.0, 8.0))
      #ax_vy.set_ylim((0.0, 7.0))
      ax_vy.set_xlabel(r'$y$ (km)', fontsize=22)
      ax_vy.set_ylabel(r'Melt rate (m a$^{-1}$)', fontsize=22)
      ax_vy.plot(1e-3 * YC, MR * year_in_seconds, lw=1.5, label=r'MITgcm')


      ax_vy.plot(1e-3 * x1, top_boundary_firedrake['Melt_t95040'] * year_in_seconds, label=r'Firedrake')

      plt.legend()
      plt.title('Melt rate along ice shelf boundary after 60 days')
      #plt.savefig(output_folder + "fig1_melt_after_30days_dt864.png")

      # Initialize plotting environment
      fig = plt.figure(figsize=(16, 7))
      ax_vy = plt.subplot()
      # ax_vy.set_xlim((0.0, 8.0))
      # ax_vy.set_ylim((0.0, 7.0))
      ax_vy.set_xlabel(r'Meridional velocity / m/s ', fontsize=22)
      ax_vy.set_ylabel(r'Depth / m', fontsize=22)

      bathymetry = profile_4km_firedrake['Z_profile'][49]
      ice_depth  = profile_4km_firedrake['Z_profile'][0]
      print("bathy:", bathymetry)
      print("ice bathy:", ice_depth)

      cav_height = ice_depth - bathymetry
      hfrac_firedrake = (profile_4km_firedrake['Z_profile'] - bathymetry)/cav_height

      ax_vy.plot(profile_4km_firedrake['V_t_95040'], hfrac_firedrake, 'bo',label=r'Firedrake')
      ax_vy.plot(V[mask, 120, 0], hfrac[mask], 'o-', label='MITgcm')
      ax_vy.set_ylim(0,1)
      plt.legend()
      plt.title('Meridonal velocity 6km from gl after 60 days ')
      plt.savefig('meridonal_velocity6km_fromGL18days.png')

     # plt.show()



'''
