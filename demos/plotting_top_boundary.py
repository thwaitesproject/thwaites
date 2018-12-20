import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

output_folder = "/data/thwaites/output_plots_04.12/"
n = 100
dx = 5000. / float(n)
x1 = np.array([i * dx for i in range(n)])

# dt = 864, delT = 3.0, meltrate param constant gamma - no fric vel. constant diffusivity.
df_dt_864_delT_3 = pd.read_csv("/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_1/top_boundary_data.csv")
df_dt_432_delT_3 = pd.read_csv("/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_432_1/top_boundary_data.csv")
df_dt_216_delT_3 = pd.read_csv("/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_216_1/top_boundary_data.csv") # atm only goes to t=25days - will go to 30!!
df_dt_864_delT_3_lin_mu = pd.read_csv("/data/thwaites/3.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_3.0_linear_mu/top_boundary_data.csv")

df_dt_864_delT_3_lin_mu_fric_vel = pd.read_csv("/data/thwaites/3.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_3.0_linear_mu_w.fric_vel/top_boundary_data.csv")
df_dt_864_delT_3_fric_vel = pd.read_csv("/data/thwaites/2.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_3.0_w.fric_vel/top_boundary_data.csv")
df_dt_864_delT_1 = pd.read_csv("/data/thwaites/3.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_1.0/top_boundary_data.csv")
df_dt_864_delT_point1 = pd.read_csv("/data/thwaites/3.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_0.1/top_boundary_data.csv")



#print(df_dt_864_delT_3)
figsize=(20,10)
year_in_seconds = 3600*24*365.25
plot_1 = False
if plot_1:
    # plot Meltrate after 30 days for dt =864, delT =3.0
    plt.figure(1,figsize=figsize)
    plt.plot(x1,df_dt_864_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 30 days')
    plt.savefig(output_folder+"fig1_melt_after_30days_dt864.png")

    # plot Meltrate after 30 days for dt =432 and dt = 864, (delT =3.0)
    plt.figure(2,figsize=figsize)
    plt.plot(x1,df_dt_432_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 432s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 30 days')
    plt.savefig(output_folder+"fig2_melt_after_30days_dt864_dt432.png")

    plt.figure(3,figsize=figsize)
    plt.plot(x1,df_dt_432_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 432s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0,20])
    plt.ylim([1.96,1.98])
    plt.title('Melt rate along ice shelf boundary after 30 days close to GL')
    plt.savefig(output_folder+"fig3_melt_after_30days_dt864_dt432_nearGL.png")

    plt.figure(4,figsize=figsize)
    plt.plot(x1, df_dt_432_delT_3['Melt_t1.0'] * year_in_seconds, label=r'dt = 432s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_3['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C ')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([15.8, 16.2])
    plt.title('Melt rate along ice shelf boundary after 1 day close to GL')
    plt.savefig(output_folder+"fig4_melt_after_1day_dt864_dt432_nearGL.png")


# plot convergence of melt for dt 864 and dt 432 in subplot
plot_2 = False
if plot_2:
    plt.figure(5,figsize=figsize)

    for i in range(1,7):
        t_str = str(5*i)
        plt.subplot(1, 2, 1)
        plt.plot(x1, df_dt_864_delT_3['Melt_t'+t_str+'.0'] * year_in_seconds, label=('t = '+t_str+r' days, dt = 864s, $\Delta$T = 3.0$\degree$C'))
        plt.xlabel('Distance from Grounding Line / m')
        plt.ylabel('Melt rate / m/yr')
        plt.legend()
        plt.title('Convergence of melt rate along ice shelf boundary up to 30 days for dt =864s')

        # subplot for dt = 432
        plt.subplot(1, 2, 2)
        plt.plot(x1, df_dt_432_delT_3['Melt_t' + t_str + '.0'] * year_in_seconds,
                 label=('t = ' + t_str + r' days, dt = 432s, $\Delta$T = 3.0$\degree$C'))
        plt.xlabel('Distance from Grounding Line / m')
        plt.ylabel('Melt rate / m/yr')
        plt.legend()
        plt.title('Convergence of melt rate along ice shelf boundary up to 30 days for dt = 432s')
        plt.savefig(output_folder+"fig5_convergence_melt_upto_30days_dt864_dt432.png")



plot_4 = False
# error in dt 864 c.f dt 432
if plot_4:
    n = 31
    dt = 1.0
    t1 = np.array([dt * i for i in range(1, n)])

    plt.figure(6,figsize=figsize)
    melt_error=[]
    percent_melt_error=[]
    for i in range(1,31):
        t_str = str(i)
        dt_864_time_i = df_dt_864_delT_3['Melt_t' + t_str + '.0'].tolist()[0]
        dt_432_time_i = df_dt_432_delT_3['Melt_t' + t_str + '.0'].tolist()[0]

        err = abs((dt_864_time_i-dt_432_time_i))
        percent_err = (err/dt_432_time_i)*100.0
        melt_error.append(err*year_in_seconds)
        percent_melt_error.append(percent_err)
    plt.subplot(1,2,1)
    plt.plot(t1, melt_error, label=(r'error = |dt_864-dt_432| ($\Delta$T = 3.0$\degree$C)'))
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Error Melt rate / m/yr')
    plt.legend()
    plt.title('Absolute Error in Meltrate at GL for dt = 864s')


    plt.subplot(1, 2, 2)
    plt.plot(t1, percent_melt_error, label=('% error ($\Delta$T = 3.0$\degree$C)'))
    plt.xlabel('Simulation Time / days')
    plt.ylabel('(Error Melt rate / Melt_dt432) * 100 %')
    plt.legend()
    plt.title('% Error in Meltrate at GL for dt = 864s')
    plt.savefig(output_folder + "fig6_melt_error_dt864_dt432_atGL.png")

plot_5=False

if plot_5:
    plt.figure(7,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_3_lin_mu['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$')
    plt.plot(x1, df_dt_864_delT_3['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 30 days with linear change in viscosity across the domain')
    plt.savefig(output_folder + "fig7_melt_after_30days_dt864_cf_linear_mu_nearGL.png")
#plt.show()





df_salinity_dt864_delT3_lin_mu=pd.read_csv("/data/thwaites/3.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_3.0_linear_mu/salinity_data_frame.csv")
df_salinity_dt864_delT3=pd.read_csv("/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_1/salinity_data_frame.csv")
df_salinity_dt432_delT3=pd.read_csv("/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_432_1/salinity_data_frame.csv")

plot_6=False
#plot salinity
if plot_6:
    plt.figure(8,figsize=figsize)
    plt.plot(df_salinity_dt864_delT3_lin_mu['x_top_boundary_t'+sal_str_t],salinity_dataframe_top_linear_mu['sal_t' + sal_str_t])
    plt.xlabel('Distance from GL / m')
    plt.ylabel('Salinity / PSU')





plot_7 = False
# plot convergence for salinity for dt=864, mu linearly varies over domain. (at the moment up to 90 days - not reached steady state)
if plot_7:
    t2 = [(i+1)*5 for i in range(18)]
    plt.figure(9, figsize=figsize)
    sal_gl = []
    for j in range(18):
        sal_str_t = str((j + 1) * 5)
        sal_time_i = df_salinity_dt864_delT3_lin_mu['sal_t' + sal_str_t].tolist()[0]
        sal_gl.append(sal_time_i)
    plt.plot(t2,sal_gl,'bx')
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Salinity / PSU')
    plt.title('Convergence of Salinity along ice shelf boundary up to 50 days (with spatially varying viscosity)')
    plt.savefig(output_folder + "fig9_convergence_sal_upto_90days_dt864_linear_mu.png")



plot_8=False
#plot salinity linear mu vs dt 864
if plot_8:
    plt.figure(10,figsize=figsize)
    plt.plot(df_salinity_dt864_delT3_lin_mu['x_top_boundary_t30'],df_salinity_dt864_delT3_lin_mu['sal_t30'],
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$')
    plt.plot(df_salinity_dt864_delT3['x_top_boundary_t30'], df_salinity_dt864_delT3['sal_t30'], label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.xlabel('Distance from GL / m')
    plt.ylabel('Salinity / PSU')
    plt.legend()
    plt.title("Salinity along top boundary at t = 30 days, comparing constant vs linear viscosity")
    plt.savefig(output_folder + "fig10_sal_top_t30days_dt864_constant_cf_linear_mu.png")


plot_9=False
#plot salinity dt432 vs dt 864 at t =30days over the domain
if plot_9:
    plt.figure(11,figsize=figsize)
    plt.plot(df_salinity_dt432_delT3['x_top_boundary_t30'],df_salinity_dt432_delT3['sal_t30'],label=r'dt = 432s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.plot(df_salinity_dt864_delT3['x_top_boundary_t30'], df_salinity_dt864_delT3['sal_t30'], label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.xlabel('Distance from GL / m')
    plt.ylabel('Salinity / PSU')
    plt.legend()
    plt.title("Salinity along top boundary at t = 30 days, comparing dt = 864s and dt = 432s")
    plt.savefig(output_folder + "fig11_sal_top_t30days_dt864_cf_dt432.png")


plot_10=False
#plot salinity dt432 vs dt 864  at t = 30days near the grounding line
if plot_10:
    plt.figure(12,figsize=figsize)
    plt.plot(df_salinity_dt432_delT3['x_top_boundary_t30'],df_salinity_dt432_delT3['sal_t30'],label=r'dt = 432s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.plot(df_salinity_dt864_delT3['x_top_boundary_t30'], df_salinity_dt864_delT3['sal_t30'], label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.xlabel('Distance from GL / m')
    plt.ylabel('Salinity / PSU')
    plt.xlim([0,20])
    plt.ylim([33.837, 33.838])
    plt.legend()
    plt.title("Salinity along top boundary at t = 30 days, comparing dt = 864s and dt = 432s near GL")
    plt.savefig(output_folder + "fig12_sal_top_t30days_dt864_cf_dt432_nearGL.png")

##########################

# these plots below are still running - redo when runs finished!!


plot_11=False
if plot_11:
# plot Meltrate after 30 days for dt =432 and dt = 864, (delT =3.0)
    plt.figure(13,figsize=figsize)
    plt.plot(x1,df_dt_216_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 216s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1,df_dt_432_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 432s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 30 days')
    plt.savefig(output_folder+"fig13_melt_after_30days_dt864_dt432_dt216.png")

    plt.figure(14,figsize=figsize)
    plt.plot(x1, df_dt_216_delT_3['Melt_t30.0'] * year_in_seconds, label=r'dt = 216s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_432_delT_3['Melt_t30.0'] * year_in_seconds, label=r'dt = 432s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_3['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0,20])
    plt.ylim([1.96,1.98])
    plt.title('Melt rate along ice shelf boundary after 30 days close to GL')
    plt.savefig(output_folder+"fig14_melt_after_30days_dt864_dt432_dt216_nearGL.png")



# plot convergence of melt for dt 864, dt 432 and dt 216 in subplot up to t = 30
plot_12 = False
if plot_12:
    plt.figure(15,figsize=figsize)

    for i in range(1,7):
        t_str = str(5*i)
        plt.subplot(1, 3, 1)
        plt.plot(x1, df_dt_864_delT_3['Melt_t'+t_str+'.0'] * year_in_seconds, label=('t = '+t_str+r' days, dt = 864s, $\Delta$T = 3.0$\degree$C'))
        plt.xlabel('Distance from Grounding Line / m')
        plt.ylabel('Melt rate / m/yr')
        plt.legend()
        #plt.title('Convergence of melt rate along ice shelf boundary up to 30 days for dt =864s')

        # subplot for dt = 432
        plt.subplot(1, 3, 2)
        plt.plot(x1, df_dt_432_delT_3['Melt_t' + t_str + '.0'] * year_in_seconds,
                 label=('t = ' + t_str + r' days, dt = 432s, $\Delta$T = 3.0$\degree$C'))
        plt.xlabel('Distance from Grounding Line / m')
        plt.ylabel('Melt rate / m/yr')
        plt.legend()
        plt.title('Convergence of melt rate along ice shelf boundary up to 30 days for dt = 864s, dt = 432s and dt = 216s')
        plt.savefig(output_folder+"fig5_convergence_melt_upto_30days_dt864_dt432.png")

        plt.subplot(1, 3, 3)
        plt.plot(x1, df_dt_216_delT_3['Melt_t' + t_str + '.0'] * year_in_seconds,
                 label=('t = ' + t_str + r' days, dt = 216s, $\Delta$T = 3.0$\degree$C'))
        plt.xlabel('Distance from Grounding Line / m')
        plt.ylabel('Melt rate / m/yr')
        plt.legend()
        #plt.title('Convergence of melt rate along ice shelf boundary up to 30 days for dt = 216s')
        plt.savefig(output_folder + "fig15_convergence_melt_upto_30days_dt864_dt432_dt216.png")



plot_13 = False
# error in dt 864 c.f dt 432
if plot_13:
    n = 31
    dt = 1.0
    t1 = np.array([dt * i for i in range(1, 31)])

    plt.figure(16,figsize=figsize)
    melt_error_432=[]
    melt_error_864=[]
    percent_melt_error_432=[]
    percent_melt_error_864=[]
    for i in range(1,31):
        t_str = str(i)
        dt_864_time_i = df_dt_864_delT_3['Melt_t' + t_str + '.0'].tolist()[0]
        dt_432_time_i = df_dt_432_delT_3['Melt_t' + t_str + '.0'].tolist()[0]
        dt_216_time_i = df_dt_216_delT_3['Melt_t' + t_str + '.0'].tolist()[0]

        err_864 = abs((dt_864_time_i-dt_216_time_i))
        percent_err_864 = (err_864/dt_216_time_i)*100.0
        melt_error_864.append(err_864*year_in_seconds)
        percent_melt_error_864.append(percent_err_864)

        err_432 = abs((dt_432_time_i - dt_216_time_i))
        percent_err_432 = (err_432 / dt_216_time_i) * 100.0
        melt_error_432.append(err_432 * year_in_seconds)
        percent_melt_error_432.append(percent_err_432)

    plt.subplot(1,2,1)
    plt.plot(t1, melt_error_432, label=(r'error = |dt_432-dt_216| ($\Delta$T = 3.0$\degree$C)'))
    plt.plot(t1, melt_error_864, label=(r'error = |dt_864-dt_216| ($\Delta$T = 3.0$\degree$C)'))
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Error Melt rate / m/yr')
    plt.legend()
    plt.title('Absolute Error in Meltrate at GL for dt = 864s and dt = 432s')


    plt.subplot(1, 2, 2)
    plt.plot(t1, percent_melt_error_432, label=('% error for dt = 432s ($\Delta$T = 3.0$\degree$C)'))
    plt.plot(t1, percent_melt_error_864, label=('% error for dt = 864s ($\Delta$T = 3.0$\degree$C)'))
    plt.xlabel('Simulation Time / days')
    plt.ylabel(r'(Error Melt rate / Melt$_{dt = 216}$) * 100 %')
    plt.legend()
    plt.title('% Error in Meltrate at GL for dt = 864s and dt = 432s')
    plt.savefig(output_folder + "fig16_melt_error_dt864_dt432_atGL.png")

#plt.figure()
#plt.plot(x1,x1**0.5-x1**2)


# reload new delT3 which goes on to 120...
df_dt_864_delT_3 = pd.read_csv("/data/thwaites/2.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_t120_delT_3.0/top_boundary_data.csv")
# plot captures of meltrate for different delT. atm delT =0.1 only goes to 50days!!!
plot_14=False
if plot_14:

    deltaT = [0.1,1.0,3.0]
    melt_t10x_50 = np.array([df_dt_864_delT_point1['Melt_t10.0'].tolist()[1],
                    df_dt_864_delT_1['Melt_t10.0'].tolist()[1],
                    df_dt_864_delT_3['Melt_t10.0'].tolist()[1]])

    melt_t10x_100 = np.array([df_dt_864_delT_point1['Melt_t10.0'].tolist()[2],
                    df_dt_864_delT_1['Melt_t10.0'].tolist()[2],
                    df_dt_864_delT_3['Melt_t10.0'].tolist()[2]])



    melt_t10x_1000 = np.array([df_dt_864_delT_point1['Melt_t10.0'].tolist()[20],
                     df_dt_864_delT_1['Melt_t10.0'].tolist()[20],
                     df_dt_864_delT_3['Melt_t10.0'].tolist()[20]])

    melt_t10x_2000 = np.array([df_dt_864_delT_point1['Melt_t10.0'].tolist()[40],
                     df_dt_864_delT_1['Melt_t10.0'].tolist()[40],
                     df_dt_864_delT_3['Melt_t10.0'].tolist()[40]])

    melt_t10x_4000 = np.array([df_dt_864_delT_point1['Melt_t10.0'].tolist()[80],
                      df_dt_864_delT_1['Melt_t10.0'].tolist()[80],
                      df_dt_864_delT_3['Melt_t10.0'].tolist()[80]])

    melt_t30x_50 = np.array([df_dt_864_delT_point1['Melt_t30.0'].tolist()[1],
                             df_dt_864_delT_1['Melt_t30.0'].tolist()[1],
                             df_dt_864_delT_3['Melt_t30.0'].tolist()[1]])

    melt_t30x_100 = np.array([df_dt_864_delT_point1['Melt_t30.0'].tolist()[2],
                              df_dt_864_delT_1['Melt_t30.0'].tolist()[2],
                              df_dt_864_delT_3['Melt_t30.0'].tolist()[2]])

    melt_t30x_1000 = np.array([df_dt_864_delT_point1['Melt_t30.0'].tolist()[20],
                               df_dt_864_delT_1['Melt_t30.0'].tolist()[20],
                               df_dt_864_delT_3['Melt_t30.0'].tolist()[20]])

    melt_t30x_2000 = np.array([df_dt_864_delT_point1['Melt_t30.0'].tolist()[40],
                               df_dt_864_delT_1['Melt_t30.0'].tolist()[40],
                               df_dt_864_delT_3['Melt_t30.0'].tolist()[40]])

    melt_t30x_4000 = np.array([df_dt_864_delT_point1['Melt_t30.0'].tolist()[80],
                               df_dt_864_delT_1['Melt_t30.0'].tolist()[80],
                               df_dt_864_delT_3['Melt_t30.0'].tolist()[80]])


    melt_t40x_50 = np.array([df_dt_864_delT_point1['Melt_t40.0'].tolist()[1],
                             df_dt_864_delT_1['Melt_t40.0'].tolist()[1],
                             df_dt_864_delT_3['Melt_t40.0'].tolist()[1]])

    melt_t40x_100 = np.array([df_dt_864_delT_point1['Melt_t40.0'].tolist()[2],
                              df_dt_864_delT_1['Melt_t40.0'].tolist()[2],
                              df_dt_864_delT_3['Melt_t40.0'].tolist()[2]])

    melt_t40x_1000 = np.array([df_dt_864_delT_point1['Melt_t40.0'].tolist()[20],
                               df_dt_864_delT_1['Melt_t40.0'].tolist()[20],
                               df_dt_864_delT_3['Melt_t40.0'].tolist()[20]])

    melt_t40x_2000 = np.array([df_dt_864_delT_point1['Melt_t40.0'].tolist()[40],
                               df_dt_864_delT_1['Melt_t40.0'].tolist()[40],
                               df_dt_864_delT_3['Melt_t40.0'].tolist()[40]])

    melt_t40x_4000 = np.array([df_dt_864_delT_point1['Melt_t40.0'].tolist()[80],
                               df_dt_864_delT_1['Melt_t40.0'].tolist()[80],
                               df_dt_864_delT_3['Melt_t40.0'].tolist()[80]])

    melt_t50x_50 = np.array([df_dt_864_delT_point1['Melt_t50.0'].tolist()[1],
                             df_dt_864_delT_1['Melt_t50.0'].tolist()[1],
                             df_dt_864_delT_3['Melt_t50.0'].tolist()[1]])

    melt_t50x_100 = np.array([df_dt_864_delT_point1['Melt_t50.0'].tolist()[2],
                              df_dt_864_delT_1['Melt_t50.0'].tolist()[2],
                              df_dt_864_delT_3['Melt_t50.0'].tolist()[2]])

    melt_t50x_1000 = np.array([df_dt_864_delT_point1['Melt_t50.0'].tolist()[20],
                               df_dt_864_delT_1['Melt_t50.0'].tolist()[20],
                               df_dt_864_delT_3['Melt_t50.0'].tolist()[20]])

    melt_t50x_2000 = np.array([df_dt_864_delT_point1['Melt_t50.0'].tolist()[40],
                               df_dt_864_delT_1['Melt_t50.0'].tolist()[40],
                               df_dt_864_delT_3['Melt_t50.0'].tolist()[40]])

    melt_t50x_4000 = np.array([df_dt_864_delT_point1['Melt_t50.0'].tolist()[80],
                               df_dt_864_delT_1['Melt_t50.0'].tolist()[80],
                               df_dt_864_delT_3['Melt_t50.0'].tolist()[80]])








    plt.figure(17,figsize=figsize)
    plt.suptitle(r"Varying Meltrate forcing $\Delta$ T")
    plt.subplot(2,4,1)
    plt.plot(x1, df_dt_864_delT_3['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_1['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_point1['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate after 30 days / m/yr')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(x1, df_dt_864_delT_3['Melt_t40.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_1['Melt_t40.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_point1['Melt_t40.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate after 40 days / m/yr')
    plt.legend()

    plt.subplot(2, 4, 3)

    plt.plot(x1, df_dt_864_delT_3['Melt_t50.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_1['Melt_t50.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_point1['Melt_t50.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate after 50 days / m/yr')
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.plot(x1, df_dt_864_delT_3['Melt_t60.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_1['Melt_t60.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 1.0$\degree$C',color='orange')
    # plt.plot(x1, df_dt_864_delT_point1['Melt_t40.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate after 60 days/ m/yr')
    plt.legend()

    plt.subplot(2,4,5)
    plt.plot(deltaT,melt_t30x_50 *year_in_seconds,'-x',label='x = 50m')
    plt.plot(deltaT, melt_t30x_100* year_in_seconds,'-x',label='x = 100m')
    plt.plot(deltaT, melt_t30x_1000 * year_in_seconds,'-x',label='x = 1000m')
    plt.plot(deltaT, melt_t30x_2000 * year_in_seconds, '-x',label='x = 2000m')
    plt.plot(deltaT, melt_t30x_4000 * year_in_seconds, '-x',label='x = 4000m',)
    plt.xlabel(r'$\Delta$ T')
    plt.ylabel('Melt rate after 30 days / m/yr')
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.plot(deltaT, melt_t40x_50 * year_in_seconds, '-x', label='x = 50m')
    plt.plot(deltaT, melt_t40x_100 * year_in_seconds, '-x', label='x = 100m')
    plt.plot(deltaT, melt_t40x_1000 * year_in_seconds, '-x', label='x = 1000m')
    plt.plot(deltaT, melt_t40x_2000 * year_in_seconds, '-x', label='x = 2000m')
    plt.plot(deltaT, melt_t40x_4000 * year_in_seconds, '-x', label='x = 4000m', )
    plt.xlabel(r'$\Delta$ T')
    plt.ylabel('Melt rate after 30 days / m/yr')
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.plot(deltaT, melt_t50x_50 * year_in_seconds, '-x', label='x = 50m')
    plt.plot(deltaT, melt_t50x_100 * year_in_seconds, '-x', label='x = 100m')
    plt.plot(deltaT, melt_t50x_1000 * year_in_seconds, '-x', label='x = 1000m')
    plt.plot(deltaT, melt_t50x_2000 * year_in_seconds, '-x', label='x = 2000m')
    plt.plot(deltaT, melt_t50x_4000 * year_in_seconds, '-x', label='x = 4000m' )
    plt.xlabel(r'$\Delta$ T')
    plt.ylabel('Melt rate after 50 days / m/yr')
    plt.legend()
    plt.savefig(output_folder+"fig17melt_rate_diff_delT_forcing.png")


plot_15 = False
# plot convergence for melt for dt=864, for different temp forcings . (at the moment up to 50 days - not reached steady state)
if plot_15:
    t2 = [(i+1)*10 for i in range(5)]
    plt.figure(18, figsize=figsize)
    melt_gl_delTpoint1 = []
    melt_gl_delT1 = []
    melt_gl_delT3= []
    for j in range(5):
        melt_str_t = str((j + 1) * 10)
        melt_time_i_delTpoint1 = df_dt_864_delT_point1['Melt_t' + melt_str_t+'.0'].tolist()[0]
        melt_time_i_delT1 = df_dt_864_delT_1['Melt_t' + melt_str_t+'.0'].tolist()[0]
        melt_time_i_delT3 = df_dt_864_delT_3['Melt_t' + melt_str_t+'.0'].tolist()[0]

        melt_gl_delTpoint1.append(melt_time_i_delTpoint1)
        melt_gl_delT1.append(melt_time_i_delT1)
        melt_gl_delT3.append(melt_time_i_delT3)

    melt_gl_delTpoint1 = np.array(melt_gl_delTpoint1)
    melt_gl_delT1 = np.array(melt_gl_delT1)
    melt_gl_delT3 = np.array(melt_gl_delT3)

    plt.plot(t2,melt_gl_delT3 * year_in_seconds,'-x',label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.plot(t2, melt_gl_delT1 * year_in_seconds, '-x',label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(t2, melt_gl_delTpoint1  * year_in_seconds, '-x',label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')

    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at GL / m/yr')
    plt.title(r'Convergence of Melt rate at GL up to 50 days for different $\Delta$T forcings')
    plt.savefig(output_folder + "fig18_convergence_melt_upto_50days_for_delT_forcings.png")


plot_16=False

if plot_16:
    plt.figure(19,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_3['Melt_t60.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$')
    plt.plot(x1, df_dt_864_delT_3_lin_mu['Melt_t60.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 60 days with linear change in viscosity across the domain')
    plt.savefig(output_folder + "fig19_melt_after_60days_dt864_cf_linear_mu_nearGL.png")


plot_17=False

if plot_17:
    plt.figure(20,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_3['Melt_t60.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}$ constant')
    plt.plot(x1, df_dt_864_delT_3_fric_vel['Melt_t60.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 60 days including u$_{fric}$ in the melt param')
    plt.savefig(output_folder + "fig20_melt_after_60days_dt864_cf_fric_vel.png")


plot_18=False

if plot_18:
    plt.figure(21,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_3['Melt_t40.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}$ constant')

    plt.plot(x1, df_dt_864_delT_3_fric_vel['Melt_t40.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$')

    plt.plot(x1, df_dt_864_delT_3_lin_mu['Melt_t40.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}$ constant')

    plt.plot(x1, df_dt_864_delT_3_lin_mu_fric_vel['Melt_t40.0'] * year_in_seconds,
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')

    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 40 days comparing linear change in viscosity across the domain and including u$_{fric}$ in the melt param')
    plt.savefig(output_folder + "fig21_melt_after_40days_dt864_cf_fric_vel_and_linmu.png")


plot_19 = False
# plot convergence for melt for dt=864, for different temp forcings . (at the moment up to 50 days - not reached steady state)
if plot_19:
    plt.figure(22, figsize=figsize)
    #dt_output = 10days
    t_output10 = [(i + 1) * 10 for i in range(12)]
    melt_gl_delT3 = []
    for j in range(12):
        melt_str_t = str((j + 1) * 10)
        try:
            melt_time_i_delT3 = df_dt_864_delT_3['Melt_t' + melt_str_t + '.0'].tolist()[0]
        except:
            melt_time_i_delT3 = np.nan
        melt_gl_delT3.append(melt_time_i_delT3)

    melt_gl_delT3 = np.array(melt_gl_delT3)

    # dt_output = 5days
    t_output5 = [(i+1)*5 for i in range(24)]

    melt_gl_delT3_lin_mu = []
    melt_gl_delT3_fric_vel = []
    melt_gl_delT3_lin_mu_fric_vel = []
    for j in range(24):
        melt_str_t = str((j + 1) * 5)
        try:
            melt_time_i_delT3_lin_mu = df_dt_864_delT_3_lin_mu['Melt_t' + melt_str_t + '.0'].tolist()[0]
        except:
            melt_time_i_delT3_lin_mu = np.nan
        try:
            melt_time_i_delT3_fric_vel = df_dt_864_delT_3_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[0]
        except:
            melt_time_i_delT3_fric_vel = np.nan
        try:
            melt_time_i_delT3_lin_mu_fric_vel = df_dt_864_delT_3_lin_mu_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[0]
        except:
            melt_time_i_delT3_lin_mu_fric_vel = np.nan


        melt_gl_delT3_lin_mu.append(float(melt_time_i_delT3_lin_mu))
        melt_gl_delT3_fric_vel.append(float(melt_time_i_delT3_fric_vel))
        melt_gl_delT3_lin_mu_fric_vel.append(float(melt_time_i_delT3_lin_mu_fric_vel))

    melt_gl_delT3_lin_mu = np.array(melt_gl_delT3_lin_mu)
    melt_gl_delT3_fric_vel = np.array(melt_gl_delT3_fric_vel)
    melt_gl_delT3_lin_mu_fric_vel = np.array(melt_gl_delT3_lin_mu_fric_vel)

    plt.subplot(2,1,1)
    plt.plot(t_output10,melt_gl_delT3 * year_in_seconds,'-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}$ constant')
    plt.plot(t_output5, melt_gl_delT3_lin_mu * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}$ constant ')
    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')

    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at GL / m/yr')
    plt.xlim([0, 100])
    plt.suptitle(r'Convergence of Melt rate at GL comparing linear change in viscosity across the domain and including u$_{fric}$ in the melt param')

    plt.subplot(2, 1, 2)

    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ',color='green')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ',color='red')
    #plt.ylim([0.0045,0.0065])
    plt.xlim([0,100])
    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at GL / m/yr')

    plt.savefig(output_folder + "fig22_convergence_melt_for_lin_mu_and_fric_vel_atGL.png")



plot_20 = False
# plot convergence for melt for dt=864, for changing mu and fric vel
if plot_20:
    plt.figure(23, figsize=figsize)
    #dt_output = 10days
    t_output10 = [(i + 1) * 10 for i in range(12)]
    melt_gl_delT3 = []
    for j in range(12):
        melt_str_t = str((j + 1) * 10)
        try:
            melt_time_i_delT3 = df_dt_864_delT_3['Melt_t' + melt_str_t + '.0'].tolist()[20]
        except:
            melt_time_i_delT3 = np.nan
        melt_gl_delT3.append(melt_time_i_delT3)

    melt_gl_delT3 = np.array(melt_gl_delT3)

    # dt_output = 5days
    t_output5 = [(i+1)*5 for i in range(24)]

    melt_gl_delT3_lin_mu = []
    melt_gl_delT3_fric_vel = []
    melt_gl_delT3_lin_mu_fric_vel = []
    for j in range(24):
        melt_str_t = str((j + 1) * 5)
        try:
            melt_time_i_delT3_lin_mu = df_dt_864_delT_3_lin_mu['Melt_t' + melt_str_t + '.0'].tolist()[20]
        except:
            melt_time_i_delT3_lin_mu = np.nan
        try:
            melt_time_i_delT3_fric_vel = df_dt_864_delT_3_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[20]
        except:
            melt_time_i_delT3_fric_vel = np.nan
        try:
            melt_time_i_delT3_lin_mu_fric_vel = df_dt_864_delT_3_lin_mu_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[20]
        except:
            melt_time_i_delT3_lin_mu_fric_vel = np.nan


        melt_gl_delT3_lin_mu.append(float(melt_time_i_delT3_lin_mu))
        melt_gl_delT3_fric_vel.append(float(melt_time_i_delT3_fric_vel))
        melt_gl_delT3_lin_mu_fric_vel.append(float(melt_time_i_delT3_lin_mu_fric_vel))

    melt_gl_delT3_lin_mu = np.array(melt_gl_delT3_lin_mu)
    melt_gl_delT3_fric_vel = np.array(melt_gl_delT3_fric_vel)
    melt_gl_delT3_lin_mu_fric_vel = np.array(melt_gl_delT3_lin_mu_fric_vel)

    plt.subplot(2,1,1)
    plt.plot(t_output10,melt_gl_delT3 * year_in_seconds,'-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}$ constant')
    plt.plot(t_output5, melt_gl_delT3_lin_mu * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}$ constant ')
    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')

    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at 1000m from GL / m/yr')
    plt.xlim([0, 100])
    plt.suptitle(r'Convergence of Melt rate at 1000m comparing linear change in viscosity across the domain and including u$_{fric}$ in the melt param')

    plt.subplot(2, 1, 2)

    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ', color='green')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ',color='red')
    #plt.ylim([0.0045,0.0065])
    plt.xlim([0,100])
    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at 1000m from GL / m/yr')

    plt.savefig(output_folder + "fig23_convergence_melt_for_lin_mu_and_fric_vel_at1000m.png")


plot_21 = False
# plot convergence for melt for dt=864, for changing mu and fric vel at 3500m
if plot_21:
    plt.figure(24, figsize=figsize)
    #dt_output = 10days
    t_output10 = [(i + 1) * 10 for i in range(12)]
    melt_gl_delT3 = []
    for j in range(12):
        melt_str_t = str((j + 1) * 10)
        try:
            melt_time_i_delT3 = df_dt_864_delT_3['Melt_t' + melt_str_t + '.0'].tolist()[70]
        except:
            melt_time_i_delT3 = np.nan
        melt_gl_delT3.append(melt_time_i_delT3)

    melt_gl_delT3 = np.array(melt_gl_delT3)

    # dt_output = 5days
    t_output5 = [(i+1)*5 for i in range(24)]

    melt_gl_delT3_lin_mu = []
    melt_gl_delT3_fric_vel = []
    melt_gl_delT3_lin_mu_fric_vel = []
    for j in range(24):
        melt_str_t = str((j + 1) * 5)
        try:
            melt_time_i_delT3_lin_mu = df_dt_864_delT_3_lin_mu['Melt_t' + melt_str_t + '.0'].tolist()[70]
        except:
            melt_time_i_delT3_lin_mu = np.nan
        try:
            melt_time_i_delT3_fric_vel = df_dt_864_delT_3_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[70]
        except:
            melt_time_i_delT3_fric_vel = np.nan
        try:
            melt_time_i_delT3_lin_mu_fric_vel = df_dt_864_delT_3_lin_mu_fric_vel['Melt_t' + melt_str_t + '.0'].tolist()[70]
        except:
            melt_time_i_delT3_lin_mu_fric_vel = np.nan


        melt_gl_delT3_lin_mu.append(float(melt_time_i_delT3_lin_mu))
        melt_gl_delT3_fric_vel.append(float(melt_time_i_delT3_fric_vel))
        melt_gl_delT3_lin_mu_fric_vel.append(float(melt_time_i_delT3_lin_mu_fric_vel))

    melt_gl_delT3_lin_mu = np.array(melt_gl_delT3_lin_mu)
    melt_gl_delT3_fric_vel = np.array(melt_gl_delT3_fric_vel)
    melt_gl_delT3_lin_mu_fric_vel = np.array(melt_gl_delT3_lin_mu_fric_vel)

    plt.subplot(2,1,1)
    plt.plot(t_output10,melt_gl_delT3 * year_in_seconds,'-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}$ constant')
    plt.plot(t_output5, melt_gl_delT3_lin_mu * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}$ constant ')
    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ')

    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at 3500m from GL / m/yr')
    plt.xlim([0, 100])
    plt.suptitle(r'Convergence of Melt rate at 3500m comparing linear change in viscosity across the domain and including u$_{fric}$ in the melt param')

    plt.subplot(2, 1, 2)

    plt.plot(t_output5, melt_gl_delT3_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=2 $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ', color='green')
    plt.plot(t_output5, melt_gl_delT3_lin_mu_fric_vel * year_in_seconds, '-x',
             label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, v=0.02+3.96E-4x $m^2/s$, melt param: $\gamma_{T,S}*u_{fric}$ ',color='red')
    #plt.ylim([0.0045,0.0065])
    plt.xlim([0,100])
    plt.legend()
    plt.xlabel('Simulation Time / days')
    plt.ylabel('Melt rate at 3500m from GL / m/yr')

    plt.savefig(output_folder + "fig24_convergence_melt_for_lin_mu_and_fric_vel_at3500m.png")


# get linear mu and linear kappa runs!#

df_dt_864_delT_point1_lin_kappa = pd.read_csv("/data/thwaites/9.12.18.iSWING_3_eq_param_dt_864_t120_delT_0.1/top_boundary_data.csv")
df_dt_864_delT_1_lin_kappa = pd.read_csv("/data/thwaites/9.12.18.iSWING_3_eq_param_dt_864_t120_delT_1.0/top_boundary_data.csv")
df_dt_864_delT_3_lin_kappa = pd.read_csv("/data/thwaites/9.12.18.iSWING_3_eq_param_dt_864_t120_delT_3.0/top_boundary_data.csv")
plot_22=False
if plot_22:
# plot Meltrate after 30 days for lin kappa runs delT - 0.1,1,3
    plt.figure(25,figsize=figsize)
    plt.plot(x1,df_dt_864_delT_point1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.plot(x1,df_dt_864_delT_1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary after 30 days')
    plt.savefig(output_folder+"fig25_melt_after_30days_lin_kappa.png")

    plt.figure(26,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_point1_lin_kappa['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    plt.plot(x1, df_dt_864_delT_1_lin_kappa['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_3_lin_kappa['Melt_t30.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0,20])
    plt.ylim([0,1])
    plt.title('Melt rate along ice shelf boundary after 30 days close to GL')
    plt.savefig(output_folder+"fig26_melt_after_30days_lin_kappa_nearGL.png")





df_dt_864_delT_5 = pd.read_csv("/data/thwaites/18.12.18.iSWING_3_eq_param_dt_864_t120_delT_5.0_change_p_split/top_boundary_data.csv")
df_dt_864_delT_4 = pd.read_csv("/data/thwaites/18.12.18.iSWING_3_eq_param_dt_864_t120_delT_4.0_change_p_split/top_boundary_data.csv")
df_dt_864_delT_minus3 = pd.read_csv("/data/thwaites/18.12.18.iSWING_3_eq_param_dt_864_t120_delT_minus3.0_change_p_split/top_boundary_data.csv")
df_dt_864_delT_3_2eq = pd.read_csv("/data/thwaites/18.12.18.iSWING_2_eq_param_dt_864_t120_delT_0.1_change_p_split/top_boundary_data.csv")
df_dt_864_delT_point1_no_fric = pd.read_csv("/data/thwaites/18.12.18.iSWING_3_eq_param_dt_864_t120_delT_0.1_change_p_split_no_fric_vel/top_boundary_data.csv")
df_dt_864_delT_point1 = pd.read_csv("/data/thwaites/18.12.18.iSWING_3_eq_param_dt_864_t120_delT_0.1_change_p_split/top_boundary_data.csv")
plot_23=False
if plot_23:
# plot Meltrate after 30 days for lin kappa runs delT - 0.1,1,3
    plt.figure(27,figsize=figsize)
    plt.plot(x1,df_dt_864_delT_5['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 5.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_4['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 4.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3_2eq['Melt_t0.1']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, 2 equation param')
    plt.plot(x1,df_dt_864_delT_point1_no_fric['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, no fric vel')
    plt.plot(x1,df_dt_864_delT_point1['Melt_t_0.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, with fric vel')
    plt.plot(x1,df_dt_864_delT_minus3['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = -3.0$\degree$C')


    #plt.plot(x1,df_dt_864_delT_1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    #plt.plot(x1,df_dt_864_delT_3_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary ')
    plt.savefig(output_folder+"fig27_melt_after_1day_delT_5.0.png")

    plt.figure(28,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_5['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 5.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_4['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 4.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3_2eq['Melt_t0.1']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, 2 equation param')
    plt.plot(x1,df_dt_864_delT_point1_no_fric['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, no fric vel')
    plt.plot(x1,df_dt_864_delT_point1['Melt_t_0.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, with fric vel')
    plt.plot(x1, df_dt_864_delT_minus3['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = -3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0,20])
    plt.ylim([-10E-5,10E-5])
    plt.title('Melt rate along ice shelf boundary after 1 day close to GL')
    plt.savefig(output_folder+"fig28_melt_after_1_day_nearGL.png")

    plt.figure(29,figsize=figsize)
    plt.plot(x1, df_dt_864_delT_5['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 5.0$\degree$C')
    plt.plot(x1, df_dt_864_delT_4['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = 4.0$\degree$C')
    plt.plot(x1,df_dt_864_delT_3_2eq['Melt_t0.1']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, 2 equation param')
    plt.plot(x1,df_dt_864_delT_point1_no_fric['Melt_t1.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, no fric vel')
    plt.plot(x1,df_dt_864_delT_point1['Melt_t_0.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, with fric vel')
    plt.plot(x1, df_dt_864_delT_minus3['Melt_t1.0'] * year_in_seconds, label=r'dt = 864s, $\Delta$T = -3.0$\degree$C')
    plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.xlim([0,20])
   # plt.ylim([-2,2])
    plt.title('Melt rate along ice shelf boundary after 1 day close to GL')
    plt.savefig(output_folder+"fig29_melt_after_1_day_nearGL.png")

# replotting 9.12 long runs...

plot_24=True
if plot_24:
# plot Meltrate after 30 days for lin kappa runs delT - 0.1,1,3
    plt.figure(30,figsize=figsize)
    for j in range(1,13):
        plt.plot(x1, df_dt_864_delT_point1_lin_kappa['Melt_t'+str(j)+'0.0'] * year_in_seconds,
                 label=r'dt = 864s, $\Delta$T = 0.1$\degree$C, time = '+str(j)+'0 days')
    #plt.plot(x1,df_dt_864_delT_point1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    #plt.plot(x1,df_dt_864_delT_1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    #plt.plot(x1,df_dt_864_delT_3_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    #plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary')
    plt.savefig(output_folder+"fig30_melt_after_120days_lin_kappa.png")
    plt.figure(31,figsize=figsize)
    for j in range(1,13):
        plt.plot(x1, df_dt_864_delT_3_lin_kappa['Melt_t'+str(j)+'0.0'] * year_in_seconds,
                 label=r'dt = 864s, $\Delta$T = 3.0$\degree$C, time = '+str(j)+'0 days')
    #plt.plot(x1,df_dt_864_delT_point1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 0.1$\degree$C')
    #plt.plot(x1,df_dt_864_delT_1_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 1.0$\degree$C')
    #plt.plot(x1,df_dt_864_delT_3_lin_kappa['Melt_t30.0']*year_in_seconds,label=r'dt = 864s, $\Delta$T = 3.0$\degree$C')
    #plt.xlabel('Distance from Grounding Line / m')
    plt.ylabel('Melt rate / m/yr')
    plt.legend()
    plt.title('Melt rate along ice shelf boundary')
    plt.savefig(output_folder+"fig31_melt_after_120days_lin_kappa.png")

plt.show()

'''
Q_mixed_top = []
Q_ice_top = []
Q_latent_top = []
Tb_top = []
Melt_top = []
Ice_height_top = []


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
    x_top = np.array(x_top)
    Q_mixed_top = np.array(Q_mixed_top) * 1025. * 3974

    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(x_top, Ice_height_top)
    plt.ylabel("Ice thickness /m")

    plt.subplot(3, 1, 2)
    plt.plot(x_top, Q_mixed_top, label="Qm")
    plt.plot(x_top, Q_ice_top, label="Qi")
    plt.plot(x_top, Q_latent_top, label="Qlat=Qi-Qm")
    plt.legend()
    # plt.ylabel("Heat flux through the top boundary / W/m^2")

    plt.grid()
    plt.ylabel("Heat flux through top boundary / W/m^2")

    plt.subplot(3, 1, 3)
    plt.plot(x_top, np.array(Melt_top) * (3600 * 24. * 365), label="wb = -Qlat/(rho_m*Lf)")
    plt.legend()
    plt.ylabel("Melt rate / m/yr")
    plt.xlabel("Distance along top boundary / m")
    plt.grid()

    plt.figure(2)
    plt.grid()
    plt.plot(x_top, Q_mixed_top)

except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

try:
    plt.show()
except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e)
'''
