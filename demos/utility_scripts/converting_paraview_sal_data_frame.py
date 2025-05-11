import pandas as pd
import numpy as np

salinity_dataframe_top_linear_mu = pd.DataFrame()

#folder = "/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_432_1/"
folder = "/data/thwaites/1.12.18.ice_shelf_cavity_circulation_3_eq_param_dt_864_1/"

def cavity_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x



for j in range(1):
    #sal_str_t = str((j + 1) * 5)
    sal_str_t = str(30)
    salinity_file = open(folder+"salinity_t_"+sal_str_t+".csv")
    x_values = []
    z_values = []
    sal_values = []

    while True:
        try:
            salinity_file.readline()
            line = salinity_file.readline()
            words = line.split(',')
            x_values.append(float(words[1]))
            z_values.append(float(words[2]))
            sal_values.append(float(words[0]))
        except:
            break
    x_array=np.array(x_values)
    z_array=np.array(z_values)
    sal_array=np.array(sal_values)


    x_top = []
    sal_top = []
    z = z_array - cavity_thickness(x_array, 0.0, 1.0, 5000.0, 100.0)

    for i in range(len(z_values)):

        if z[i] == 0.0:
            x_top.append(float(x_values[i]))
            sal_top.append(float(sal_values[i]))



    salinity_dataframe_top_linear_mu['x_top_boundary_t'+sal_str_t] = x_top
    salinity_dataframe_top_linear_mu['sal_t' + sal_str_t] = sal_top

salinity_dataframe_top_linear_mu.to_csv(folder+"salinity_data_frame.csv")
