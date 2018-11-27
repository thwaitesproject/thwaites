from firedrake import conditional
from firedrake import Constant


class MeltRateParam():
    """methods for calculating meltrate parameterisation used in ice-ocean boundary"""
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
    
    g = 9.81
    rho0 = 1025.0

    



    def __init__(self,S,T,Pin,h_ice,z,dz):
        self.z = z
        self.dz_calc = dz*1.0
        self.P_ice = self.rho_ice * self.g * h_ice  # hydrostatic pressure just from ice
        self.Pfull = self.rho0 * (Pin - self.g * self.z) + self.P_ice
        self.S = S
        self.T = T




    def two_eq_param_meltrate(self):

        Tb = conditional(z > 0 - self.dz_calc, self.a * self.S + self.b + self.c * self.Pfull, 0.0)
        # Q_ice = conditional(z > 0-dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick!
        Q_ice = Constant(0.0)
        Q_mixed = conditional(self.z > 0 - self.dz_calc, -self.rho0 * (self.Tb - self.T) * self.c_p_m * self.gammaT, 0.0)
        Q_latent = conditional(self.z > 0 - self.dz_calc, Q_ice - Q_mixed, 0.0)
        wb = conditional(self.z > 0 - self.dz_calc, -Q_latent / (self.Lf * self.rho0), 0.0)

        Q_mixed_bc = conditional(self.z > 0 - self.dz_calc, -(wb + self.gammaT) * (Tb - self.T),
                                 0.0)  # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
        Q_s = wb*self.S

        return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s, wb, Tb, self.Pfull  # these are all still expressions


    def three_eq_param_meltrate_without_Qice(self):


        Aa = self.c_p_m*self.gammaT*self.a

        Bb = self.c_p_m*self.gammaT*self.b - self.c_p_m*self.gammaT*self.T
        Bb = Bb + self.c_p_m*self.c*self.Pfull*self.gammaT - self.gammaS*Lf

        Cc = self.gammaS*self.Lf*self.S

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

        loc_Tb = conditional(self.z > 0 - self.dz_calc, self.a * loc_Sb + self.b + self.c * self.Pfull, 0.0)
        # Q_ice = conditional(self.z > 0-self.dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick!
        Q_ice = Constant(0.0)
        Q_mixed = conditional(self.z > 0 - self.dz_calc, -self.rho0 * (loc_Tb - self.T) * self.c_p_m * self.gammaT, 0.0)
        Q_latent = conditional(self.z > 0 - self.dz_calc, Q_ice - Q_mixed, 0.0)
        wb = conditional(self.z > 0 - self.dz_calc, -Q_latent / (self.Lf * self.rho0), 0.0)

        Q_mixed_bc = conditional(self.z > 0 - self.dz_calc, -(wb + self.gammaT) * (loc_Tb - self.T),
                                 0.0)  # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
        Q_s_bc = conditional(self.z > 0 - self.dz_calc, -(wb + self.gammaS) * (loc_Sb - self.S), 0.0)

        return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s_bc, wb, loc_Tb, self.Pfull  # these are all still expressions


    def three_eq_param_meltrate(self):
        # solve with two equation param.
        b_plus_cPb = (self.b + self.c * self.Pfull) # save calculating this each time...

        Aa = self.c_p_m * self.gammaT * self.a - self.c_p_i * self.gammaS * self.a

        Bb = self.c_p_i * self.gammaS * self.T_ice - self.c_p_i * self.gammaS * b_plus_cPb + self.c_p_i * self.gammaS * self.S * self.a
        Bb = Bb + self.c_p_m * self.gammaT * b_plus_cPb - self.c_p_m * self.gammaT * self.T - self.gammaS * self.Lf

        Cc = self.c_p_i * self.gammaS * self.S * b_plus_cPb - self.c_p_m * self.gammaS * self.S * self.T_ice + self.gammaS * self.Lf * self.S

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

        loc_Tb = conditional(self.z > 0 - self.dz_calc, self.a * loc_Sb + self.b + self.c * self.Pfull, 0.0)

        wb = conditional(self.z > 0 - self.dz_calc, -self.gammaS*(loc_Sb-self.S)/loc_Sb, 0.0)
        # Q_ice = conditional(z > 0-dz_calc,-rho_ice*c_p_i*k_i*(T_ice-Tb)/h_ice,0.0)  # assumption 2 in holland and jenkins - not so good because ice is thick!
        Q_ice = conditional(self.z > 0 - self.dz_calc, -self.rho0 * (self.T_ice - loc_Tb) * self.c_p_i * wb, 0.0)
        Q_mixed = conditional(self.z > 0 - self.dz_calc, -self.rho0 * (loc_Tb - self.T) * self.c_p_m * self.gammaT, 0.0)
        Q_latent = conditional(self.z > 0 - self.dz_calc, Q_ice - Q_mixed, 0.0)


        Q_mixed_bc = conditional(self.z > 0 - self.dz_calc, -(wb + self.gammaT) * (loc_Tb - self.T),
                                 0.0)  # units of Km/s , add in meltrate to capture flux of water through boundary Jenkins et al 2001 eq 25
        Q_s_bc = conditional(self.z > 0 - self.dz_calc, -(wb + self.gammaS) * (loc_Sb - self.S), 0.0)

        return Q_ice, Q_mixed_bc, Q_mixed, Q_latent, Q_s_bc, wb, loc_Tb, self.Pfull  # these are all still expressions
