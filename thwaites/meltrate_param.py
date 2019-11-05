from firedrake import conditional
from firedrake import Constant


class MeltRateParam:
    """methods for calculating melt rate parameterisation used in ice-ocean boundary"""
    a = -5.73E-2  # salinity coefficient of freezing equation
    b = 0.0832  # use fluidity value #9.39E-2  # constant coefficient of freezing equation
    c = -7.53E-8  # pressure coefficient of freezing equation

    c_p_m = 3974.  # specific heat capacity of mixed layer
    c_p_i = 2009.
    gammaT = 1E-4  # roughly thermal exchange velocity
    gammaS = 5.05E-7

    gammaT_fric = 1.1E-2  # from jenkins et al 2010
    gammaS_fric = gammaT_fric / 35.0  # from jenkins et al 2010 - matches fluidity
    C_d = 0.0025  # 0.0097  # drag coefficient for ice...

    Lf = 3.34E5  # latent heat of fusion

    k_i = 0.0  # before isomip 1.14E-6

    rho_ice = 920.0
    T_ice = -25.0
    
    g = 9.81
    rho0 = 1027.5

    def __init__(self, salinity, temperature, pressure_perturbation, z):
        P_hydrostatic = -self.rho0 * self.g * z
        self.P_full = P_hydrostatic + self.rho0 * pressure_perturbation
        self.S = salinity
        self.T = temperature

    def freezing_point(self):
        return self.a * self.S + self.b + self.c * self.P_full


class TwoEqMeltRateParam(MeltRateParam):

    def __init__(self, salinity, temperature, pressure_perturbation, z):
        super().__init__(salinity, temperature, pressure_perturbation, z)

        self.Tb = self.a * self.S + self.b + self.c * self.P_full
        self.Q_ice = Constant(0.0)
        self.Q_mixed = -self.rho0 * (self.Tb - self.T) * self.c_p_m * self.gammaT
        self.Q_latent = self.Q_ice - self.Q_mixed
        self.wb = -self.Q_latent / (self.Lf * self.rho0)
        self.T_flux_bc = -(self.wb + self.gammaT) * (self.Tb - self.T)
        self.Q_s = self.wb*self.S


class ThreeEqMeltRateParamWithoutQice(MeltRateParam):
    def __init__(self, salinity, temperature, pressure_perturbation, z):
        super().__init__(salinity, temperature, pressure_perturbation, z)

        Aa = self.c_p_m*self.gammaT*self.a

        Bb = self.c_p_m*self.gammaT*self.b - self.c_p_m*self.gammaT*self.T
        Bb = Bb + self.c_p_m*self.c*self.P_full*self.gammaT - self.gammaS*self.Lf

        Cc = self.gammaS*self.Lf*self.S

        S1 = (-Bb + pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        S2 = (-Bb - pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        self.Sb = conditional(S1 > 0.0, S1, S2)

        self.Tb = self.a * self.Sb + self.b + self.c * self.P_full

        self.Q_ice = Constant(0.0)
        self.Q_mixed = -self.rho0 * (self.Tb - self.T) * self.c_p_m * self.gammaT
        self.Q_latent = self.Q_ice - self.Q_mixed
        self.wb = -self.Q_latent / (self.Lf * self.rho0)

        self.T_flux_bc = -(self.wb + self.gammaT) * (self.Tb - self.T)
        self.S_flux_bc = -(self.wb + self.gammaS) * (self.Sb - self.S)


class ThreeEqMeltRateParamWithoutFrictionVel(MeltRateParam):
    def __init__(self, salinity, temperature, pressure_perturbation, z):
        super().__init__(salinity, temperature, pressure_perturbation, z)
        b_plus_cPb = (self.b + self.c * self.P_full)  # save calculating this each time...

        Aa = self.c_p_m * self.gammaT * self.a - self.c_p_i * self.gammaS * self.a

        Bb = self.c_p_i * self.gammaS * self.T_ice
        Bb -= self.c_p_i * self.gammaS * b_plus_cPb
        Bb += self.c_p_i * self.gammaS * self.S * self.a
        Bb += self.c_p_m * self.gammaT * b_plus_cPb
        Bb -= self.c_p_m * self.gammaT * self.T
        Bb -= self.gammaS * self.Lf

        Cc = self.c_p_i * self.gammaS * self.S * b_plus_cPb
        Cc -= self.c_p_m * self.gammaS * self.S * self.T_ice
        Cc += self.gammaS * self.Lf * self.S

        S1 = (-Bb + pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        S2 = (-Bb - pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        self.Sb = conditional(S1 > 0.0, S1, S2)

        self.Tb = self.a * self.Sb + self.b + self.c * self.P_full
        self.wb = -self.gammaS*(self.Sb-self.S)/self.Sb

        self.Q_ice = -self.rho0 * (self.T_ice - self.Tb) * self.c_p_i * self.wb
        self.Q_mixed = -self.rho0 * (self.Tb - self.T) * self.c_p_m * self.gammaT
        self.Q_latent = self.Q_ice - self.Q_mixed

        self.T_flux_bc = -(self.wb + self.gammaT) * (self.Tb - self.T)
        self.S_flux_bc = -(self.wb + self.gammaS) * (self.Sb - self.S)


class ThreeEqMeltRateParam(MeltRateParam):
    def __init__(self, salinity, temperature, pressure_perturbation, z, velocity):
        super().__init__(salinity, temperature, pressure_perturbation, z)

        u = velocity
        u_tidal = 0.01
        u_star = pow(self.C_d*(pow(u, 2)+pow(u_tidal, 2)), 0.5)

        T_param = self.gammaT_fric * u_star
        S_param = self.gammaS_fric * u_star

        b_plus_cPb = (self.b + self.c * self.P_full)  # save calculating this each time...

        Aa = self.c_p_m * T_param * self.a - self.c_p_i * S_param * self.a

        Bb = self.c_p_i * S_param * self.T_ice
        Bb -= self.c_p_i * S_param * b_plus_cPb
        Bb += self.c_p_i * S_param * self.S * self.a
        Bb += self.c_p_m * T_param * b_plus_cPb
        Bb -= self.c_p_m * T_param * self.T
        Bb -= S_param * self.Lf

        Cc = self.c_p_i * S_param * self.S * b_plus_cPb
        Cc -= self.c_p_m * S_param * self.S * self.T_ice
        Cc += S_param * self.Lf * self.S

        S1 = (-Bb + pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        S2 = (-Bb - pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        self.Sb = conditional(S1 > 0.0, S1, S2)

        self.Tb = self.a * self.Sb + self.b + self.c * self.P_full
        self.wb = -S_param * (self.Sb - self.S) / self.Sb

        self.Q_ice = -self.rho0 * (self.T_ice - self.Tb) * self.c_p_i * self.wb
        self.Q_mixed = -self.rho0 * (self.Tb - self.T) * self.c_p_m * T_param
        self.Q_latent = self.Q_ice - self.Q_mixed

        self.T_flux_bc = -(self.wb + T_param) * (self.Tb - self.T)
        self.S_flux_bc = -(self.wb + S_param) * (self.Sb - self.S)
