from firedrake import conditional
from firedrake import Constant


class MeltRateParam:
    """methods for calculating melt rate parameterisation used in ice-ocean boundary"""
    a = -5.73E-2  # salinity coefficient of freezing equation
    b = 0.0832  # use fluidity value #9.39E-2  # constant coefficient of freezing equation
    c = -7.53E-8  # pressure coefficient of freezing equation

    c_p_m = 3974.  # specific heat capacity of mixed layer
    c_p_i = 2009.
    gammaT = 1.0E-4  # thermal exchange velocity, m/s
    gammaS = 5.05E-7 # salt exchange velocity, m/s

    GammaT = 1.1E-2  # dimensionless GammaT (capital). gammaT = GammaT.u* from jenkins et al 2010
    GammaS = GammaT / 35.0  # dimensionless GammaS (capital). gammaS = GammaS.u* from jenkins et al 2010
    C_d = 0.0025  # 0.0097  # drag coefficient for ice...

    Lf = 3.34E5  # latent heat of fusion

    k_i = 1.14E-6

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

        if isinstance(self.Tb, float):
            self.Q_ice = 0.0
        else:
            self.Q_ice = Constant(0.0)

        self.Q_mixed = -self.rho0 * self.c_p_m * self.gammaT * (self.Tb - self.T)
        self.Q_latent = self.Q_ice - self.Q_mixed
        self.wb = -self.Q_latent / (self.Lf * self.rho0)
        self.T_flux_bc = -(self.wb + self.gammaT) * (self.Tb - self.T)
        self.Q_s = self.wb*self.S


class ThreeEqMeltRateParam(MeltRateParam):
    def __init__(self, salinity, temperature, pressure_perturbation, z, velocity=None, ice_heat_flux=True):
        super().__init__(salinity, temperature, pressure_perturbation, z)

        if velocity is None:
            gammaT = self.gammaT
            gammaS = self.gammaS
        else:
            u = velocity
            u_tidal = 0.01
            u_star = pow(self.C_d*(pow(u, 2)+pow(u_tidal, 2)), 0.5)

            gammaT = self.GammaT * u_star
            gammaS = self.GammaS * u_star

        b_plus_cPb = (self.b + self.c * self.P_full)  # save calculating this each time...

        # Calculate coefficients in quadratic equation for salinity at ice-ocean boundary.
        # Aa.Sb^2 + Bb.Sb + Cc = 0
        Aa = self.c_p_m * gammaT * self.a
        Bb = -gammaS * self.Lf
        Bb -= self.c_p_m * gammaT * self.T
        Bb += self.c_p_m * gammaT * b_plus_cPb
        Cc = gammaS * self.S * self.Lf

        if ice_heat_flux:
            Aa -= gammaS * self.c_p_i * self.a
            Bb += gammaS * self.S * self.c_p_i * self.a
            Bb -= gammaS * self.c_p_i * b_plus_cPb
            Bb += gammaS * self.c_p_i * self.T_ice
            Cc += gammaS * self.S * self.c_p_i * b_plus_cPb
            Cc -= gammaS * self.S * self.c_p_i * self.T_ice

        S1 = (-Bb + pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)
        S2 = (-Bb - pow(Bb ** 2 - 4.0 * Aa * Cc, 0.5)) / (2.0 * Aa)

        if isinstance(S1, float):
            # Print statements for testing
            print("S1 = ", S1)
            print("S2 = ", S2)
            if S1 > 0:
                self.Sb = S1
                print("Choose S1")
            else:
                self.Sb = S2
                print("Choose S2")

        else:
            self.Sb = conditional(S1 > 0.0, S1, S2)

        self.Tb = self.a * self.Sb + self.b + self.c * self.P_full
        self.wb = gammaS * (self.S - self.Sb) / self.Sb

        if ice_heat_flux:
            self.Q_ice = -self.rho0 * (self.T_ice - self.Tb) * self.c_p_i * self.wb
        else:
            if isinstance(S1, float):
                self.Q_ice = 0.0
            else:
                self.Q_ice = Constant(0.0)

        self.Q_mixed = -self.rho0 * self.c_p_m * gammaT * (self.Tb - self.T)
        self.Q_latent = self.Q_ice - self.Q_mixed

        self.QS_mixed = -self.rho0 * gammaS * (self.Sb - self.S)

        self.T_flux_bc = -(self.wb + gammaT) * (self.Tb - self.T)
        self.S_flux_bc = -(self.wb + gammaS) * (self.Sb - self.S)
