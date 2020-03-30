from firedrake import conditional
from firedrake import Constant
from firedrake import Max,ln

class MeltRateParam:
    """methods for calculating melt rate parameterisation used in ice-ocean boundary"""
    a = -5.73E-2  # salinity coefficient of freezing equation
    b = 0.0832  # constant coefficient of freezing equation
    c = -7.53E-8  # pressure coefficient of freezing equation

    c_p_m = 3974.  # specific heat capacity of mixed layer, J / kg /K
    c_p_i = 2000.  # Specific heat capacity of ice, J/kg/K
    gammaT = 1.0E-4  # thermal exchange velocity, m/s
    gammaS = 5.05E-7 # salt exchange velocity, m/s

    GammaT = 1.1E-2  # dimensionless GammaT (capital). gammaT = GammaT.u* from jenkins et al 2010
    GammaS = GammaT / 35.0  # dimensionless GammaS (capital). gammaS = GammaS.u* from jenkins et al 2010
    C_d = 0.0097  # Drag coefficient for ice, dimensionless

    Lf = 3.34E5  # Latent heat of fusion of ice, J/kg

    k_i = 1.14E-6  # Molecular thermal conductivity of ice, m^2/s

    rho_ice = 920.0  # Reference density of ice, kg/m^3
    T_ice = -20.0  # Temperature of far field ice, degC
    
    g = 9.81  # gravitational acceleration, m/s^2
    rho0 = 1030.  # Reference density of sea water, kg/m^3

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

    def __init__(self, salinity, temperature, pressure_perturbation, z):
        P_hydrostatic = -self.rho0 * self.g * z
        self.P_full = P_hydrostatic #+ self.rho0 * pressure_perturbation
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
    def __init__(self, salinity, temperature, pressure_perturbation, z, velocity=None, ice_heat_flux=True, HJ99Gamma=False, f=None):
        super().__init__(salinity, temperature, pressure_perturbation, z)

        if velocity is None:
            # Use constant turbulent thermal and salinity exchange values given in Holland and Jenkins 1999. 
            gammaT = self.gammaT
            gammaS = self.gammaS

        else:
            u = velocity

            if HJ99Gamma:
                # Holland and Jenkins 1999 turbulent heat/salt exchange velocity as a function
                # of friction velocity. This should be the same as in MITgcm.  

                # Holland, D.M. and Jenkins, A., 1999. Modeling thermodynamic ice–ocean interactions at 
                #the base of an ice shelf. Journal of Physical Oceanography, 29(8), pp.1787-1800.

                # Calculate friction velocity
                print(u)
                print(type(u))
                if isinstance(u, float):
                    u_bounded = max(u,1e-3)
                else:
                    u_bounded = conditional(u > 1e-3, u, 1e-3)
                u_star = pow(self.C_d * pow(u_bounded, 2), 0.5)
                
                # Calculate turbulent component of thermal and salinity exchange velocity (Eq 15) 
                # N.b MITgcm sets etastar = 1 so that the first part of Gamma_turb term below is constant. 
                # eta_star = (1 + (zeta_N * u_star) / (f * L0 * Rc))^-1/2  (Eq 18)
                # In H&J99  eta_star is set to 1 when the Obukhov length is negative (i.e the buoyancy flux 
                # is destabilising. This occurs during freezing. (Melting is stabilising) 
                # So it does seem a bit odd because then eta_star is tuned to freezing conditions 
                # need to work this out...?
                Gamma_Turb = 1.0 / (2.0 * self.zeta_N * self.eta_star) - 1.0 / self.k
                if f is not None:
                    # Add extra term if using coriolis term
                    # Calculate viscous sublayer thickness (Eq 17)
                    h_nu = 5.0 * self.nu / u_star 
                    Gamma_Turb += ln(u_star * self.zeta_N * pow(self.eta_star, 2) / (abs(f) * h_nu)) / self.k
                
                # Calculate molecular components of thermal exchange velocity (Eq 16)
                GammaT_Mole = 12.5 * pow(self.Pr, 2.0/3.0) - 6.0

                # Calculate molecular component of salinity exchange velocity (Eq 16)
                GammaS_Mole = 12.5 * pow(self.Sc, 2.0/3.0) - 6.0
                
                # Calculate thermal and salinity exchange velocity. (Eq 14)
                # Do we need to catch -ve gamma? could have -ve Gamma_Turb?
                gammaT = u_star / (Gamma_Turb + GammaT_Mole)
                gammaS = u_star / (Gamma_Turb + GammaS_Mole)
                
                # print exchange velocities if testing when input velocity is a float.
                if isinstance(gammaT, float) or isinstance(gammaS, float):
                    print("gammaT = ", gammaT)
                    print("gammaS = ", gammaS)

            else:
                # ISOMIP+ based on Jenkins et al 2010. Measurement of basal rates beneath Ronne Ice Shelf
                u_tidal = 0.01
                u_star = pow(self.C_d*(pow(u, 2)+pow(u_tidal, 2)), 0.5)

                gammaT = self.GammaT * u_star
                gammaS = self.GammaS * u_star

                # print exchange velocities if testing when input velocity is a float.
                if isinstance(gammaT, float) or isinstance(gammaS, float):
                    print("gammaT = ", gammaT)
                    print("gammaS = ", gammaS)

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
