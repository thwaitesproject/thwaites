from thwaites import *
from thwaites.meltrate_param import TwoEqMeltRateParam
import numpy as np
import matplotlib.pyplot as plt

S = 33.5
T = -2.5
p = 0
z = -1000.
u = 0.0001

######################################################################

# Print out results for standard three equation parameterisation
print("#"*40)
print('''Test: Conservation of heat
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z, u)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb


print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)
print("#"*40)
print('''Test: Conservation of salt
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)

######################################################################

# Print out results for three equation parameterisation with H&J 1999 
# friction velocity no coriolis.

print("#"*40)
print('''Test: Conservation of heat HJ1999 gamma(ufric), f = 0
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z, u, HJ99Gamma=True)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb


print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)
print("#"*40)
print('''Test: Conservation of salt HJ1999 gamma(ufric), f = 0
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)

######################################################################                                                                                                                                                                                                                                                                                                                                                                # Print out results for three equation parameterisation with H&J 1999                                                                                                                                              # friction velocity and with coriolis.

print("#"*40)
print('''Test: Conservation of heat HJ1999 gamma(ufric), f = -1e-4s^-1
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z, u, HJ99Gamma=True,f=-1e-4)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb
print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)
print("#"*40)
print('''Test: Conservation of salt HJ1999 gamma(ufric), f = -1e-4s^-1
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)


#####################################################################################

# Print out results for three equation parameterisation without frictional velocity
print("#"*40)
print('''Test: Conservation of heat (without u*)
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb

print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)
print("#"*40)
print('''Test: Conservation of salt (without u*)
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)

###################################################################

# Print out results for three equation parameterisation and Qice
print("#"*40)
print('''Test: Conservation of heat (without Qice)
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z, u, False)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb

print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)

print("#"*40)
print('''Test: Conservation of salt (without Qice)
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)


###################################################################

# Print out results for three equation parameterisation without u* and Qice
print("#"*40)
print('''Test: Conservation of heat (without u* and Qice)
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParam(S, T, p, z, None, False)
Qlat = -mp.wb*mp.rho0*mp.Lf
QSbrine = -mp.rho0*mp.wb*mp.Sb

print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)

print("#"*40)
print('''Test: Conservation of salt (without u* and Qice)
QSbrine = QSice - QSmixed
QSice = 0
AND
QSbrine = -m'.rho_sw.Sb''')
print("This script: QSbrine = ", QSbrine)
print("Melt rate param: -QSmixed = ", -mp.QS_mixed)

############################################################

# Print out results for two equation parameterisation
print("#"*40)
print('''Test: Conservation of heat (Two equation parameterisation)
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = TwoEqMeltRateParam(S, T, p, z)
Qlat = -mp.wb*mp.rho0*mp.Lf

print("Melt rate param: m' = ", mp.wb)
print("Melt rate param: Qice = ", mp.Q_ice)
print("Melt rate param: Qmixed = ", mp.Q_mixed)
print("This script: Qlatent = ", Qlat)
print("Melt rate param: Qice - Qmixed = ", mp.Q_latent)

################################################################

# Print out comparison of melt rates for parameterisations
print("#"*40)
print('''Comparison of melt rates
melt rate of ice, m = (rho_sw/rho_ice) * wb (Kimura et al. 2013) 
wb = velocity of ocean normal to boundary''')
mp = ThreeEqMeltRateParam(S, T, p, z, u)
m = mp.wb*(mp.rho0/mp.rho_ice)
m3eq = m
print("Three eq mp: m = ", m*3600*24*265,
      "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = ThreeEqMeltRateParam(S, T, p, z, u, False)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Three eq mp wout Qice: m = ", m*3600*24*265,
      "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = ThreeEqMeltRateParam(S, T, p, z)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Three eq mp wout u*: m = ", m*3600*24*265,
      "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")


mp = ThreeEqMeltRateParam(S, T, p, z, None, False)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Three eq mp wout u* & Qice: m = ", m*3600*24*265,
      "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = TwoEqMeltRateParam(S, T, p, z)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Two eq mp: m = ", m*3600*24*265,
      "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

#######################################################################################

# Automatic tests can be run using pytest
# $ pytest melt_parameterisation.py


def test_heat_conservation():
    mp = ThreeEqMeltRateParam(S, T, p, z, u)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent)/abs(Qlat) <= 1E-12



def test_salt_conservation():
    mp = ThreeEqMeltRateParam(S, T, p, z, u)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed)/abs(QSbrine) <= 1E-12


def test_heat_conservation_wout_fric_vel():
    mp = ThreeEqMeltRateParam(S, T, p, z)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent)/abs(Qlat) <= 1E-12


def test_salt_conservation_wout_fric_vel():
    mp = ThreeEqMeltRateParam(S, T, p, z)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed)/abs(QSbrine) <= 1E-12


def test_heat_conservation_wout_qice():
    mp = ThreeEqMeltRateParam(S, T, p, z, u, False)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent)/abs(Qlat) <= 1E-12


def test_salt_conservation_wout_qice():
    mp = ThreeEqMeltRateParam(S, T, p, z, u, False)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed)/abs(QSbrine) <= 1E-12


def test_heat_conservation_wout_fric_vel_qice():
    mp = ThreeEqMeltRateParam(S, T, p, z, None, False)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent)/abs(Qlat) <= 1E-12


def test_salt_conservation_wout_fric_vel_qice():
    mp = ThreeEqMeltRateParam(S, T, p, z, None, False)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed)/abs(QSbrine) <= 1E-12


def test_heat_conservation_two_eq_param():
    mp = TwoEqMeltRateParam(S, T, p, z)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent)/abs(Qlat) <= 1E-12


def test_ocean_heat_flux_sign():
    mp = ThreeEqMeltRateParam(S, T, p, z, u)

    if mp.wb > 0:
        assert mp.Q_mixed > mp.Q_ice

        # if melting then heat from ocean into ice ocean boundary and heat from boundary into ice shelf
        assert mp.Q_mixed > 0.0
        assert mp.Q_ice > 0.0

        # if melting then fresh water flux into ocean due to fresh melt water
        assert mp.QS_mixed > 0.0
    else:
        assert mp.Q_mixed < mp.Q_ice
        # if freezing then heat from ice into ice ocean boundary and heat from boundary into ocean
        assert mp.Q_mixed < 0.0
        assert mp.Q_ice < 0.0

        # if freezing then salt flux into ocean due to brine rejection
        assert mp.QS_mixed < 0.0



#### does potential temperature conversion make a difference?

mp = ThreeEqMeltRateParam(35, -0.0448, p, 1000, 0.001)
print("Potential temperature = -0.0448degC. At 1000db and 35PSU In situ temperature = 0degC")
print("Meltrate without conversion to insitu temp = ", mp.wb)

mp_conv = ThreeEqMeltRateParam(35, 0.0, p, 1000, 0.001)
print("Meltrate with conversion to insitu temp = ", mp_conv.wb)

print("with conversion melt rate is {:.2f} times larger....".format(mp_conv.wb / mp.wb))
