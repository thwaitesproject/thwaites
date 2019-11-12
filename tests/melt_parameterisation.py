from thwaites import *
from thwaites.meltrate_param import ThreeEqMeltRateParamWithoutQice, ThreeEqMeltRateParamWithoutFrictionVel
from thwaites.meltrate_param import TwoEqMeltRateParam
import numpy as np
import matplotlib.pyplot as plt

S = 34.5
T = -2.6
p = 0
z = -1000.
u = 0.1

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

#####################################################################################

# Print out results for three equation parameterisation without frictional velocity
print("#"*40)
print('''Test: Conservation of heat (without u*)
Qlatent = Qice - Qmixed
AND
Qlatent = -m'.rho_sw.L''')
mp = ThreeEqMeltRateParamWithoutFrictionVel(S, T, p, z)
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
print('''Test: Conservation of heat (without u* and Qice)
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



# Print out comparison of melt rates for parameterisations
print("#"*40)
print('''Comparison of melt rates
melt rate of ice, m = (rho_sw/rho_ice) * wb (Kimura et al. 2013) 
wb = velocity of ocean normal to boundary''')
mp = ThreeEqMeltRateParam(S, T, p, z, u)
m = mp.wb*(mp.rho0/mp.rho_ice)
m3eq = m
print("Three eq mp: m = ", m*3600*24*265, "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = ThreeEqMeltRateParamWithoutFrictionVel(S, T, p, z)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Three eq mp wout u*: m = ", m*3600*24*265, "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = ThreeEqMeltRateParamWithoutQice(S, T, p, z)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Three eq mp wout u* & Qice: m = ", m*3600*24*265, "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")

mp = TwoEqMeltRateParam(S, T, p, z)
m = mp.wb*(mp.rho0/mp.rho_ice)
print("Two eq mp: m = ", m*3600*24*265, "m/yr", "|m - m_3eq| / |m_3eq|  = ", str.format('{0:.2f}', 100*abs(m-m3eq)/abs(m3eq)), "%")


# Automatic tests can be run using pytest
# $ pytest melt_parameterisation.py

def test_heat_conservation():

    mp = ThreeEqMeltRateParam(S, T, p, z, u)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent) <= 1E-12


def test_salt_conservation():
    mp = ThreeEqMeltRateParam(S, T, p, z, u)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed) <= 1E-12


def test_heat_conservation_wout_fric_vel():
    mp = ThreeEqMeltRateParamWithoutFrictionVel(S, T, p, z)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent) <= 1E-12


def test_salt_conservation_wout_fric_vel():
    mp = ThreeEqMeltRateParamWithoutFrictionVel(S, T, p, z)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed) <= 1E-12


def test_heat_conservation_wout_fric_vel_qice():
    mp = ThreeEqMeltRateParamWithoutQice(S, T, p, z)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent) <= 1E-12


def test_salt_conservation_wout_fric_vel_qice():
    mp = ThreeEqMeltRateParamWithoutQice(S, T, p, z)
    QSbrine = -mp.rho0 * mp.wb * mp.Sb
    assert abs(QSbrine - -mp.QS_mixed) <= 1E-12


def test_heat_conservation_two_eq_param():
    mp = TwoEqMeltRateParam(S, T, p, z)
    Qlat = -mp.wb * mp.rho0 * mp.Lf
    assert abs(Qlat - mp.Q_latent) <= 1E-12


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
