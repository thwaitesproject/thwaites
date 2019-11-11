from thwaites import *
from thwaites.meltrate_param import ThreeEqMeltRateParamWithoutQice, ThreeEqMeltRateParamWithoutFrictionVel
S = 34.
T = -2.6
p = 0
z = -1000.
u = 0.1

mp = ThreeEqMeltRateParam(S, T, p, z, u)

QTb = -mp.wb*mp.rho0*mp.Lf

print("Qlat here = ", QTb)
print("Qlat calculated = ", mp.Q_latent)
print("Qice = ", mp.Q_ice)
print("Qmixed", mp.Q_mixed)
print("m' = ", mp.wb)


mp = ThreeEqMeltRateParamWithoutFrictionVel(S, T, p, z)

QTb = -mp.wb*mp.rho0*mp.Lf
print("without u*... \n")
print("Qlat here = ", QTb)
print("Qlat calculated = ", mp.Q_latent)
print("Qice = ", mp.Q_ice)
print("Qmixed", mp.Q_mixed)
print("m' = ", mp.wb)



print("without Qice... \n")



mp = ThreeEqMeltRateParamWithoutQice(S, T, p, z)

QTb = -mp.wb*mp.rho0*mp.Lf

print("Qlat here = ", QTb)
print("Qlat calculated = ", mp.Q_latent)
print("Qice = ", mp.Q_ice)
print("Qmixed", mp.Q_mixed)
print("m' = ", mp.wb)

