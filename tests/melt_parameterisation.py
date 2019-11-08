from thwaites import *

S = 34.5
T = 0.3
p = 0
z = -1000.
u = 0.1

mp = ThreeEqMeltRateParam(S, T, p, z, u)

QTb = mp.wb*mp.rho0*mp.Lf

print(QTb)
print(mp.Q_latent)
print(mp.Q_ice)
print(mp.Q_mixed)

