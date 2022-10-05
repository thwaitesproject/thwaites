import gmsh
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



seabed_file = np.loadtxt("T1_seafloor.csv", delimiter=',',usecols=(2,3))
ice_file = np.loadtxt("T1_ice.csv", delimiter=',',usecols=(2,3))

print(seabed_file)

print(np.shape(seabed_file))

seabed_decimate = signal.decimate(seabed_file,30,ftype='fir', axis=0)
#seabed_decimate_1_y = signal.decimate(seabed`[:,1],5)

seabed = seabed_decimate[10:-10,:]

print(ice_file)

print(np.shape(ice_file))

ice_decimate = signal.decimate(ice_file,30,ftype='fir', axis=0)

ice = ice_decimate[10:-10,:]
print("ice", ice)
print("seabed", seabed)

print("shape(ice)", np.shape(ice))
print("shape(seabed)", np.shape(seabed))
# beginning of arrays is far from grounding line
# end of arrays is close to grounding line
# correct for vertical side walls
# left wall (open ocean)
print("ice", ice[0,0], "seabed", seabed[0,0])
if ice[0,0] < seabed[0,0]:
    ice = np.insert(ice,0, (seabed[0,0],ice[0,1]),0)
else:
    seabed = np.insert(seabed,0, (ice[0,0],seabed[0,1]),0)

# right wall grounding line
print("ice", ice[-1,0], "seabed", seabed[-1,0])
if ice[-1,0] < seabed[-1,0]:
    seabed = np.append(seabed, [[ice[-1,0],seabed[-1,1]]],0)
else:
    ice = np.append(ice, (seabed[-1,0],ice[-1,1]),0)
print("ice after vertical walls", ice)
print("seabed after vertical walsl", seabed)

print("shape(ice)", np.shape(ice))
print("shape(seabed)", np.shape(seabed))
gmsh.initialize()


gmsh.model.add("icefinT1")

lc = 1e-2

len_ice = len(ice)
len_seabed = len(seabed)

print("len_ice", len_ice)
print("len_seabed", len_seabed)
print("len_ice+len_seabed", len_ice+len_seabed)
# add ice points (left to right = open ocean to gl)
for i in range(len_ice):
    gmsh.model.geo.addPoint(ice[i,0], ice[i,1], 0, 5, i+1)


# add seabed points (left to right = open ocean to gl)
for i in range(len_seabed):
    gmsh.model.geo.addPoint(seabed[i,0], seabed[i,1], 0, 5, len_ice+i+1)


# add ice lines (left to right = open ocean to gl)
for i in range(len_ice-1):
    gmsh.model.geo.addLine(i+1,i+2, i+1)

# add line rhs (gl wall)
gmsh.model.geo.addLine(len_ice,len_ice+len_seabed, len_ice)

# add seabed lines (left to right = open ocean to gl)
for i in range(len_seabed-1):
    gmsh.model.geo.addLine(len_ice+i+1,len_ice+i+2, len_ice+i+1)

# add line lhs (open ocean wall)
gmsh.model.geo.addLine(len_ice+1,1, len_ice+len_seabed)


lines = []


# add ice lines to list
# left to right +ve

for i in range(len_ice-1):
    lines.append(i+1)

# add rhs gls 
lines.append(len_ice)

# add seabed lines backwards! grounding line to open ocean
for i in range(len_ice+len_seabed-1,len_ice,-1):
    lines.append(-i)

# add lhs open ocean
lines.append(len_ice+len_seabed)

# add curve loop
gmsh.model.geo.addCurveLoop(lines, 1)
#add surface
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [len_ice], 1)  # grounding line 
gmsh.model.addPhysicalGroup(1, [len_ice+len_seabed], 2)  # open ocean
gmsh.model.addPhysicalGroup(1, [len_ice+i+1 for i in range(len_seabed-1)], 3)  # seabed
gmsh.model.addPhysicalGroup(1, [i+1 for i in range(len_ice-1)], 4)  # ice-ocean

gmsh.model.addPhysicalGroup(2, [1], name = "My surface")

#gmsh.model.mesh.generate(2)

#gmsh.write("icefin_new_T1.msh")
gmsh.write("icefin_new_T1_decimate30.geo_unrolled")
