import os
from subprocess import call

call(["mkdir", "mesh"])
os.chdir("mesh")
call(["rm", "ice_shelf_cavity_mesh.msh"])

f1 = open("ice_shelf_cavity_mesh.geo","w")

outline = [[0,0],[5000,0],[5000,100],[0,1]]
#resolution = [5 for i in range(len(outline))]
resolution = [0.1, 10, 10, 0.1]
print (len(outline), resolution)

def gmsh_generator(outline,resolution):
    for i in range (len(outline)):
        f1.write('Point('+str(i+1)+') = { ' +"{}, {}, 0, {}".format(outline[i][0], outline[i][1],resolution[i]) + "}; \n")

    for i in range (len(outline)-1):
        f1.write('Line('+str(i+1)+') = { ' +"{}, {}".format(len(outline)-i, len(outline)-1-i,) + "}; \n")

    # Final connection
    f1.write('Line('+str(len(outline))+') = { ' +"{}, {}".format(1 , len(outline),) + "}; \n")
    f1.write('Line Loop(1) = {')
    for i in range (len(outline)):
        f1.write(str(i+1))
        if i < len(outline)-1:
            f1.write(", ")
    f1.write('};\n')

    f1.write('Plane Surface(6) = {1};\n')
    for i in range (len(outline)):
        f1.write('Physical Line('+str(i+1)+') = { '+"{}".format(i+1) + "}; \n")

    f1.write('Physical Surface(11) = {6};\n' )
    f1.write('Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1')

    f1.close()

gmsh_generator(outline,resolution)

call(["gmsh", "ice_shelf_cavity_mesh.geo", "-2", "ice_shelf_cavity_mesh.msh"])

print("done")
