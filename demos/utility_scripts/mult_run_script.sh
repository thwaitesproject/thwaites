#!/bin/bash
#mpirun.openmpi -np 4 python iSWING_delT3.0.py
mpirun.openmpi -np 4 python ./iSWING_delT_minus3.0.py
mpirun.openmpi -np 4 python ./iSWING_delT_minus2.0.py
mpirun.openmpi -np 4 python ./iSWING_del_Tminus1.0.py
#mpirun.openmpi -np 4 python iSWING_delT3.0.py

