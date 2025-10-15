all: geostrophic_spinup_3d_test.log

geostrophic_spinup_3d_test.log: geostrophic_spinup_3d.py clean
	echo "running $<" >&2
	mpiexec -np 2 python3 $< --T 345600 --testing
	mv geostrophic_spinup_3d_test.log ../../tests/regression/

clean:
	rm -f geostrophic_spinup_3d_test.log


