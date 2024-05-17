all: isomip_3d_melt_test.log

isomip_3d_melt_test.log: isomip_plus_3d_extruded_layered.py
	echo "running $<" >&2
	python3 $< date 8000 15 6 1 1e-3 1e-3 900 9000 4500 --testing
	mv isomip_3d_melt_test.log ../../../tests/regression/

clean:
	rm -f isomip_3d_melt_test.log


