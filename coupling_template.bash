
NAME=test
MIXFILE=vertical_mix

# run the iteration for a sufficient number of iterations (e.g., 10)
for i in {0..10..1}
do
	# run HELIOS first
	python3 ./helios.py	-name ${NAME} \
				-opacity_mixing on-the-fly \
				-file_with_vertical_mixing_ratios ../your_chemistry_code_dir/output/${MIXFILE}_$i.txt \
				-coupling_mode yes \
				-coupling_iteration_step $i

	# stops iteration after convergence is found
	if (( $i > 0 )) 
	then
		STOP=$(<helios_main_dir/output/${NAME}/${NAME}_coupling_convergence.dat)
   		echo -e "--> Converged? ${STOP} (1 = yes, 0 = no)"	
   		if ((${STOP}==1))
   		then
   			break
   		fi
	fi

	# run here your photochemical kinetics code
	# --> read helios_main_dir/output/test/test_tp_coupling_$i.dat
	# --> and produce vertical_mix_(($i+1)).txt so that it can be read next iteration step by HELIOS
done
