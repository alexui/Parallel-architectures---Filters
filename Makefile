build: test.c
	gcc test.c -Wall -o test
omp:
	gcc -fopenmp test.c -Wall -o test
threads:
	gcc -fopenmp test.c -Wall -o test -lpthread -g
mpi:
	mpicc -Wall -fopenmp test_mpi.c -o test_mpi -lpthread -g
clean:
	rm test 
