make mpi
#exemplu de rulare
#$1 file.bmp
#par 1 = option (grayscale,filter,noise,clear,blur)
#par 2 = thread_type (OMP,Pthreads)
#par 3 = thread_num (case Pthreads)
#par 4 = BLUR_LEVEL
export OMP_NUM_THREADS=2
mpirun -n 1 test_mpi $1 1 0 1 1
mpirun -n 1 test_mpi $1 1 1 2 1
