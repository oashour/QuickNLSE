#!/bin/bash

mpirun -np 4 ./mpi_fft.out && echo "Finished running mpi_fft.out" 
mpirun -np 4 ./mpi_fdtd.out && echo "Finished running mpi_fdtd.out"
./cpu_fft.out && echo "Finished running cpu_fft.out"  
./cpu_fdtd.out && echo "Finished running cpu_fdtd.out" 
./gpu_fft.out && echo "Finished running gpu_fft.out" 
./gpu_ffts.out && echo "Finished running gpu_ffts.out" 
./gpu_fdtd.out && echo "Finished running gpu_fdtd.out" 
./gpu_fdtds.out && echo "Finished running gpu_fdtds.out" 
