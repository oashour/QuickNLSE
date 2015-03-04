#!/bin/bash

gcc cpu_fft.c ../lib/helpers.c ../lib/timers.c -o cpu_fft.out -std=c99 -Wall -lm  -lfftw3 && echo "cpu_fft.c compiled"
gcc cpu_fdtd.c ../lib/helpers.c ../lib/timers.c -o cpu_fdtd.out -std=c99 -Wall -lm && echo "cpu_fdtd.c compiled" 
mpicc mpi_fdtd.c ../lib/helpers.c  -o mpi_fdtd.out -std=c99 -Wall -lm   && echo "mpi_fdtd.c compiled" 
mpicc mpi_fft.c ../lib/helpers.c -o mpi_fft.out -std=c99 -Wall -lm -lfftw3_mpi -lfftw3  && echo "mpi_fft.c compiled" 
nvcc -arch compute_30 gpu_ffts.cu ../lib/cu_helpers.cu -o gpu_ffts.out -lcufft && echo "gpu_ffts.cu compiled" 
nvcc -arch compute_30 gpu_fft.cu ../lib/cu_helpers.cu -o gpu_fft.out -lcufft && echo "gpu_fft.cu compiled" 
nvcc -arch compute_30 gpu_fdtd.cu ../lib/cu_helpers.cu -o gpu_fdtd.out && echo "gpu_fdtd.cu compiled" 
nvcc -arch compute_30 gpu_fdtds.cu ../lib/cu_helpers.cu -o gpu_fdtds.out && echo "gpu_fdtd.cu compiled" 
