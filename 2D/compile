#!/bin/bash
if [ "$#" -lt 1 ]
then
	echo -e	"`basename $0` usage: include compile flag to debugging or not"
	echo -e	"(-debug[default]/-nodebug) [second arg], and which file(s) you want to compile:\n"
	echo
	echo "-a: all"
	echo "-gf: gpu_fdtd.cu"
	echo "-gfs: gpu_fdtds.cu"
	echo "-gs: gpu_fft.cu"
	echo "-gss: gpu_ffts.cu"
	echo "-cf: gpu_fdtd.cu"
	echo "-cs: gpu_fdtd.cu"
	echo "-mf: gpu_fdtd.cu"
	echo "-ms: gpu_fdtd.cu"
	echo "-m: all MPI"
	echo "-c: all CPU"
	echo "-g: all GPU"
fi

if [ "$2" = "-nodebug" ]
then
 	echo "Turning off error checking."
	sed -i "s/#define CUDA_ERROR_CHECKING 1/#define CUDA_ERROR_CHECKING 0/" ../lib/cu_helpers.h
fi

if [ "$1" = "-a" ]
then
	gcc cpu_fft.c ../lib/helpers.c ../lib/timers.c -o cpu_fft.out -std=c99 -Wall -lm  -lfftw3 && echo "cpu_fft.c compiled"
	gcc cpu_fdtd.c ../lib/helpers.c ../lib/timers.c -o cpu_fdtd.out -std=c99 -Wall -lm && echo "cpu_fdtd.c compiled" 
	mpicc mpi_fdtd.c ../lib/helpers.c  -o mpi_fdtd.out -std=c99 -Wall -lm   && echo "mpi_fdtd.c compiled" 
	mpicc mpi_fft.c ../lib/helpers.c -o mpi_fft.out -std=c99 -Wall -lm -lfftw3_mpi -lfftw3  && echo "mpi_fft.c compiled" 
	nvcc -arch compute_30 gpu_ffts.cu ../lib/cu_helpers.cu -o gpu_ffts.out -lcufft && echo "gpu_ffts.cu compiled" 
	nvcc -arch compute_30 gpu_fft.cu ../lib/cu_helpers.cu -o gpu_fft.out -lcufft && echo "gpu_fft.cu compiled" 
	nvcc -arch compute_30 gpu_fdtd.cu ../lib/cu_helpers.cu -o gpu_fdtd.out && echo "gpu_fdtd.cu compiled" 
	nvcc -arch compute_30 gpu_fdtds.cu ../lib/cu_helpers.cu -o gpu_fdtds.out && echo "gpu_fdtd.cu compiled" 
	echo "Compiled all!"
elif [ "$1" = "-gf" ]
then
	nvcc -arch compute_30 gpu_fdtd.cu ../lib/cu_helpers.cu -o gpu_fdtd.out && echo "gpu_fdtd.cu compiled" 
elif [ "$1" = "-gfs" ]
then
	nvcc -arch compute_30 gpu_fdtds.cu ../lib/cu_helpers.cu -o gpu_fdtds.out && echo "gpu_fdtd.cu compiled" 
elif [ "$1" = "-gs" ]
then
	nvcc -arch compute_30 gpu_fft.cu ../lib/cu_helpers.cu -o gpu_fft.out -lcufft && echo "gpu_fft.cu compiled" 
elif [ "$1" = "-gss" ]
then
	nvcc -arch compute_30 gpu_ffts.cu ../lib/cu_helpers.cu -o gpu_ffts.out -lcufft && echo "gpu_ffts.cu compiled" 
elif [ "$1" = "-cf" ]
then
	gcc cpu_fdtd.c ../lib/helpers.c ../lib/timers.c -o cpu_fdtd.out -std=c99 -Wall -lm && echo "cpu_fdtd.c compiled" 
elif [ "$1" = "-cs" ]
then
	gcc cpu_fft.c ../lib/helpers.c ../lib/timers.c -o cpu_fft.out -std=c99 -Wall -lm  -lfftw3 && echo "cpu_fft.c compiled"
elif [ "$1" = "-mf" ]
then
	mpicc mpi_fdtd.c ../lib/helpers.c  -o mpi_fdtd.out -std=c99 -Wall -lm   && echo "mpi_fdtd.c compiled" 
elif [ "$1" = "-ms" ]
then
	mpicc mpi_fft.c ../lib/helpers.c -o mpi_fft.out -std=c99 -Wall -lm -lfftw3_mpi -lfftw3  && echo "mpi_fft.c compiled" 
fi

if [ "$2" = "-nodebug" ]
then
 	echo "Reverting to default."
	sed -i "s/#define CUDA_ERROR_CHECKING 0/#define CUDA_ERROR_CHECKING 1/" ../lib/cu_helpers.h
fi

exit
