#########################################################################
# PREPARE DIRECTORIES
#########################################################################
IDIR =../lib
ODIR=obj
BINDIR=bin
LDIR =../lib

#########################################################################
# PREPARE FILES
#########################################################################
# CPU FILES
_CPU_FDTD_O=cpu_fdtd.o
CPU_FDTD_O=$(patsubst %,$(ODIR)/%,$(_CPU_FDTD_O))
#_CPU_FFT_O=cpu_fft.o
#CPU_FFT_O=$(patsubst %,$(ODIR)/%,$(_CPU_FFT_O))

# GPU FILES
#_GPU_FDTD_O=gpu_fdtd.o
#GPU_FDTD_O=$(patsubst %,$(ODIR)/%,$(_GPU_FDTD_O))
#_GPU_FFT_O=gpu_fft.o
#GPU_FFT_O=$(patsubst %,$(ODIR)/%,$(_GPU_FFT_O))

# GPU SINGLE PRECISION FILES
#_GPU_FDTDS_O=cpu_fdtds.o
#GPU_FDTDS_O=$(patsubst %,$(ODIR)/%,$(_GPU_FDTDS_O))
#_GPU_FFTS_O=gpu_ffts.o
#GPU_FFTS_O=$(patsubst %,$(ODIR)/%,$(_GPU_FFTS_O))

# MPI FILES
#_MPI_FDTD_O=mpi_fdtd.o
#MPI_FDTD_O=$(patsubst %,$(ODIR)/%,$(_CPU_FDTD_O))
#_MPI_FFT_O=mpi_fft.o
#MPI_FFT_O=$(patsubst %,$(ODIR)/%,$(_CPU_FFT_O))

# LIBRARY FILES
_TIMERS_O=timers.o 
TIMERS_O=$(patsubst %,$(ODIR)/%,$(_TIMERS_O))
_HELPERS_O=helpers.o 
HELPERS_O=$(patsubst %,$(ODIR)/%,$(_HELPERS_O))
#_CU_HELPERS_O=cu_helpers.o 
#CU_HELPERS_O=$(patsubst %,$(ODIR)/%,$(_CU_HELPERS_O))

#########################################################################
# PREPARE TARGETS
#########################################################################
# CPU TARGETS
CPU_FDTD=$(CPU_FDTD_O) $(TIMERS_O) $(HELPERS_O)
#CPU_FFT=$(CPU_FFT_O) $(TIMERS_O) $(HELPERS_O)
# GPU TARGETS
#GPU_FDTD=$(GPU_FDTD_O) $(CU_HELPERS_O)
#GPU_FFT=$(GPU_FFT_O) $(CU_HELPERS_O)
# GPU SINGLE PRECISION TARGETS
#GPU_FDTDS=$(GPU_FDTDS_O) $(CU_HELPERS_O)
#GPU_FFTS=$(GPU_FFTS_O) $(CU_HELPERS_O)
# MPI TARGETS
#MPI_FDTD=$(MPI_FDTD_O) $(HELPERS_O)
#MPI_FFT=$(MPI_FFT_O) $(HELPERS_O)

#########################################################################
# RULES
#########################################################################
# cpu_fdtd= $(BINDIR)/cpu_fdtd.out
# gpu_fdtd= $(BINDIR)/gpu_fdtd.out
# gpu_fdtds= $(BINDIR)/gpu_fdtds.out
# mpi_fdtd= $(BINDIR)/mpi_fdtd.out

#default:
#	@echo "Error. Must specify build target"

$(BINDIR)/cpu_fdtd.out: $(CPU_FDTD)
	gcc -Wall -g -std=gnu99 -I$(IDIR) $^ -o $@ -lm 
#$(BINDIR)/cpu_fft.out: $(CPU_FFT)
#	gcc -Wall -std=gnu99 -I$(IDIR) $^ -o $@ -lm -lfftw3
#$(BINDIR)/gpu_fdtd.out: $(GPU_FDTD)
#	nvcc -arch compute_30 -I$(IDIR) $^ -o $@ -lm  
#$(BINDIR)/gpu_fft.out: $(GPU_FFT)
#	nvcc -arch compute_30 -I$(IDIR) $^ -o $@ -lm -lcufft

$(CPU_FDTD_O): CC=gcc  
$(CPU_FDTD_O): CFLAGS=-Wall -std=gnu99 -c -I$(IDIR) 
$(CPU_FDTD_O): _DEPS=helpers.h timers.h 
$(CPU_FDTD_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

#$(CPU_FFT_O): CC=gcc  
#$(CPU_FFT_O): CFLAGS=-Wall -std=gnu99 -c -I$(IDIR) 
#$(CPU_FFT_O): LIBS=-lm -lfftw3
#$(CPU_FFT_O): _DEPS=helpers.h timers.h 
#$(CPU_FFT_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

#$(GPU_FDTD_O): CC=nvcc  
#$(GPU_FDTD_O): CFLAGS=-arch compute_30 -I$(IDIR) 
#$(GPU_FDTD_O): _DEPS=cu_helpers.h 
#$(GPU_FDTD_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

#$(GPU_FFT_O): CC=nvcc  
#$(GPU_FFT_O): CFLAGS=-arch compute_30 -I$(IDIR) 
#$(GPU_FFT_O): LIBS=-lm -lcufft
#$(GPU_FFT_O): _DEPS=cu_helpers.h 
#$(GPU_FFT_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

$(TIMERS_O): CC=gcc  
$(TIMERS_O): CFLAGS=-Wall -std=gnu99 -c -I$(IDIR) 
$(TIMERS_O): _DEPS=timers.h 
$(TIMERS_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

$(HELPERS_O): CC=gcc  
$(HELPERS_O): CFLAGS=-Wall -std=gnu99 -c -I$(IDIR) 
$(HELPERS_O): _DEPS=helpers.h 
$(HELPERS_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

#$(CU_HELPERS_O): CC=nvcc  
#$(CU_HELPERS_O): CFLAGS=-arch compute_30 -c -I$(IDIR) 
#$(CU_HELPERS_O): LIBS=-lm
#$(CU_HELPERS_O): _DEPS=cu_helpers.h 
#$(CU_HELPERS_O): DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS)) 

VPATH=../lib
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core 
	rm -f $(BINDIR)/*.out *~ core 
