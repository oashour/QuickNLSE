// nlse (1+1)D
#include "../lib/helpers.h"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846264338327

// given stuff
#define XN	1024		  // number of Fourier Modes
#define TN	10000		  // number of temporal nodes
#define L	10.0		  // Spatial Period
#define TT	10.0          // Max time
#define DX	(2*L / XN) 	  // spatial step size
#define DT	(TT / TN)     // temporal step size

#define ROOT 0
void nonlin(fftw_complex *psi, double dt, ptrdiff_t end);
void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end);
void normalize(fftw_complex *psi, int size, ptrdiff_t end);

int main(int argc, char **argv)
{
	ptrdiff_t alloc_local, local_ni, local_i_start, local_no, local_o_start;
    int rank, np;

	fftw_plan forward, backward;
	// double startwtime, endwtime;
    
	MPI_Init(&argc, &argv);
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    
    alloc_local = fftw_mpi_local_size_1d(XN, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE,
                                        &local_ni, &local_i_start, 
										&local_no, &local_o_start);
	
	printf("processor: %d, alloc_local: %td. \n", rank, alloc_local);

	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex *psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	
	forward  = fftw_mpi_plan_dft_1d(XN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_1d(XN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);

	// generate full wave number
	double dkx=2*M_PI/XN/DX;
	double *kx=(double*)malloc(XN*sizeof(double));
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2-i]=(XN/2-i)*dkx;
	for(int i = XN/2+1; i < XN; i++)
		kx[i]=(i-XN)*dkx; 

	// generate full x and psi_0 array
    double *x = (double*)malloc(sizeof(double) * XN);
	for (int i = 0; i < XN; i++)
	{
        x[i] = (i-XN/2)*DX;
		//psi_0[i] = sqrt(2.0)/(cosh(x[i])) + 0*I;  
		psi_0[i] = 4.0*exp(-(x[i]*x[i])/4.0/4.0) + 0*I;
	}
	
	// allocate and initialize the partial arrays
    double *k2 = (double*)malloc(sizeof(double) * local_ni);
	for (int i = 0; i < local_ni; i++)
	{
		k2[i] = kx[i+local_i_start]*kx[i+local_i_start];
		psi[i] = psi_0[i+local_i_start];  
	}
    
	// forward transform
	fftw_execute(forward);
	
	for (int i = 1; i < TN; i++)
	{
		// linear
		lin(psi, k2, DT/2, local_ni);  
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN, local_ni);
		// nonlinear
		nonlin(psi, DT, local_ni);
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2, local_ni);
	}
	
	// backward tranform
	fftw_execute(backward);
	// scale down
	normalize(psi, XN, local_ni);
	
	//printf("time elapsed: %f s.\n", elapsed);

    MPI_Gather(psi, local_ni, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);

	if(rank == ROOT)
		cm_plot_1d(psi_0, psi_new, L, XN, "mpifft.m");
	
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	fftw_free(psi_new);
	free(x);
	free(k2);
    free(kx);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t end)
{                  
	for(int i = 0; i < end; i++)
    	psi[i] = cexp(I * cabs(psi[i]) * cabs(psi[i]) * dt)*psi[i];
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end)
{                  
	for(int i = 0; i < end; i++)
    	psi[i] = cexp(-I * k2[i] * dt)*psi[i];
}

void normalize(fftw_complex *psi, int size, ptrdiff_t end)
{
	for (int i = 0; i < end; i++)
		psi[i] = psi[i]/size;
}
