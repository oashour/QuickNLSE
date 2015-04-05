/**********************************************************************************
* Numerical Solution for the Cubic Nonlinear Schrodinger Equation in (1+1)D	 	  *
* using symmetric split step Fourier method		                           		  *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/helpers.h"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846264338327

// Grid parameters
#define XN	32		  // number of Fourier Modes
#define TN	10000		  // number of temporal nodes
#define L	10.0		  // Spatial Period
#define TT	10.0          // Max time
#define DX	(2*L / XN) 	  // spatial step size
#define DT	(TT / TN)     // temporal step size

// MPI root
#define ROOT 0

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "mpi_fft_plot.m"
#define TIME_F argv[1]

// Function Prototypes
void nonlin(fftw_complex *psi, double dt, ptrdiff_t end, int rank, int p);
void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end, int rank, int p);
void normalize(fftw_complex *psi, int size, ptrdiff_t end);

int main(int argc, char *argv[])
{
    // MPI set up
    int rank, p;

	MPI_Init(&argc, &argv);
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    
    // Print basic info about simulation
	if(rank == ROOT)
		printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
 
	// FFT grid set up
	ptrdiff_t alloc_local, local_ni, local_i_start, local_no, local_o_start;
    alloc_local = fftw_mpi_local_size_1d(XN, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE,
                                        &local_ni, &local_i_start,
										&local_no, &local_o_start);
	
	// Allocate the arrays
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	double *kx = (double*)malloc(sizeof(double) * local_ni);
	double *time = (double*)malloc(sizeof(double) * TN/IRVL);
	fftw_complex *psi_new, *psi_0;
	double *kx_0;
	
	// Create transform plans
	fftw_plan forward, backward;
	forward  = fftw_mpi_plan_dft_1d(XN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_1d(XN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);

	// Initial conditions on root
	if (rank == ROOT)
	{
		// Create wave numbers
		double dkx=2*M_PI/XN/DX;
		kx_0 = (double*)malloc(sizeof(double) * XN);
		for(int i = XN/2; i >= 0; i--) 
			kx_0[XN/2-i]=(XN/2-i)*dkx;
		for(int i = XN/2+1; i < XN; i++)
			kx_0[i]=(i-XN)*dkx; 

		// Initial conditions
		double *x = (double*)malloc(sizeof(double) * XN);
		psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
		for (int i = 0; i < XN; i++)
		{
			x[i] = (i-XN/2)*DX;
			psi_0[i] = sqrt(2.0)/(cosh(x[i])) + 0*I;  
			//psi_0[i] = 4.0*exp(-(x[i]*x[i])/4.0/4.0) + 0*I;
		}

		free(x);
	}
	// Scatter the initial array to divide among processes
	MPI_Scatter(&psi_0[0], local_ni, MPI_C_DOUBLE_COMPLEX, &psi[0], local_ni, 
											MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&kx_0[0], local_ni, MPI_DOUBLE, &kx[0], local_ni, 
											MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Local initial conditions   
    double *k2 = (double*)malloc(sizeof(double) * local_ni);
	
	for (int i = 0; i < local_ni; i++)
		k2[i] = kx[i]*kx[i];
    
	// Forward transform
	fftw_execute(forward);
	
	// Timing starts here
	double t1 = MPI_Wtime();
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin(psi, k2, DT/2, local_ni, rank, p);  
		// Backward transform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN, local_ni);
		// Solve nonlinear part
		nonlin(psi, DT, local_ni, rank, p);
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, local_ni, rank, p);
		// Print time at specific intervals
		if (rank == ROOT)
			if (i % IRVL == 0)
				time[i/IRVL-1] = MPI_Wtime()-t1;
	}
	// Backward transform to retreive data
	fftw_execute(backward);
	// Normalize the transform
	normalize(psi, XN, local_ni);
	
    // Prepare new array for receiving results
	if (rank == ROOT)
		psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);

    // Gather the results into new array
	MPI_Gather(psi, local_ni, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);

	// Plot results
	if(rank == ROOT)
	{
		// Plot timing results
		print_time(time, TN, IRVL, TIME_F);

		cm_plot_1d(psi_0, psi_new, L, XN, PLOT_F);
		fftw_free(psi_new); fftw_free(psi_0); free(kx_0);
	}

	// Clean up 
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi);
	free(k2);
	free(kx);
    free(time);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t end, int rank, int p)
{                  
	for(int i = 0; i < end; i++)
	{
		// Avoid boundary conditions (needs fixing)
		// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	continue;
    	
		psi[i] = cexp(I * cabs(psi[i]) * cabs(psi[i]) * dt)*psi[i];
	}
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end, int rank, int p)
{                  
	for(int i = 0; i < end; i++)
	{
		// Avoid boundary conditions (needs fixing)
		// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	continue;

		psi[i] = cexp(-I * k2[i] * dt)*psi[i];
	}
}

void normalize(fftw_complex *psi, int size, ptrdiff_t end)
{
	for (int i = 0; i < end; i++)
		psi[i] = psi[i]/size;
}

