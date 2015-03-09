/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation in      *
* (2+1)D using symmetric split step Fourier method.								  *		                           		  *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
**********************************************************************************/
#include "../lib/helpers.h"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846264338327

// Grid Parameters
#define XN	64						// Number of x-spatial nodes
#define YN	64						// Number of y-spatial nodes
#define TN	10000					// Number of temporal nodes
#define LX	50.0					// x-spatial domain [-LX,LX)
#define LY	50.0					// y-spatial domain [-LY,LY)
#define TT	10.0            		// Max time
#define DX	(2*LX / XN)				// x-spatial step size
#define DY	(2*LY / YN)				// y-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   

// MPI root
#define ROOT 0

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "mpi_fft_plot.m"
#define TIME_F "mpi_fft_time.m"

// Index linearization
#define ind(i,j)  ((i)*XN+(j))			// [i  ,j  ] 

// Function prototypes
void nonlin(fftw_complex *psi, double dt, ptrdiff_t local_ni, int yn, int rank, int p);
void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t local_ni, int yn, int rank, int p);
void normalize(fftw_complex *psi, int size, ptrdiff_t local_ni, int yn);

int main(int argc, char **argv)
{
    // MPI set up
    int rank, p;

	MPI_Init(&argc, &argv);
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    
	// Timing starts here
	double t1 = MPI_Wtime();
	
    // Print basic info about simulation
	if(rank == ROOT)
		printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
 
	// FFT grid set up
	ptrdiff_t alloc_local, local_ni, local_i_start;
    alloc_local = fftw_mpi_local_size_2d(XN, YN, MPI_COMM_WORLD, &local_ni, &local_i_start);	
	
	// Allocate the arrays
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	double *kx = (double*)malloc(sizeof(double) * local_ni); 
	double *ky = (double*)malloc(sizeof(double) * YN); 
    double *k2 = (double*)malloc(sizeof(double) * local_ni*YN);
    double *max = (double*)calloc(TN, sizeof(double));
	fftw_complex *psi_new, *psi_0;
	double *kx_0;
	
	// Create transform plans
	fftw_plan forward, backward;
	forward  = fftw_mpi_plan_dft_2d(XN, YN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_2d(XN, YN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);

	// Initial conditions on root
	if (rank == ROOT)
	{
		// Create x wave number
		double dkx=2*M_PI/XN/DX;
		kx_0 = (double*)malloc(sizeof(double) * XN);
		for(int i = XN/2; i >= 0; i--) 
			kx_0[XN/2-i]=(XN/2-i)*dkx;
		for(int i = XN/2+1; i < XN; i++)
			kx_0[i]=(i-XN)*dkx; 

		// Initial conditions
		psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN * YN);
		double *x = (double*)malloc(sizeof(double) * XN);
		double *y = (double*)malloc(sizeof(double) * YN);
		
		for (int i = 0; i < XN; i++)
			x[i] = (i-XN/2)*DX;
		
		for(int i = 0; i < YN ; i++)
			y[i] = (i-YN/2)*DY;
		
		for(int i = 0; i < XN; i++)
			for(int j = 0; j < YN; j++)
				psi_0[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])
													/(2*R*R*R_S*R_S)) + 0*I; 

		free(x); free(y); 
	}
	// Scatter the initial array to divide among processes
	MPI_Scatter(&psi_0[0], local_ni*YN, MPI_C_DOUBLE_COMPLEX, &psi[0], local_ni*YN, 
											MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);
	// Scatter the x wavenumber to divide among processes
	MPI_Scatter(&kx_0[0], local_ni, MPI_DOUBLE, &kx[0], local_ni, 
											MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Create y wavenumber
	double dky=2*M_PI/YN/DY;
	for(int i = YN/2; i >= 0; i--) 
		ky[YN/2-i]=(YN/2-i)*dky;
	for(int i = XN/2+1; i < XN; i++)
		ky[i]=(i-YN)*dky; 
	
	// Local initial conditions   
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < YN; j++)
			k2[ind(i,j)] = kx[i]*kx[i] + ky[j]*ky[j];
 	
	// Print timing info to file
	FILE *fp;
	if (rank == ROOT)
	{
		fp = fopen(TIME_F, "w");
		fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
		fprintf(fp, "time = [0, ");
	}

	// Forward transform
	fftw_execute(forward);
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin(psi, k2, DT/2, local_ni, YN, rank, p);  
		// Backward transform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN*YN, local_ni, YN);
		// Solve nonlinear part
		nonlin(psi, DT, local_ni, YN, rank, p);
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, local_ni, YN, rank, p);
		// Print time at specific intervals
		if (rank == ROOT)
			if (i % IRVL == 0)
				fprintf(fp, "%f, ", MPI_Wtime() - t1);
	}
	// Wrap up timing file
	if (rank == ROOT)
	{
		fprintf(fp, "];\n");
		fprintf(fp, "plot(steps, time, '-*r');\n");
		fclose(fp);
	}
	
	// Backward transform to retreive data
	fftw_execute(backward);
	// Normalize the transform
	normalize(psi, XN*YN, local_ni, YN);
	
    // Prepare new array for receiving results
	if (rank == ROOT)
		psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*XN*YN);

    // Gather the results into new array
	MPI_Gather(psi, local_ni*YN, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni*YN, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);
	// Plot results
	if (rank == ROOT)
	{
		cm_plot_2d(psi_0, psi_new, max, LX, LY, XN, YN, TN, PLOT_F);
	    fftw_free(psi_new); fftw_free(psi_0); free(kx_0);
	}
		
	// Clean up
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi);
	free(k2);
    free(kx);
    free(ky);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t local_ni, int yn, int rank, int p)
{                  
	double psi2;
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < yn; j++)
		{
			// Avoid boundary conditions (needs fixing)
			// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	
				// continue;
    		psi2 = cabs(psi[ind(i,j)])*cabs(psi[ind(i,j)]);
			psi[ind(i,j)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j)];
		}
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t local_ni, int yn, int rank, int p)
{                  
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < yn; j++)
		{
			// Avoid boundary conditions (needs fixing)
			// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	continue;
    		psi[ind(i,j)] = cexp(-I * k2[ind(i,j)] * dt)*psi[ind(i,j)];
		}
}

void normalize(fftw_complex *psi, int size, ptrdiff_t local_ni, int yn)
{
	for (int i = 0; i < local_ni; i++)
		for(int j = 0; j < yn; j++)
		psi[ind(i,j)] = psi[ind(i,j)]/size;
}

