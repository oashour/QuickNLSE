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

#define ROOT 0

// Grid Parameters
#define XN	32						// Number of x-spatial nodes        
#define YN	32						// Number of y-spatial nodes          
#define ZN  32						// Number of z-spatial nodes         
#define TN	1000   					// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	10.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0 "mpi_fft_0.vtk"
#define VTK_1 "mpi_fft_1.vtk"
#define TIME_F "mpi_fft_time.m"

// Index linearization                                                    
// Flat[((((x * Height) * Depth) + (y * Depth)) + z)] = Original[x, y, z]                  
#define ind(i,j,k) (((((i) * ZN) * YN) + ((j) * YN)) + (k))
//		   		 ____WIDTH____  
//		   		|_|_|_|_|_|_|_|H
//		   	 	|_|_|_|_|_|_|_|E
//		   	   Z|_|_|_|_|_|_|_|I
//		   	   N|_|_|_|_|_|_|_|G
//		   		|_|_|_|_|_|_|_|H
//		   	    |_|_|_|_|_|_|_|T
//		   	    \_\_\_\_\_\_\_\D
//               \_\_\_\_\_\_\_\E
//               Y\_\_\_\_\_\_\_\P
//                N\_\_\_\_\_\_\_\T
//					\_\_\_\_\_\_\_\H             
// 						  XN                          

// Function prototypes
void nonlin(fftw_complex *psi, double dt, ptrdiff_t local_ni, int yn, int zn,
																int rank, int p);
void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t local_ni, int yn, int zn,
																int rank, int p);
void normalize(fftw_complex *psi, int size, ptrdiff_t local_ni, int yn, int zn);

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
    alloc_local = fftw_mpi_local_size_3d(XN, YN, ZN, MPI_COMM_WORLD, 
												&local_ni, &local_i_start);	
	
	// Allocate the arrays
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	double *kx = (double*)malloc(sizeof(double) * local_ni); 
	double *ky = (double*)malloc(sizeof(double) * YN);
	double *kz = (double*)malloc(sizeof(double) * ZN);
    double *k2 = (double*)malloc(sizeof(double) * local_ni*YN*ZN);
    // double *max = (double*)calloc(TN, sizeof(double));
	fftw_complex *psi_new, *psi_0; 
	double *kx_0, *x, *y, *z;
	
	// Create transform plans
	fftw_plan forward, backward;
	forward  = fftw_mpi_plan_dft_3d(XN, YN, ZN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_3d(XN, YN, ZN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);
	
	// Initial conditions on root
	if (rank == ROOT)
	{
		// Create x wavenumber
		double dkx=2*M_PI/XN/DX;
		kx_0=(double*)malloc(sizeof(double)*XN);
		for(int i = XN/2; i >= 0; i--) 
			kx_0[XN/2-i]=(XN/2-i)*dkx;
		for(int i = XN/2+1; i < XN; i++)
			kx_0[i]=(i-XN)*dkx; 
	
		// Initial conditions
		psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN*YN*ZN);
		x = (double*)malloc(sizeof(double) * XN);
		y = (double*)malloc(sizeof(double) * YN);
		z = (double*)malloc(sizeof(double) * ZN);
		
		for(int i = 0; i < XN ; i++)
			x[i] = (i-XN/2)*DX;
		
		for(int i = 0; i < YN ; i++)
			y[i] = (i-YN/2)*DY;

		for(int i = 0; i < ZN ; i++)
			z[i] = (i-ZN/2)*DZ;
		
		for(int k = 0; k < ZN; k++)
			for(int j = 0; j < YN; j++)
				for(int i = 0; i < XN; i++)
						psi_0[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
															/(2*R*R*R_S*R_S)) + 0*I; 
	}
	// Scatter the initial array to divide among processes
	MPI_Scatter(&psi_0[0], local_ni*YN*ZN, MPI_C_DOUBLE_COMPLEX, &psi[0], local_ni*YN*ZN, 
											MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);
	// Scatter the x wavenumber to divide among processes
	MPI_Scatter(&kx_0[0], local_ni, MPI_DOUBLE, &kx[0], local_ni, 
											MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Create y and z wavenumbers
	double dky = 2*M_PI/YN/DY;
	for(int i = YN/2; i >= 0; i--) 
		ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		ky[i]=(i - YN) * dky; 
                                       
	double dkz = 2*M_PI/ZN/DZ;
	for(int i = ZN/2; i >= 0; i--) 
		kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		kz[i]=(i - ZN) * dkz; 
	
	// Local initial conditions   
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
				k2[ind(i,j,k)] = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];

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
		lin(psi, k2, DT/2, local_ni, YN, ZN, rank, p);  
		// Backward tranform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN*YN*ZN, local_ni, YN, ZN);
		// Solve nonlinear part
		nonlin(psi, DT, local_ni, YN, ZN, rank, p);
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, local_ni, YN, ZN, rank, p);
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
	normalize(psi, XN*YN*ZN, local_ni, YN, ZN);
	
    // Prepare new array for receiving results
	if (rank == ROOT)
		psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*XN*YN*ZN);

    // Gather the results into new array
	MPI_Gather(psi, local_ni*YN*ZN, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni*YN*ZN, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);

	// Plot results
	if(rank == ROOT)
	{
		vtk_3dc(x, y, z, psi_new, XN, YN, ZN, VTK_1 );
		vtk_3dc(x, y, z, psi_0, XN, YN, ZN, VTK_0);

		fftw_free(psi_0); fftw_free(psi_new); free(kx_0); free(x); free(y); free(z);
	}
	
	// Clean up
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi);
	free(k2);
    free(kx);
    free(ky);
	free(kz);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t local_ni, int yn, int zn, 
																	int rank, int p)
{                  
	double psi2;
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < yn; j++)
			for(int k = 0; k < zn; k++)
			{
				// Avoid boundary conditions (needs fixing)
				// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	
					// continue;
				psi2 = cabs(psi[ind(i,j,k)])*cabs(psi[ind(i,j,k)]);
				psi[ind(i,j,k)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j,k)];
			}
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t local_ni, int yn, int zn,
																		int rank, int p)
{                  
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < yn; j++)
			for(int k = 0; k < zn; k++)
			{
				// Avoid boundary conditions (needs fixing)
				// if(((i == 0) && (rank == ROOT)) || ((i == end-1) && (rank == p-1)))	
					// continue;
				psi[ind(i,j,k)] = cexp(-I * k2[ind(i,j,k)] * dt)*psi[ind(i,j,k)];
			}
}

void normalize(fftw_complex *psi, int size, ptrdiff_t local_ni, int yn, int zn)
{
	for (int i = 0; i < local_ni*yn*zn; i++)
		psi[i] = psi[i]/size;
}
