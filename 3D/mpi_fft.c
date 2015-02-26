// nlse (1+1)D
#include "../lib/helpers.h"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846264338327
#define ROOT 0
// Grid Parameters
#define XN	64						// Number of x-spatial nodes        
#define YN	64						// Number of y-spatial nodes          
#define ZN  64						// Number of z-spatial nodes         
#define TN	1000						// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	100.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Index linearization                                                    
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]                  
// ((((x * Height) * Depth) + (y * Depth)) + z) // NEWWWWW
//#define ind(i,j,k) ((i) + XN * ((j) + YN * (k)))		                     
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
void nonlin(fftw_complex *psi, double dt, ptrdiff_t end);
void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end);
void normalize(fftw_complex *psi, int size, ptrdiff_t end);

int main(int argc, char **argv)
{
	ptrdiff_t alloc_local, local_ni, local_i_start;
    int rank, np;

	fftw_plan forward, backward;
	// double startwtime, endwtime;
    
	MPI_Init(&argc, &argv);
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    
    alloc_local = fftw_mpi_local_size_3d(XN, YN, ZN, MPI_COMM_WORLD, 
												&local_ni, &local_i_start);	
	printf("processor: %d, alloc_local: %td, local_i_start: %td. local_ni: %td \n",
									rank, alloc_local, local_i_start, local_ni);

	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex *psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *XN*YN*ZN);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *XN*YN*ZN);
	
	forward  = fftw_mpi_plan_dft_3d(XN, YN, ZN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_3d(XN, YN, ZN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);
 	MPI_Barrier(MPI_COMM_WORLD);   
	
	// generate full wave number
	double dkx=2*M_PI/XN/DX;
	double *kx=(double*)malloc(sizeof(double)*XN);
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2-i]=(XN/2-i)*dkx;
	for(int i = XN/2+1; i < XN; i++)
		kx[i]=(i-XN)*dkx; 
	
	double dky = 2*M_PI/YN/DY;
	double *ky = (double*)malloc(sizeof(double) * YN);
	for(int i = YN/2; i >= 0; i--) 
		ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		ky[i]=(i - YN) * dky; 
                                       
	double dkz = 2*M_PI/ZN/DZ;
	double *kz = (double*)malloc(sizeof(double)*ZN);
	for(int i = ZN/2; i >= 0; i--) 
		kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		kz[i]=(i - ZN) * dkz; 
	
 	MPI_Barrier(MPI_COMM_WORLD);   
	
	// generate full x and y and psi_0 array
    double *x = (double*)malloc(sizeof(double) * XN);
    double *y = (double*)malloc(sizeof(double) * YN);
	double *z = (double*)malloc(sizeof(double) * ZN);
	
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

	for(int i = 0; i < ZN ; i++)
		z[i] = (i-ZN/2)*DZ;
	
	for(int i = 0; i < XN; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
					psi_0[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
														/(2*R*R*R_S*R_S)) + 0*I; 
	
 	MPI_Barrier(MPI_COMM_WORLD);   
	// allocate and initialize the partial arrays
    double *k2 = (double*)malloc(sizeof(double) * alloc_local);
	
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
			{
				psi[ind(i,j,k)] = psi_0[ind(i+local_i_start,j,k)];
				k2[ind(i,j,k)] = kx[i+local_i_start]*kx[i+local_i_start] 
				  					+ ky[j]*ky[j] + kz[k]*kz[k];
			}

 	MPI_Barrier(MPI_COMM_WORLD);   
	for (int i = 1; i < TN; i++)
	{
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2, local_ni);  
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN*YN*ZN, local_ni);
		// nonlinear
		nonlin(psi, DT, local_ni);
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2, local_ni);
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN*YN*ZN, local_ni);
	}
	//printf("time elapsed: %f s.\n", elapsed);
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(psi, local_ni*YN*ZN, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni*YN*ZN, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);

	if(rank == ROOT)
	{
		double *psi2 = (double*)malloc(sizeof(double)*XN*YN*ZN);
		double *psi2_0 = (double*)malloc(sizeof(double)*XN*YN*ZN);
		
		for(int k = 0; k < ZN; k++)
			for(int j = 0; j < YN; j++)
				for(int i = 0; i < XN; i++)
				{
					psi2[ind(i,j,k)] = cabs(psi_new[ind(i,j,k)]);
					psi2_0[ind(i,j,k)] = cabs(psi_0[ind(i,j,k)]);
				}
	

		// Generate MATLAB file to plot max |psi| and the initial and final pulses
		vtk_3d(x, y, z, psi2, XN, YN, ZN, "test_mfft1.vtk");
		vtk_3d(x, y, z, psi2_0, XN, YN, ZN, "test_mfft0.vtk");

		free(psi2); free(psi2_0);
	}
	
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	fftw_free(psi_new);
	free(x);
	free(y);
	free(z);
	free(k2);
    free(kx);
    free(ky);
	free(kz);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t end)
{                  
	double psi2;
	for(int i = 0; i < end; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
			{
				psi2 = cabs(psi[ind(i,j,k)])*cabs(psi[ind(i,j,k)]);
				psi[ind(i,j,k)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j,k)];
			}
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end)
{                  
	for(int i = 0; i < end; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
				psi[ind(i,j,k)] = cexp(-I * k2[ind(i,j,k)] * dt)*psi[ind(i,j,k)];
}

void normalize(fftw_complex *psi, int size, ptrdiff_t end)
{
	for (int i = 0; i < end*YN*ZN; i++)
		psi[i] = psi[i]/size;
}
