// nlse (1+1)D
#include "../lib/helpers.h"
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846264338327

// Grid Parameters
#define XN	256						// Number of x-spatial nodes
#define YN	256						// Number of y-spatial nodes
#define TN	10000					// Number of temporal nodes
#define LX	50.0					// x-spatial domain [-LX,LX)
#define LY	50.0					// y-spatial domain [-LY,LY)
#define TT	10.0            		// Max time
#define DX	(2*LX / XN)				// x-spatial step size
#define DY	(2*LY / YN)				// y-spatial step size
#define DT	(TT / TN)    			// temporal step size

#define ROOT 0

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   

// Index linearization
#define ind(i,j)  ((i)*XN+(j))			// [i  ,j  ] 

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
    
    alloc_local = fftw_mpi_local_size_2d(XN, YN, MPI_COMM_WORLD, &local_ni, &local_i_start);	
	printf("processor: %d, alloc_local: %td, local_i_start: %td. \n", rank, alloc_local, local_i_start);

	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex *psi_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN * YN);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN * YN);
	
	forward  = fftw_mpi_plan_dft_2d(XN, YN, psi, psi, MPI_COMM_WORLD,
									 FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_mpi_plan_dft_2d(XN, YN, psi, psi, MPI_COMM_WORLD,
									 FFTW_BACKWARD, FFTW_ESTIMATE);

	// generate full wave number
	double dkx=2*M_PI/XN/DX;
	double *kx=(double*)malloc(sizeof(double)*YN);
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
                                       
	// generate full x and y and psi_0 array
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
	
	
	// allocate and initialize the partial arrays
    double *k2 = (double*)malloc(sizeof(double) * alloc_local);
    double *max = (double*)calloc(TN, sizeof(double));
	
	for(int i = 0; i < local_ni; i++)
		for(int j = 0; j < YN; j++)
		{
			psi[ind(i,j)] = psi_0[ind(i+local_i_start,j)];
			k2[ind(i,j)] = kx[i+local_i_start]*kx[i+local_i_start] + ky[j]*ky[j];
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
		normalize(psi, XN*YN, local_ni);
		// nonlinear
		nonlin(psi, DT, local_ni);
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2, local_ni);
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN*YN, local_ni);
	}
	//printf("time elapsed: %f s.\n", elapsed);
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(psi, local_ni*YN, MPI_C_DOUBLE_COMPLEX, psi_new, 
	    					local_ni*YN, MPI_C_DOUBLE_COMPLEX, ROOT, MPI_COMM_WORLD);

	if(rank == ROOT)
		cm_plot_2d(psi_0, psi_new, max, LX, LY, XN, YN, TN, "mpi_2d_plotting.m");
	
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	fftw_free(psi_new);
	free(x);
	free(y);
	free(k2);
    free(kx);
    free(ky);

    MPI_Finalize();

	return 0;
}

void nonlin(fftw_complex *psi, double dt, ptrdiff_t end)
{                  
	double psi2;
	for(int i = 0; i < end; i++)
		for(int j = 0; j < YN; j++)
		{
    		psi2 = cabs(psi[ind(i,j)])*cabs(psi[ind(i,j)]);
			psi[ind(i,j)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j)];
		}
}

void lin(fftw_complex *psi, double *k2, double dt, ptrdiff_t end)
{                  
	for(int i = 0; i < end; i++)
		for(int j = 0; j < XN; j++)
    		psi[ind(i,j)] = cexp(-I * k2[ind(i,j)] * dt)*psi[ind(i,j)];
}

void normalize(fftw_complex *psi, int size, ptrdiff_t end)
{
	for (int i = 0; i < end*YN; i++)
		psi[i] = psi[i]/size;
}
