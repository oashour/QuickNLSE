/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
**********************************************************************************/
#include <sys/time.h>
#include <stddef.h>
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	2048					// Number of x-spatial nodes
#define TN	10000					// Number of temporal nodes
#define LX	10.0					// x-spatial domain [-LX,LX)
#define TT	10.0            		// Max time
#define DX	(2*LX / XN)				// x-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Error checking Macros
// Macro to catch cufft errors
#define CUFFT_SAFE_CALL( call) do {                                            \
    cufftResult err = call;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
        fprintf(stderr, "CUFFT error in file '%s' in line %i: %s, %d.\n",	   \
                __FILE__, __LINE__, _cudaGetErrorEnum(err), err);          	   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

// Function prototypes
__global__ void nonlin(cufftComplex *psi, double dt);
__global__ void lin(cufftComplex *psi, double *k2, double dt);
__global__ void normalize(cufftComplex *psi, int size);

int main(void)
{                                                                          
	// Allocate and initialize the arrays
    double *x = (double*)malloc(sizeof(double) * XN);
	double *h_k2 = (double*)malloc(sizeof(double) * XN);
	cufftComplex *h_psi = (cufftComplex*)
										malloc(sizeof(cufftComplex)*XN);
	cufftComplex *h_psi_0 = (cufftComplex*)
										malloc(sizeof(cufftComplex)*XN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan1d(&plan, XN, CUFFT_C2C, 1));

    // X and Y wave numbers
	double dkx = 2*M_PI/XN/DX;
	double *kx = (double*)malloc(XN * sizeof(double));
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		kx[i]=(i - XN) * dkx; 

	// initialize x.
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
	
	// Initial Conditions and square of wave number
	for(int i = 0; i < XN; i++)
		{
			h_psi[i].x = sqrt(2)/cosh(x[i]);
			//h_psi[i].x = 2*exp(-(x[i]*x[i]/2.0/2.0));
			h_psi[i].y = 0;
			h_psi_0[i].x = h_psi[i].x;
			h_psi_0[i].y = h_psi[i].y;
			h_k2[i] = kx[i]*kx[i];
		}   
	
	// Allocate and copy device memory
    cufftComplex *d_psi; double *d_k2;
	cudaMalloc((void **)&d_psi, sizeof(cufftComplex)*XN);
	cudaMalloc((void **)&d_k2, sizeof(double)*XN);
    cudaMemcpy(d_psi, h_psi, sizeof(cufftComplex)*XN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k2, h_k2, sizeof(double)*XN, cudaMemcpyHostToDevice);
	
	// initialize the grid
	dim3 threadsPerBlock(128,1,1);
	dim3 blocksPerGrid((XN + 127)/128,1,1);

	for (int i = 1; i < TN; i++)
	{
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN);
		// nonlinear calculation
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT);
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN);
	}

	cudaMemcpy(h_psi, d_psi, sizeof(cufftComplex)*XN, cudaMemcpyDeviceToHost);
	// plot results
	cm_plot_1df(h_psi_0, h_psi, LX, XN, "plottingf.m");

	// garbage collection
	cufftDestroy(plan);
	free(x);
	free(h_k2);
	free(kx);
    free(h_psi_0);
	free(h_psi);
	cudaFree(d_psi);
	cudaFree(d_k2);
	return 0;
}

__global__ void nonlin(cufftComplex *psi, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
    
	double psi2 = cuCabsf(psi[i])*cuCabsf(psi[i]);
    cufftComplex expo = make_cuComplex(cos(psi2*dt), sin(psi2*dt));
	psi[i] = cuCmulf(psi[i], expo);
}

__global__ void lin(cufftComplex *psi, double *k2, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
    cufftComplex expo = make_cuComplex(cos(k2[i]*dt), -sin(k2[i]*dt));
	psi[i] = cuCmulf(psi[i], expo);
}

__global__ void normalize(cufftComplex *psi, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 

	psi[i].x = psi[i].x/size; psi[i].y = psi[i].y/size;
}
