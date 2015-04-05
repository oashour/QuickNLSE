/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
**********************************************************************************/
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	nodes					// Number of Fourier modes
#define TN	100						// Number of temporal nodes
#define LX	10.0					// x-spatial domain [-LX,LX)
#define TT	10.0            		// Max time
#define DX	(2*LX / XN)				// x-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "gpu_ffts_plot.m"
#define TIME_F argv[2]

// Function prototypes
__global__ void nonlin(cufftComplex *psi, float dt, int xn);
__global__ void lin(cufftComplex *psi, float *k2, float dt, int xn);
__global__ void normalize(cufftComplex *psi, int size);

int main(int argc, char **argv)
{                                                                          
    // Timing info
	cudaEvent_t begin_event, end_event;
	cudaEventCreate(&begin_event);
	cudaEventCreate(&end_event);
    
	// Print basic info about simulation
	const int nodes = atoi(argv[1]);
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
	
	// Allocate host arrays
    float *h_x = (float*)malloc(sizeof(float) * XN);
	float *h_k2 = (float*)malloc(sizeof(float) * XN);
	float *h_kx = (float*)malloc(XN * sizeof(float));
	cufftComplex *h_psi = (cufftComplex*)malloc(sizeof(cufftComplex)*XN);
	cufftComplex *h_psi_0 = (cufftComplex*)malloc(sizeof(cufftComplex)*XN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan1d(&plan, XN, CUFFT_C2C, 1));

    // Create wave number
	float dkx = 2*M_PI/XN/DX;
	for(int i = XN/2; i >= 0; i--) 
		h_kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		h_kx[i]=(i - XN) * dkx; 

	// Initial Conditions on host
	for(int i = 0; i < XN; i++)
		{
			h_x[i] = (i-XN/2)*DX;
			h_psi[i].x = sqrt(2)/cosh(h_x[i]);
			//h_psi[i].x = 2*exp(-(x[i]*x[i]/2.0/2.0));
			h_psi[i].y = 0;
			h_psi_0[i].x = h_psi[i].x;
			h_psi_0[i].y = h_psi[i].y;
			h_k2[i] = h_kx[i]*h_kx[i];
		}   
	
	// Allocate device arrays and copy from host
    cufftComplex *d_psi; float *d_k2;
	cudaMalloc((void **)&d_psi, sizeof(cufftComplex)*XN);
	cudaMalloc((void **)&d_k2, sizeof(float)*XN);
    cudaMemcpy(d_psi, h_psi, sizeof(cufftComplex)*XN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k2, h_k2, sizeof(float)*XN, cudaMemcpyHostToDevice);
	
	// Initialize the grid
	dim3 threadsPerBlock(128,1,1);
	dim3 blocksPerGrid((XN + 127)/128,1,1);

	// Forward transform
	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
	
	// Timing starts here
	cudaEventRecord(begin_event, 0);
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN);  
		// Backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// Normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN);
		// Solve nonlinear part
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT, XN);
		// Forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// Solve linear part
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN);  
	}
	float time_value;	
	cudaEventRecord(end_event, 0);
	cudaEventSynchronize(end_event);
	cudaEventElapsedTime(&time_value, begin_event, end_event);

	// Print time to file
	FILE *fp = fopen(TIME_F, "a");
	fprintf(fp, "%f, ", time_value);
	fclose(fp);
	
	// Backward transform to retreive data
	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
	// Normalize the transform
	normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN);
	
	// Copy results to device
	cudaMemcpy(h_psi, d_psi, sizeof(cufftComplex)*XN, cudaMemcpyDeviceToHost);
	
	// Plot results
	cm_plot_1df(h_psi_0, h_psi, LX, XN, PLOT_F);

	// Wrap up
	cufftDestroy(plan);
	free(h_x);
	free(h_k2);
	free(h_kx);
    free(h_psi_0);
	free(h_psi);
	cudaFree(d_psi);
	cudaFree(d_k2);
	
	return 0;
}

__global__ void nonlin(cufftComplex *psi, float dt, int xn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
    
	// Avoid first and last point (boundary conditions) (needs fixing)
	//if (i >= xn - 1 || i == 0) return; 
	if (i >= xn) return; 

	float psi2 = cuCabsf(psi[i])*cuCabsf(psi[i]);
	psi[i] = cuCmulf(psi[i], make_cuComplex(cos(psi2*dt), sin(psi2*dt)));
}

__global__ void lin(cufftComplex *psi, float *k2, float dt, int xn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	// Avoid first and last point (boundary conditions) (needs fixing)
	//if (i >= xn - 1 || i == 0) return; 
	if (i >= xn) return; 
	
	psi[i] = cuCmulf(psi[i], make_cuComplex(cos(k2[i]*dt), -sin(k2[i]*dt)));
}

__global__ void normalize(cufftComplex *psi, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 

	// Stay within range since grid might be larger
	if (i >= size) return; 

	psi[i].x = psi[i].x/size; psi[i].y = psi[i].y/size;
}
