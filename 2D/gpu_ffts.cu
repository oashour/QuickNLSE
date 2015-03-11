/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	64				   		// Number of x-spatial nodes
#define YN	64						// Number of y-spatial nodes
#define TN	10000					// Number of temporal nodes
#define LX	50.0f					// x-spatial domain [-LX,LX)
#define LY	50.0f					// y-spatial domain [-LY,LY)
#define TT	10.0f            		// Max time
#define DX	(2*LX / XN)				// x-spatial step size
#define DY	(2*LY / YN)				// y-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0f/sqrt(8.0f))
#define  R_S 	(sqrt(32.0f/9.0f))
#define  A 		0.6f
#define  R 		(1.0f/(A*sqrt(1.0f-A*A)))   

// Index linearization
#define ind(i,j)  (i*XN+j)			// [i  ,j  ] 

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "gpu_ffts_plot.m"
#define TIME_F "gpu_ffts_time.m"

// Function prototypes
__global__ void nonlin(cufftComplex *psi, float dt, int xn, int yn);
__global__ void lin(cufftComplex *psi, float *k2, float dt, int xn, int yn);
__global__ void normalize(cufftComplex *psi, int size, int xn, int yn);

int main(void)
{                                                                          
    // Timing info
	cudaEvent_t begin_event, end_event;
	cudaEventCreate(&begin_event);
	cudaEventCreate(&end_event);
    
	// Timing starts here
	cudaEventRecord(begin_event, 0);
	
	// Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate host arrays
    float *h_x = (float*)malloc(sizeof(float) * XN);
	float *h_y = (float*)malloc(sizeof(float) * YN);
	float *h_kx = (float*)malloc(sizeof(float)*XN);
	float *h_ky = (float*)malloc(sizeof(float)*YN);
	float *h_max = (float*)calloc(TN+1, sizeof(float));
	//float *h_max = (float*)malloc(sizeof(float) * TN);
	float *h_k2 = (float*)malloc(sizeof(float) * XN*YN);
	cufftComplex *h_psi = (cufftComplex*)
											malloc(sizeof(cufftComplex) * XN*YN);
	cufftComplex *h_psi_0 = (cufftComplex*)
											malloc(sizeof(cufftComplex) * XN*YN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan2d(&plan, XN, YN, CUFFT_C2C));

    // Create wavenumbers
	float dkx = 2*M_PI/XN/DX;
	for(int i = XN/2; i >= 0; i--) 
		h_kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		h_kx[i]=(i - XN) * dkx; 

	float dky = 2*M_PI/YN/DY;
	for(int i = YN/2; i >= 0; i--) 
		h_ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		h_ky[i]=(i - YN) * dky; 

	// Initialize x and y
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

	// Initial conditions on host
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				h_psi[ind(i,j)].x = A_S*A*exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j])
															/(2*R*R*R_S*R_S));
				h_psi[ind(i,j)].y = 0;
				h_psi_0[ind(i,j)].x = h_psi[ind(i,j)].x;
				h_psi_0[ind(i,j)].y = h_psi[ind(i,j)].y;
				h_k2[ind(i,j)] = h_kx[i]*h_kx[i] + h_ky[j]*h_ky[j];
			}   
	
	// Allocate and copy device memory
    cufftComplex *d_psi; float *d_k2, *d_max;
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_psi, sizeof(cufftComplex)*XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_k2, sizeof(float)*XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_max, sizeof(float)*TN));
    CUDAR_SAFE_CALL(cudaMemcpy(d_psi, h_psi, sizeof(cufftComplex)*XN*YN, cudaMemcpyHostToDevice));
    CUDAR_SAFE_CALL(cudaMemcpy(d_k2, h_k2, sizeof(float)*XN*YN, cudaMemcpyHostToDevice));
	
	// Initialize the grid
	dim3 threadsPerBlock(16,16,1);
	dim3 blocksPerGrid((XN + 15)/16,(YN+15)/16,1);
	
	// Print timing info to file
	float time_value;
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");

	// Forward transform 
	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
	
	// Start time evolution
	for (int i = 1; i < TN; i++)
	{
		// Solve linear part
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN, YN);  
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// Normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN, XN, YN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Solve nonlinear part
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT, XN, YN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// Solve linear part
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN, YN);  
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Print time at specific intervals
		if(i % IRVL == 0)
		{
			cudaEventRecord(end_event, 0);
			cudaEventSynchronize(end_event);
			cudaEventElapsedTime(&time_value, begin_event, end_event);
			fprintf(fp, "%f, ", time_value);
		}
	}
	// Wrap up timing file 
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time/1000, '-*r');\n");
	fclose(fp);
	
	// Backward tranform to retreive data
	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
	// Normalize the transform
	normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN, XN, YN);
	#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
	
	// Copy results to device
	CUDAR_SAFE_CALL(cudaMemcpy(h_psi, d_psi, sizeof(cufftComplex)*XN*YN, cudaMemcpyDeviceToHost));
	
	// Plot results
	cm_plot_2df(h_psi_0, h_psi, h_max, LX, LY, XN, YN, TN, PLOT_F);

	// Clean up
	CUFFT_SAFE_CALL(cufftDestroy(plan));
	free(h_x);
	free(h_y);
	free(h_k2);
	free(h_kx);
	free(h_ky);
	free(h_psi);
	free(h_psi_0);
	free(h_max);
	CUDAR_SAFE_CALL(cudaFree(d_psi));
	CUDAR_SAFE_CALL(cudaFree(d_k2));
	CUDAR_SAFE_CALL(cudaFree(d_max));
	
	return 0;
}

__global__ void nonlin(cufftComplex *psi, float dt, int xn, int yn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	
	// Avoid first and last point (boundary conditions) (needs fixing)
	//if (i >= xn - 1 || j >= yn-1 || i == 0 || j == 0) return; 
    if (i >= xn || j >= yn) return;

	float psi2 = cuCabsf(psi[ind(i,j)])*cuCabsf(psi[ind(i,j)]);
    float non = psi2 - psi2*psi2;
	
	psi[ind(i,j)] = cuCmulf(psi[ind(i,j)], make_cuComplex(cos(non*dt), sin(non*dt)));
}

__global__ void lin(cufftComplex *psi, float *k2, float dt, int xn, int yn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// Avoid first and last point (boundary conditions) (needs fixing)
	//if (i >= xn - 1 || j >= yn-1 || i == 0 || j == 0) return; 
    if (i >= xn || j >= yn) return;
	
	psi[ind(i,j)] = cuCmulf(psi[ind(i,j)], 
						make_cuComplex(cos(k2[ind(i,j)]*dt), -sin(k2[ind(i,j)]*dt)));
}

__global__ void normalize(cufftComplex *psi, int size, int xn, int yn)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 

	// Stay within range since the grid might be larger
    if (i >= xn || j >= yn) return;
	
	psi[ind(i,j)].x = psi[ind(i,j)].x/size; psi[ind(i,j)].y = psi[ind(i,j)].y/size;
}

