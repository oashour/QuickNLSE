// Cubic Quintic Nonlinear Schrodinger Equation
#include <stddef.h>
#include "../lib/cu_helpers.h"

// Grid Parameters
#define XN	256				   		// Number of x-spatial nodes
#define YN	256						// Number of y-spatial nodes
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
#define ind(i,j)  ((i)*XN+(j)) 		// [i  ,j  ] 
// Function prototypes 
__global__ void Re_lin_kernel(float *Re, float *Im, float dt);
__global__ void Im_lin_kernel(float *Re, float *Im, float dt);
__global__ void nonlin_kernel(float *Re, float *Im, float dt);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));
    
	// Timing set up
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
 
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	
    // allocate x, y, R and I on host. Max will be use to store max |psi|
	float *h_x 	= (float*)malloc(sizeof(float) * XN);
	float *h_y 	= (float*)malloc(sizeof(float) * YN);
	float *h_max 	= (float*)malloc(sizeof(float) * TN);
	float *h_Re 	= (float*)malloc(sizeof(float) * XN * YN);
    float *h_Im	= (float*)malloc(sizeof(float) * XN * YN);   
	float *h_Re_0 	= (float*)malloc(sizeof(float) * XN * YN);
    float *h_Im_0	= (float*)malloc(sizeof(float) * XN * YN);   

	// initialize x and y.
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

    // Initial Conditions
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				h_Re[ind(i,j)] = A_S*A*exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j])
															/(2*R*R*R_S*R_S)); 
				h_Im[ind(i,j)] = 0;
				h_Re_0[ind(i,j)] = h_Re[ind(i,j)];
				h_Im_0[ind(i,j)] = h_Im[ind(i,j)];
			}
	
	// Allocate device arrays and copy from host.
	float *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(float) * XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(float) * XN*YN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(float) * XN*YN, cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(float) * XN*YN, cudaMemcpyHostToDevice));

	// Initialize the grid
	dim3 blocksPerGrid((XN+15)/16, (YN+15)/16, 1);
	dim3 threadsPerBlock(16, 16, 1);

	// print max |psi| for initial conditions
	max_psif(d_Re, d_Im, h_max, 0, XN*YN);
	// Begin timing
	cudaEventRecord(beginEvent, 0);
	for (int i = 1; i < TN; i++)
	{
		// linear
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// nonlinear
		nonlin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// linear
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// find max psi
		max_psif(d_Re, d_Im, h_max, i, XN*YN);
	}
	// End timing
	cudaEventRecord(endEvent, 0);

    cudaEventSynchronize(endEvent);
 
	float timeValue;
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
 
	printf("Time elapsed: %f.\n", timeValue);

	
	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(float)*XN*YN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(float)*XN*YN, 
															cudaMemcpyDeviceToHost));
	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	m_plot_2df(h_Re_0, h_Im_0, h_Re, h_Im, h_max, LX, LY, XN, YN, TN, "gpu_fdtds.m");

	// wrap up                                                  
	free(h_Re); 
	free(h_Im); 
	free(h_Re_0); 
	free(h_Im_0); 
	free(h_x); 
	free(h_y);
	free(h_max);
	cudaFree(d_Re); 
	cudaFree(d_Im); 
	
	return 0;
}

__global__ void Re_lin_kernel(float *Re, float *Im, float dt)
{                  
   // setting up i j
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
   // We're avoiding Boundary Elements (kept at initial value approx = 0)
   if(i == 0 || j == 0 || i >= XN-1 || j >= YN-1) return;
    
   Re[ind(i,j)] = Re[ind(i,j)] 
					- dt/(DX*DX)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
					- dt/(DY*DY)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

__global__ void Im_lin_kernel(float *Re, float *Im, float dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
    // We're avoiding Boundary Elements (kept at initial value approx = 0);
	if(i == 0 || j == 0 || i >= XN-1 || j >= YN-1) return;
	
	Im[ind(i,j)] = Im[ind(i,j)] 
					+ dt/(DX*DX)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
					+ dt/(DY*DY)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

__global__ void nonlin_kernel(float *Re, float *Im, float dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// we're avoiding Boundary Elements (kept at initial value approx = 0);
    if(i == 0 || j == 0 || i >= XN-1 || j >= YN-1) return;

	float Rp = Re[ind(i,j)]; float Ip = Im[ind(i,j)];
	float A2 = Rp*Rp+Ip*Ip; // |psi|^2
	
	Re[ind(i,j)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	Im[ind(i,j)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}
