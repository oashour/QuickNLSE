// nlse
#include "../lib/cu_helpers.h"

// given stuff
#define XN	1000
#define TN	100000
#define L	10.0
#define TT	10.0

// calculated from given
#define DX	(2*L / XN)
#define DT	(TT / TN)

// Gaussian Pulse Parameters
#define A 1.0
#define R 2.0

__global__ void R_lin_kernel(float *Re, float *Im, float dt);
__global__ void I_lin_kernel(float *Re, float *Im, float dt);
__global__ void nonlin_kernel(float *Re, float *Im, float dt);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));

    cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
 
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	
	// create the arrays x and t
    float *h_x = (float*)malloc(sizeof(float) * XN);
    // create used arrays
	float *h_Re 	= (float*)malloc(sizeof(float) * XN);
    float *h_Im	= (float*)malloc(sizeof(float) * XN);   
	float *h_Re_0 	= (float*)malloc(sizeof(float) * XN);
    float *h_Im_0	= (float*)malloc(sizeof(float) * XN);   
	// initial conditions.
	for(int i = 0; i < XN ; i++)
	{
		h_x[i] = (i-XN/2)*DX;
		h_Re[i]	= sqrt(2.0)/(cosh(h_x[i]));	// initial
		h_Im[i]	= 0;       		 				// initial
		//h_Re[i]	= 2*exp(-(h_x[i]*h_x[i])/2.0/2.0);	// initial
		h_Im_0[i] = h_Im[i];
		h_Re_0[i] = h_Re[i];
	}
    
    // allocate arrays on device and copy them
	float *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(float) * XN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(float) * XN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(float) * XN, cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(float) * XN, cudaMemcpyHostToDevice));

	// initialize the grid
	dim3 threadsPerBlock(128,1,1);
	dim3 blocksPerGrid((XN + 127)/128,1,1);

	// solve 
	cudaEventRecord(beginEvent, 0);
	for (int i = 1; i < TN; i++)
	{
		// linear
		R_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        I_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// nonlinear
		nonlin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// linear
		R_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        I_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
	}
	cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
	float timeValue;
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
	printf("%f\n", timeValue/1000.0);

	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(float)*XN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(float)*XN, 
															cudaMemcpyDeviceToHost));
	m_plot_1df(h_Re_0, h_Im_0, h_Re, h_Im, L, XN, "gpu_fdtd.m");
	// wrap up
	free(h_Re); 
	free(h_Im); 
	free(h_Re_0); 
	free(h_Im_0); 
	free(h_x); 
	CUDAR_SAFE_CALL(cudaFree(d_Re)); 
	CUDAR_SAFE_CALL(cudaFree(d_Im)); 

	return 0;
}

__global__ void R_lin_kernel(float *Re, float *Im, float dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	if (i >= XN - 1 || i == 0) return; // avoid first and last elements

	Re[i] = Re[i] - dt/(DX*DX)*(Im[i+1] - 2*Im[i] + Im[i-1]);
}

__global__ void I_lin_kernel(float *Re, float *Im, float dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	if (i >= XN - 1 || i == 0) return; // avoid first and last elements

	Im[i] = Im[i] + dt/(DX*DX)*(Re[i+1] - 2*Re[i] + Re[i-1]);
}

__global__ void nonlin_kernel(float *Re, float *Im, float dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float Rp = Re[i]; float Ip = Im[i];
	float A2 = Rp*Rp+Ip*Ip;
	
	if (i > XN - 1) return; 
	
	Re[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
	Im[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
}

