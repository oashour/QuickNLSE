/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* in (2+1)D	 using explicit FDTD with second order splitting.                     *           *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/cu_helpers.h"

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

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "gpu_fdtds_plot.m"
#define TIME_F "gpu_fdtds_time.m"

// Index linearization
#define ind(i,j)  ((i)*XN+(j)) 		// [i  ,j  ] 

// Function prototypes 
__global__ void Re_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, 
																	float dx, float dy);
__global__ void Im_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, 
																	float dx, float dy);
__global__ void nonlin_kernel(float *Re, float *Im, float dt, int xn, int yn);

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
	float *h_x 	= (float*)malloc(sizeof(float) * XN);
	float *h_y 	= (float*)malloc(sizeof(float) * YN);
	float *h_max 	= (float*)malloc(sizeof(float) * TN);
	float *h_Re 	= (float*)malloc(sizeof(float) * XN * YN);
    float *h_Im	= (float*)malloc(sizeof(float) * XN * YN);   
	float *h_Re_0 	= (float*)malloc(sizeof(float) * XN * YN);
    float *h_Im_0	= (float*)malloc(sizeof(float) * XN * YN);   

	// Initialize x and y
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

    // Initial Conditions on host
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				h_Re[ind(i,j)] = A_S*A*exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j])
															/(2*R*R*R_S*R_S)); 
				h_Im[ind(i,j)] = 0;
				h_Re_0[ind(i,j)] = h_Re[ind(i,j)];
				h_Im_0[ind(i,j)] = h_Im[ind(i,j)];
			}
	
	// Allocate device arrays and copy from host
	float *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(float) * XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(float) * XN*YN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(float) * XN*YN, cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(float) * XN*YN, cudaMemcpyHostToDevice));

	// Initialize the grid
	dim3 blocksPerGrid((XN+15)/16, (YN+15)/16, 1);
	dim3 threadsPerBlock(16, 16, 1);

	// Print timing info to file
	float time_value;
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	#if MAX_PSI_CHECKING
	max_psif(d_Re, d_Im, h_max, 0, XN*YN);
	#endif // MAX_PSI_CHECKING
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, 
																	XN, YN, DX, DY);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
        Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, 
																	XN, YN, DX, DY);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Solve nonlinear part
		nonlin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT, XN, YN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Solve linear part
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, 
																	XN, YN, DX, DY);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
        Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, 
																	XN, YN, DX, DY);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Save max |psi| for printing
		#if MAX_PSI_CHECKING
		max_psif(d_Re, d_Im, h_max, 0, XN*YN);
		#endif // MAX_PSI_CHECKING
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

	// Copy results to device
	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(float)*XN*YN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(float)*XN*YN, 
															cudaMemcpyDeviceToHost));
	
	// Plot results
	m_plot_2df(h_Re_0, h_Im_0, h_Re, h_Im, h_max, LX, LY, XN, YN, TN, PLOT_F);

	// Clean up
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

__global__ void Re_lin_kernel(float *Re, float *Im, float dt, int xn, int yn,
															float dx, float dy)
{                  
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
   // Avoid first and last point (boundary conditions)
   if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;
    
   Re[ind(i,j)] = Re[ind(i,j)] 
					- dt/(dx*dx)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
					- dt/(dy*dy)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

__global__ void Im_lin_kernel(float *Re, float *Im, float dt, int xn, int yn,
															float dx, float dy)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// Avoid first and last point (boundary conditions)
	if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;
	
	Im[ind(i,j)] = Im[ind(i,j)] 
					+ dt/(dx*dx)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
					+ dt/(dy*dy)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

__global__ void nonlin_kernel(float *Re, float *Im, float dt, int xn, int yn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// Avoid first and last point (boundary conditions)
	if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;

	float Rp = Re[ind(i,j)]; float Ip = Im[ind(i,j)];
	float A2 = Rp*Rp+Ip*Ip; 
	
	Re[ind(i,j)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	Im[ind(i,j)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}

