/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* in (2+1)D	 using explicit FDTD with second order splitting.                     *           *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/cu_helpers.h"

// Grid Parameters
#define XN	xn				   		// Number of x-spatial nodes
#define YN	yn						// Number of y-spatial nodes
#define TN	100						// Number of temporal nodes
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

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "gpu_fdtd_plot.m"
#define TIME_F argv[2]

// Index linearization
#define ind(i,j)  ((i)*XN+(j)) 		// [i  ,j  ] 

// Function prototypes 
__global__ void Re_lin_kernel(double *Re, double *Im, double dt, int xn, int yn, 
																	double dx, double dy);
__global__ void Im_lin_kernel(double *Re, double *Im, double dt, int xn, int yn, 
																	double dx, double dy);
__global__ void nonlin_kernel(double *Re, double *Im, double dt, int xn, int yn);

int main(void)
{
    // Timing info
	cudaEvent_t begin_event, end_event;
	cudaEventCreate(&begin_event);
	cudaEventCreate(&end_event);
    
	// Print basic info about simulation
	const int xn = atoi(argv[1]);
	const int yn = atoi(argv[1]);
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
	
	// Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate host arrays
	double *h_x 	= (double*)malloc(sizeof(double) * XN);
	double *h_y 	= (double*)malloc(sizeof(double) * YN);
	double *h_max 	= (double*)calloc((TN+1), sizeof(double));
	double *h_Re 	= (double*)malloc(sizeof(double) * XN * YN);
    double *h_Im	= (double*)malloc(sizeof(double) * XN * YN);   
	double *h_Re_0 	= (double*)malloc(sizeof(double) * XN * YN);
    double *h_Im_0	= (double*)malloc(sizeof(double) * XN * YN);   

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
	double *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(double) * XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(double) * XN*YN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(double) * XN*YN, cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(double) * XN*YN, cudaMemcpyHostToDevice));

	// Initialize the grid
	dim3 blocksPerGrid((XN+15)/16, (YN+15)/16, 1);
	dim3 threadsPerBlock(16, 16, 1);

	// Save max |psi| for printing
	#if MAX_PSI_CHECKING
	max_psi(d_Re, d_Im, h_max, 0, XN*YN);
	#endif // MAX_PSI_CHECKING

	// Timing starts here
	cudaEventRecord(begin_event, 0);

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
		max_psi(d_Re, d_Im, h_max, 0, XN*YN);
		#endif // MAX_PSI_CHECKING
		// Print time at specific intervals
	}
	float time_value;	
	cudaEventRecord(end_event, 0);
	cudaEventSynchronize(end_event);
	cudaEventElapsedTime(&time_value, begin_event, end_event);

	// Print time to file
	FILE *fp = fopen(TIME_F, "a");
	fprintf(fp, "%f, ", time_value);
	fclose(fp);

	// Copy results to device
	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(double)*XN*YN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(double)*XN*YN, 
															cudaMemcpyDeviceToHost));
	
	// Plot results
	m_plot_2d(h_Re_0, h_Im_0, h_Re, h_Im, h_max, LX, LY, XN, YN, TN, PLOT_F);

	// Clean up
	free(h_Re); 
	free(h_Im); 
	free(h_Re_0); 
	free(h_Im_0); 
	free(h_x); 
	free(h_y);
	free(h_max);
	CUDAR_SAFE_CALL(cudaFree(d_Re)); 
	CUDAR_SAFE_CALL(cudaFree(d_Im)); 
	
	return 0;
}

__global__ void Re_lin_kernel(double *Re, double *Im, double dt, int xn, int yn,
															double dx, double dy)
{                  
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
   // Avoid first and last point (boundary conditions)
   if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;
    
   Re[ind(i,j)] = Re[ind(i,j)] 
					- dt/(dx*dx)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
					- dt/(dy*dy)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

__global__ void Im_lin_kernel(double *Re, double *Im, double dt, int xn, int yn,
															double dx, double dy)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// Avoid first and last point (boundary conditions)
	if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;
	
	Im[ind(i,j)] = Im[ind(i,j)] 
					+ dt/(dx*dx)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
					+ dt/(dy*dy)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

__global__ void nonlin_kernel(double *Re, double *Im, double dt, int xn, int yn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// Avoid first and last point (boundary conditions)
	if(i == 0 || j == 0 || i >= xn-1 || j >= yn-1) return;

	double Rp = Re[ind(i,j)]; double Ip = Im[ind(i,j)];
	double A2 = Rp*Rp+Ip*Ip; 
	
	Re[ind(i,j)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	Im[ind(i,j)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}

