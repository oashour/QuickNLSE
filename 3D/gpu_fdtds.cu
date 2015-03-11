/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* in (2+1)D	 using explicit FDTD with second order splitting.                     *           *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/cu_helpers.h"

// Grid Parameters
#define XN	32						// Number of x-spatial nodes        
#define YN	32						// Number of y-spatial nodes          
#define ZN	32						// Number of z-spatial nodes         
#define TN	1000					// Number of temporal nodes          
#define LX	50.0f					// x-spatial domain [-LX,LX)         
#define LY	50.0f					// y-spatial domain [-LY,LY)         
#define LZ	50.0f					// z-spatial domain [-LZ,LZ)         
#define TT	10.0f            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0f/sqrt(8.0f))
#define  R_S 	(sqrt(32.0f/9.0f))
#define  A 		0.6f
#define  R 		(1.0f/(A*sqrt(1.0f-A*A)))   
                                                                          
// Timing parameters
#define IRVL	10				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0  "gpu_fdtds_0.vtk"
#define VTK_1  "gpu_fdtds_1.vtk"
#define TIME_F "gpu_fdtds_time.m"

// Index linearization                                                    
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]                  
#define ind(i,j,k) ((i) + XN * ((j) + YN * (k)))		                     
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
__global__ void Re_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn,
														float dx, float dy, float dz);
__global__ void Im_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn,
														float dx, float dy, float dz);
__global__ void nonlin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn);

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
	float *h_z = (float*)malloc(sizeof(float) * YN);
	float *h_max = (float*)calloc(TN+1, sizeof(float));
	float *h_Re = (float*)malloc(sizeof(float) * XN*YN*ZN);
    float *h_Im = (float*)malloc(sizeof(float) * XN*YN*ZN);   
	float *h_Re_0 = (float*)malloc(sizeof(float) * XN*YN*ZN);
    float *h_Im_0 = (float*)malloc(sizeof(float) * XN*YN*ZN);   
	
	// Initialize x, y and z
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

    for(int i = 0; i < ZN ; i++)
		h_z[i] = (i-ZN/2)*DZ; 
    
	// Initial Conditions on host
    for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				h_Re[ind(i,j,k)] = A_S*A*exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j]+h_z[k]*h_z[k])
																/(2*R*R*R_S*R_S)); 
				h_Im[ind(i,j,k)] = 0;
				h_Re_0[ind(i,j,k)] = h_Re[ind(i,j,k)];
				h_Im_0[ind(i,j,k)] = h_Im[ind(i,j,k)];
			}
	
	// Allocate device arrays and copy from host
	float *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(float) * XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(float) * XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(float) * XN*YN*ZN,
														cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(float) * XN*YN*ZN,
														cudaMemcpyHostToDevice));

	// Initialize the grid
	dim3 blocksPerGrid((XN+7)/8, (YN+7)/8, (ZN+7)/8);
	dim3 threadsPerBlock(8, 8, 8);
	
	// Print timing info to file
	float time_value;
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	#if MAX_PSI_CHECKING
	max_psif(d_Re, d_Im, h_max, 0, XN*YN*ZN);
	#endif // MAX_PSI_CHECKING
	// Start time evolution
	for (int i = 1; i < TN; i++)
	{
		// Solve linear part
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, XN, YN, ZN,
																			  DX, DY, DZ);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, XN, YN, ZN,
																			  DX, DY, DZ);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        #endif // CUDAR_ERROR_CHECKING
		// Solve nonlinear part
		nonlin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT, XN, YN, ZN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        #endif // CUDAR_ERROR_CHECKING
		// Solve linear part
		Re_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, XN, YN, ZN,
																			  DX, DY, DZ);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
        #endif // CUDAR_ERROR_CHECKING
		Im_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Re, d_Im, DT*0.5, XN, YN, ZN,
																			  DX, DY, DZ);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Save max |psi| for printing
		#if MAX_PSI_CHECKING
		max_psif(d_Re, d_Im, h_max, i, XN*YN*ZN);
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
	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(float)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(float)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));
    
	// Plot results
	vtk_3df(h_x, h_y, h_z, h_Re, h_Im, XN, YN, ZN, VTK_1);
	vtk_3df(h_x, h_y, h_z, h_Re_0, h_Im_0, XN, YN, ZN, VTK_0);
	
	// Clean up                                                  
	free(h_Re); 
	free(h_Im); 
	free(h_Re_0); 
	free(h_Im_0); 
	free(h_x); 
	free(h_y);
	free(h_z);
	free(h_max);
	CUDAR_SAFE_CALL(cudaFree(d_Re)); 
	CUDAR_SAFE_CALL(cudaFree(d_Im)); 
	
	return 0;
}

__global__ void Re_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn,
														float dx, float dy, float dz)
{                  
   	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Avoid first and last point (boundary conditions)
    if(i == 0 || j == 0 || k == 0 || i >= xn-1 || j >= yn-1 || k >= zn-1) return;
    
	Re[ind(i,j,k)] = Re[ind(i,j,k)] 
			- dt/(dx*dx)*(Im[ind(i+1,j,k)] - 2*Im[ind(i,j,k)] + Im[ind(i-1,j,k)])
			- dt/(dy*dy)*(Im[ind(i,j+1,k)] - 2*Im[ind(i,j,k)] + Im[ind(i,j-1,k)])
			- dt/(dz*dz)*(Im[ind(i,j,k+1)] - 2*Im[ind(i,j,k)] + Im[ind(i,j,k-1)]);
}

__global__ void Im_lin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn,
														float dx, float dy, float dz)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;

   	// We're avoiding Boundary Elements (kept at initial value approx = 0)
   	if(i == 0 || j == 0 || k == 0 || i >= xn-1 || j >= yn-1 || k >= zn-1) return;
   	
   	Im[ind(i,j,k)] = Im[ind(i,j,k)] 
   			+ dt/(dx*dx)*(Re[ind(i+1,j,k)] - 2*Re[ind(i,j,k)] + Re[ind(i-1,j,k)])
  			+ dt/(dy*dy)*(Re[ind(i,j+1,k)] - 2*Re[ind(i,j,k)] + Re[ind(i,j-1,k)])
   			+ dt/(dz*dz)*(Re[ind(i,j,k+1)] - 2*Re[ind(i,j,k)] + Re[ind(i,j,k-1)]);
}

__global__ void nonlin_kernel(float *Re, float *Im, float dt, int xn, int yn, int zn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;

   	// We're avoiding Boundary Elements (kept at initial value approx = 0)
   	if(i == 0 || j == 0 || k == 0 || i >= xn-1 || j >= yn-1 || k >= zn-1) return;
   	
	float Rp, Ip, A2;
	
	Rp = Re[ind(i,j,k)];  Ip = Im[ind(i,j,k)];
	A2 = Rp*Rp+Ip*Ip; // |psi|^2
	
	Re[ind(i,j,k)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	Im[ind(i,j,k)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}

