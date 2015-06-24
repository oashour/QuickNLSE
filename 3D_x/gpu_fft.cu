/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include <stddef.h>
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	xn						// Number of x-spatial nodes        
#define YN	yn						// Number of y-spatial nodes          
#define ZN	zn						// Number of z-spatial nodes         
#define TN	100						// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	10.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Index flattening macro
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]                  
#define ind(i,j,k) ((((i * ZN) * YN) + (j * YN)) + k)
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

// Timing parameters
#define IRVL	10				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0  "gpu_fft_0.vtk" 
#define VTK_1  "gpu_fft_1.vtk"
#define TIME_F argv[2]

// Function prototypes
__global__ void nonlin(cufftDoubleComplex *psi, double dt, int xn, int yn, int zn);
__global__ void lin(cufftDoubleComplex *psi, double *k2, double dt, int xn, int yn, int zn);
__global__ void normalize(cufftDoubleComplex *psi, int size, int xn, int yn, int zn);

int main(int argc, char *argv[])
{                                                                          
     // Timing info
	cudaEvent_t begin_event, end_event;
	cudaEventCreate(&begin_event);
	cudaEventCreate(&end_event);
    
	// Print basic info about simulation
	const int xn = atoi(argv[1]);
	const int yn = atoi(argv[1]);
	const int zn = atoi(argv[1]);
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
	
	// Allocate host arrays
    double *h_x = (double*)malloc(sizeof(double) * XN);
	double *h_y = (double*)malloc(sizeof(double) * YN);
	double *h_z = (double*)malloc(sizeof(double) * ZN);
	double *h_k2 = (double*)malloc(sizeof(double) * XN * YN * ZN);
	double *h_kx = (double*)malloc(XN * sizeof(double));
	double *h_ky = (double*)malloc(YN * sizeof(double));
	double *h_kz = (double*)malloc(ZN * sizeof(double));
	double *h_max = (double*)calloc(TN+1, sizeof(double));
	cufftDoubleComplex *h_psi = (cufftDoubleComplex*)malloc(
							sizeof(cufftDoubleComplex) * XN * YN * ZN);
	cufftDoubleComplex *h_psi_0 = (cufftDoubleComplex*)malloc(
							sizeof(cufftDoubleComplex) * XN * YN * ZN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan3d(&plan, XN, YN, ZN, CUFFT_Z2Z));

    // Create wavenumbers
	double dkx = 2*M_PI/XN/DX;
	for(int i = XN/2; i >= 0; i--) 
		h_kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		h_kx[i]=(i - XN) * dkx; 

	double dky = 2*M_PI/YN/DY;
	for(int i = YN/2; i >= 0; i--) 
		h_ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		h_ky[i]=(i - YN) * dky; 

	double dkz = 2*M_PI/ZN/DZ;
	for(int i = ZN/2; i >= 0; i--) 
		h_kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		h_kz[i]=(i - ZN) * dkz; 
	
	// Initialize x, y and z
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

	for(int i = 0; i < ZN ; i++)
		h_z[i] = (i-ZN/2)*DZ;
	
	// Initial conditions on host
	for(int i = 0; i < XN; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
			{
				h_psi[ind(i,j,k)].x = A_S*A*
							   		  exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j]+h_z[k]*h_z[k])
															/(2*R*R*R_S*R_S));
				h_psi[ind(i,j,k)].y = 0;
				h_psi_0[ind(i,j,k)].x = h_psi[ind(i,j,k)].x;
				h_psi_0[ind(i,j,k)].y = h_psi[ind(i,j,k)].y;
				h_k2[ind(i,j,k)] = h_kx[i]*h_kx[i] + h_ky[j]*h_ky[j] + h_kz[k]*h_kz[k];
			}   
	
	// Allocate and copy device memory
    cufftDoubleComplex *d_psi; double *d_k2;
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_psi, sizeof(cufftDoubleComplex)*XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_k2, sizeof(double)*XN*YN*ZN));
    CUDAR_SAFE_CALL(cudaMemcpy(d_psi, h_psi, sizeof(cufftDoubleComplex)*XN*YN*ZN,
															cudaMemcpyHostToDevice));
    CUDAR_SAFE_CALL(cudaMemcpy(d_k2, h_k2, sizeof(double)*XN*YN*ZN, 
															cudaMemcpyHostToDevice));
	
	// Initialize the grid
	dim3 threadsPerBlock(8,8,8);
	dim3 blocksPerGrid((XN + 7)/8,(YN+7)/8,(ZN+7)/8);
	
	// Find max(|psi|) for initial pulse.
	// cmax_psi(psi, h_max, 0, XN*YN*ZN);

	// Forward transform 
	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD));
	
	// Timing starts here
	cudaEventRecord(begin_event, 0);
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN, YN, ZN);  
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Backward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE));
		// Normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN*ZN, XN, YN, ZN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Solve nonlinear part 
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT, XN, YN, ZN);
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Forward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD));
		// Linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2, XN, YN, ZN);  
		#if CUDAR_ERROR_CHECKING
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		#endif // CUDAR_ERROR_CHECKING
		// Save max |psi| for printing
		//cmax_psi(psi, h_max, i, XN*YN*ZN);
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
	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE));
	// Normalize the transform
	normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN*ZN, XN, YN, ZN);
	CUDAR_SAFE_CALL(cudaPeekAtLastError());
	
	// Copy results to device
	CUDAR_SAFE_CALL(cudaMemcpy(h_psi, d_psi, sizeof(cufftDoubleComplex)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));
	double *h_psi2 = (double*)malloc(sizeof(double)*XN*YN*ZN);
    double *h_psi2_0 = (double*)malloc(sizeof(double)*XN*YN*ZN);
	
	// Plot results
	vtk_3dc(h_x, h_y, h_z, h_psi, XN, YN, ZN, VTK_1);
	vtk_3dc(h_x, h_y, h_z, h_psi_0, XN, YN, ZN, VTK_0);

	// Clean up 
	CUFFT_SAFE_CALL(cufftDestroy(plan));
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_k2);
	free(h_kx);
	free(h_ky);
	free(h_kz);
	free(h_psi);
	free(h_psi_0);
	free(h_max);
	CUDAR_SAFE_CALL(cudaFree(d_psi));
	CUDAR_SAFE_CALL(cudaFree(d_k2));
	
	return 0;
}

__global__ void nonlin(cufftDoubleComplex *psi, double dt, int xn, int yn, int zn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 

	// Avoid first and last point (boundary conditions) (needs fixing)
	// if (i >= xn - 1 || j >= yn-1 || || k >= zn-1 || i == 0 || j == 0 || k == 0) return; 
    if (i >= xn || j >= yn || k >= zn) return;

	double psi2 = cuCabs(psi[ind(i,j,k)])*cuCabs(psi[ind(i,j,k)]);
    double non = psi2 - psi2*psi2;
	psi[ind(i,j,k)] = cuCmul(psi[ind(i,j,k)], 
							make_cuDoubleComplex(cos(non*dt), sin(non*dt)));
}

__global__ void lin(cufftDoubleComplex *psi, double *k2, double dt, int xn, int yn, int zn)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 
    
	// Avoid first and last point (boundary conditions) (needs fixing)
	// if (i >= xn - 1 || j >= yn-1 || || k >= zn-1 || i == 0 || j == 0 || k == 0) return; 
    if (i >= xn || j >= yn || k >= zn) return;
	
	psi[ind(i,j,k)] = cuCmul(psi[ind(i,j,k)], 
				make_cuDoubleComplex(cos(k2[ind(i,j,k)]*dt), -sin(k2[ind(i,j,k)]*dt)));
}

__global__ void normalize(cufftDoubleComplex *psi, int size, int xn, int yn, int zn)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 

	// Stay within range since the grid might be larger
    if (i >= xn || j >= yn || k >= zn) return;
	
	psi[ind(i,j,k)].x = psi[ind(i,j,k)].x/size; 
	psi[ind(i,j,k)].y = psi[ind(i,j,k)].y/size;
}

