/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include <stddef.h>
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	64						// Number of x-spatial nodes        
#define YN	64						// Number of y-spatial nodes          
#define ZN	64						// Number of z-spatial nodes         
#define TN	1000					// Number of temporal nodes          
#define LX	50.0f					// x-spatial domain [-LX,LX)         
#define LY	50.0f					// y-spatial domain [-LY,LY)         
#define LZ	50.0f					// z-spatial domain [-LZ,LZ)         
#define TT	100.0f            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0f/sqrt(8.0f))
#define  R_S 	(sqrt(32.0f/9.0f))
#define  A 		0.6f
#define  R 		(1.0f/(A*sqrt(1.0f-A*A)))   
                                                                          
// Index linearization                                                    
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]                  
//#define ind(i,j,k) ((i) + XN * ((j) + YN * (k)))		                     
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

// Function prototypes
__global__ void nonlin(cufftComplex *psi, float dt);
__global__ void lin(cufftComplex *psi, float *k2, float dt);
__global__ void normalize(cufftComplex *psi, int size);

int main(void)
{                                                                          
	// Allocate and initialize the arrays
	float *h_k2 = (float*)malloc(sizeof(float) * XN * YN * ZN);
	cufftComplex *h_psi = (cufftComplex*)malloc(
							sizeof(cufftComplex) * XN * YN * ZN);
	cufftComplex *h_psi_0 = (cufftComplex*)malloc(
							sizeof(cufftComplex) * XN * YN * ZN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan3d(&plan, XN, YN, ZN, CUFFT_C2C));

    // X and Y wave numbers
	float dkx = 2*M_PI/XN/DX;
	float *h_kx = (float*)malloc(XN * sizeof(float));
	for(int i = XN/2; i >= 0; i--) 
		h_kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		h_kx[i]=(i - XN) * dkx; 

	float dky = 2*M_PI/YN/DY;
	float *h_ky = (float*)malloc(ZN * sizeof(float));
	for(int i = YN/2; i >= 0; i--) 
		h_ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		h_ky[i]=(i - YN) * dky; 

	float dkz = 2*M_PI/ZN/DZ;
	float *h_kz = (float*)malloc(ZN * sizeof(float));
	for(int i = ZN/2; i >= 0; i--) 
		h_kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		h_kz[i]=(i - ZN) * dkz; 
	
	// initialize x and y.
    float *h_x = (float*)malloc(sizeof(float) * XN);
	float *h_y = (float*)malloc(sizeof(float) * YN);
	float *h_z = (float*)malloc(sizeof(float) * ZN);
	
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

	for(int i = 0; i < ZN ; i++)
		h_z[i] = (i-ZN/2)*DZ;
	
	// Initial Conditions and square of wave number
	for (int k = 0; k < ZN; k++)
    	for(int j = 0; j < YN; j++)
			for(int i = 0; i < XN; i++)
			{
				h_psi[ind(i,j,k)].x = A_S*A*
							   		  exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j]+h_z[k]*h_z[k])
															/(2*R*R*R_S*R_S));
				h_psi[ind(i,j,k)].y = 0;
				h_psi_0[ind(i,j,k)].x = h_psi[ind(i,j,k)].x;
				h_psi_0[ind(i,j,k)].y = h_psi[ind(i,j,k)].y;
				h_k2[ind(i,j,k)] = h_kx[i]*h_kx[i] + h_ky[j]*h_ky[j] + h_kz[k]*h_kz[k];
			}   
	
	FILE* fp = fopen("gpu.text", "w");
	for(int i = 0; i < XN; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
			{
				fprintf(fp, "%f ", h_k2[ind(i,j,k)]);
			}
	fclose(fp);
	// Allocate and copy device memory
    cufftComplex *d_psi; float *d_k2;
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_psi, sizeof(cufftComplex)*XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_k2, sizeof(float)*XN*YN*ZN));
    CUDAR_SAFE_CALL(cudaMemcpy(d_psi, h_psi, sizeof(cufftComplex)*XN*YN*ZN,
															cudaMemcpyHostToDevice));
    CUDAR_SAFE_CALL(cudaMemcpy(d_k2, h_k2, sizeof(float)*XN*YN*ZN, 
															cudaMemcpyHostToDevice));
	
	// initialize the grid
	dim3 threadsPerBlock(8,8,8);
	dim3 blocksPerGrid((XN + 7)/8,(YN+7)/8,(ZN+7)/8);
	
	// Find max(|psi|) for initial pulse.
	float *h_max = (float*)calloc(TN, sizeof(float));
	//float *h_max = (float*)malloc(sizeof(float) * TN);
	//cmax_psi(psi, max, 0, XN*YN);
	for (int i = 1; i < TN; i++)
	{
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN*ZN);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// nonlinear calculation
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN*ZN);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		//cmax_psi(d_psi, h_max, i, XN*YN);
	}
	CUDAR_SAFE_CALL(cudaMemcpy(h_psi, d_psi, sizeof(cufftComplex)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));
	float *h_psi2 = (float*)malloc(sizeof(float)*XN*YN*ZN);
    float *h_psi2_0 = (float*)malloc(sizeof(float)*XN*YN*ZN);
	
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				h_psi2[ind(i,j,k)] = cuCabsf(h_psi[ind(i,j,k)]);
				h_psi2_0[ind(i,j,k)] = cuCabsf(h_psi_0[ind(i,j,k)]);
            }
	
	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	//vtk_3d(h_x, h_y, h_z, h_psi2, XN, YN, ZN, "test_fft1.vtk");
	//vtk_3d(h_x, h_y, h_z, h_psi2_0, XN, YN, ZN, "test_fft0.vtk");

	// garbage collection
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

__global__ void nonlin(cufftComplex *psi, float dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 

    if (i >= XN || j >= YN || k >= ZN) return;

	float psi2 = cuCabsf(psi[ind(i,j,k)])*cuCabsf(psi[ind(i,j,k)]);
    float non = psi2 - psi2*psi2;
	cufftComplex expo = make_cuComplex(cos(non*dt), sin(non*dt));
	psi[ind(i,j,k)] = cuCmulf(psi[ind(i,j,k)], expo);
}

__global__ void lin(cufftComplex *psi, float *k2, float dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 
    
    if (i >= XN || j >= YN || k >= ZN) return;
	
	cufftComplex expo = make_cuComplex(
								cos(k2[ind(i,j,k)]*dt), -sin(k2[ind(i,j,k)]*dt));
	psi[ind(i,j,k)] = cuCmulf(psi[ind(i,j,k)], expo);
}

__global__ void normalize(cufftComplex *psi, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	int k = threadIdx.z + blockIdx.z * blockDim.z; 

    if (i >= XN || j >= YN || k >= ZN) return;
	
	psi[ind(i,j,k)].x = psi[ind(i,j,k)].x/size; 
	psi[ind(i,j,k)].y = psi[ind(i,j,k)].y/size;
}

