/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include <stddef.h>
#include "../lib/cu_helpers.h"
#include <cufft.h>

// Grid Parameters
#define XN	256				   		// Number of x-spatial nodes
#define YN	256						// Number of y-spatial nodes
#define TN	1000					// Number of temporal nodes
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

// Index linearization
#define ind(i,j)  (i*XN+j)			// [i  ,j  ] 

// Function prototypes
__global__ void nonlin(cufftDoubleComplex *psi, double dt);
__global__ void lin(cufftDoubleComplex *psi, double *k2, double dt);
__global__ void normalize(cufftDoubleComplex *psi, int size);

int main(void)
{                                                                          
	// Allocate and initialize the arrays
    double *h_x = (double*)malloc(sizeof(double) * XN);
	double *h_y = (double*)malloc(sizeof(double) * YN);
	double *h_kx = (double*)malloc(sizeof(double)*XN);
	double *h_ky = (double*)malloc(sizeof(double)*YN);
	double *h_max = (double*)calloc(TN, sizeof(double));
	//double *h_max = (double*)malloc(sizeof(double) * TN);
	double *h_k2 = (double*)malloc(sizeof(double) * XN * YN);
	cufftDoubleComplex *h_psi = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * XN * YN);
	cufftDoubleComplex *h_psi_0 = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * XN * YN);
	
	// Create transform plans
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan2d(&plan, XN, YN, CUFFT_Z2Z));

    // X and Y wave numbers
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

	// initialize x and y.
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

	// Initial Conditions and square of wave number
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
    cufftDoubleComplex *d_psi; double *d_k2, *d_max;
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_psi, sizeof(cufftDoubleComplex)*XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_k2, sizeof(double)*XN*YN));
	CUDAR_SAFE_CALL(cudaMalloc((void **)&d_max, sizeof(double)*TN));
    CUDAR_SAFE_CALL(cudaMemcpy(d_psi, h_psi, sizeof(cufftDoubleComplex)*XN*YN, cudaMemcpyHostToDevice));
    CUDAR_SAFE_CALL(cudaMemcpy(d_k2, h_k2, sizeof(double)*XN*YN, cudaMemcpyHostToDevice));
	
	// initialize the grid
	dim3 threadsPerBlock(16,16,1);
	dim3 blocksPerGrid((XN + 15)/16,(YN+15)/16,1);
	
	// Find max(|psi|) for initial pulse.
	//cmax_psi(psi, max, 0, XN*YN);
	for (int i = 1; i < TN; i++)
	{
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// nonlinear calculation
		nonlin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, DT);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// forward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD));
		// linear calculation
		lin<<<blocksPerGrid, threadsPerBlock>>>(d_psi, d_k2, DT/2);  
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		// backward transform
    	CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE));
		// normalize the transform
		normalize<<<blocksPerGrid, threadsPerBlock>>>(d_psi, XN*YN);
		CUDAR_SAFE_CALL(cudaPeekAtLastError());
		//cmax_psi(d_psi, d_max, 0, XN*YN);
	}
	CUDAR_SAFE_CALL(cudaMemcpy(h_psi, d_psi, sizeof(cufftDoubleComplex)*XN*YN, cudaMemcpyDeviceToHost));
	//CUDAR_SAFE_CALL(cudaMemcpy(h_max, d_max, sizeof(double)*TN, cudaMemcpyDeviceToHost));
	// plot results
	cm_plot_2d(h_psi_0, h_psi, h_max, LX, LY, XN, YN, TN, "gpufft.m");

	// garbage collection
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

__global__ void nonlin(cufftDoubleComplex *psi, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	
    if (i >= XN || j >= YN) return;

	double psi2 = cuCabs(psi[ind(i,j)])*cuCabs(psi[ind(i,j)]);
    double non = psi2 - psi2*psi2;
	cufftDoubleComplex expo = make_cuDoubleComplex(cos(non*dt), sin(non*dt));
	psi[ind(i,j)] = cuCmul(psi[ind(i,j)], expo);
}

__global__ void lin(cufftDoubleComplex *psi, double *k2, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
    if (i >= XN || j >= YN) return;
	
	cufftDoubleComplex expo = make_cuDoubleComplex(
								cos(k2[ind(i,j)]*dt), -sin(k2[ind(i,j)]*dt));
	psi[ind(i,j)] = cuCmul(psi[ind(i,j)], expo);
}

__global__ void normalize(cufftDoubleComplex *psi, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 

    if (i >= XN || j >= YN) return;
	
	psi[ind(i,j)].x = psi[ind(i,j)].x/size; psi[ind(i,j)].y = psi[ind(i,j)].y/size;
}

