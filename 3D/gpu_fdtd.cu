 // Cubic Quintic Nonlinear Schrodinger Equation
#include "../lib/cu_helpers.h"

// Grid Parameters
#define XN	64						// Number of x-spatial nodes        
#define YN	64						// Number of y-spatial nodes          
#define ZN  64						// Number of z-spatial nodes         
#define TN	1000					// Number of temporal nodes          
#define LX	100.0					// x-spatial domain [-LX,LX)         
#define LY	100.0					// y-spatial domain [-LY,LY)         
#define LZ	100.0					// z-spatial domain [-LZ,LZ)         
#define TT	100.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
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
__global__ void Re_lin_kernel(double *Re, double *Im, double dt);
__global__ void Im_lin_kernel(double *Re, double *Im, double dt);
__global__ void nonlin_kernel(double *Re, double *Im, double dt);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));
	
    // Allocate x, y, Re and Im on host. Max will be use to store max |psi|
	// Re_0 and Im_0 will keep a copy of initial pulse for printing
	double *h_x = (double*)malloc(sizeof(double) * XN);
	double *h_y = (double*)malloc(sizeof(double) * YN);
	double *h_z = (double*)malloc(sizeof(double) * YN);
	double *h_max = (double*)malloc(sizeof(double) * TN);
	double *h_Re = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *h_Im = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	double *h_Re_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *h_Im_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	
	// initialize x and y.
	for(int i = 0; i < XN ; i++)
		h_x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		h_y[i] = (i-YN/2)*DY;

    for(int i = 0; i < YN ; i++)
		h_z[i] = (i-ZN/2)*DZ; 
    
	// Initial Conditions
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
	
	// Allocate device arrays and copy from host.
	double *d_Re, *d_Im;
	CUDAR_SAFE_CALL(cudaMalloc(&d_Re, sizeof(double) * XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMalloc(&d_Im, sizeof(double) * XN*YN*ZN));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Re, h_Re, sizeof(double) * XN*YN*ZN,
														cudaMemcpyHostToDevice));
	CUDAR_SAFE_CALL(cudaMemcpy(d_Im, h_Im, sizeof(double) * XN*YN*ZN,
														cudaMemcpyHostToDevice));

	// Initialize the grid
	dim3 blocksPerGrid((XN+7)/8, (YN+7)/8, (ZN+7)/8);
	dim3 threadsPerBlock(8, 8, 8);
	// Begin timing
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
	}

	// Allocate device arrays and copy from host.
	CUDAR_SAFE_CALL(cudaMemcpy(h_Re, d_Re, sizeof(double)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));
	CUDAR_SAFE_CALL(cudaMemcpy(h_Im, d_Im, sizeof(double)*XN*YN*ZN, 
															cudaMemcpyDeviceToHost));

	double *h_psi2 = (double*)malloc(sizeof(double)*XN*YN*ZN);
    double *h_psi2_0 = (double*)malloc(sizeof(double)*XN*YN*ZN);
	
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				h_psi2[ind(i,j,k)] = sqrt(h_Re[ind(i,j,k)]*h_Re[ind(i,j,k)] +
									   h_Im[ind(i,j,k)]*h_Im[ind(i,j,k)]);
				h_psi2_0[ind(i,j,k)] = sqrt(h_Re_0[ind(i,j,k)]*h_Re_0[ind(i,j,k)] +
									   h_Im_0[ind(i,j,k)]*h_Im_0[ind(i,j,k)]);
            }
	
	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	vtk_3d(h_x, h_y, h_z, h_psi2, XN, YN, ZN, "test_fdtd1.vtk");
	vtk_3d(h_x, h_y, h_z, h_psi2_0, XN, YN, ZN, "test_fdtd0.vtk");
	
	// wrap up                                                  
	free(h_Re); 
	free(h_Im); 
	free(h_Re_0); 
	free(h_Im_0); 
	free(h_x); 
	free(h_y);
	free(h_z);
	free(h_max);
	free(h_psi2);
	free(h_psi2_0);
	cudaFree(d_Re); 
	cudaFree(d_Im); 
	
	return 0;
}

__global__ void Re_lin_kernel(double *Re, double *Im, double dt)
{                  
   	// setting up i j
   	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
   	// We're avoiding Boundary Elements (kept at initial value approx = 0)
   	if(i == 0 || j == 0 || k == 0 || i >= XN-1 || j >= YN-1 || k >= ZN-1) return;
    
	Re[ind(i,j,k)] = Re[ind(i,j,k)] 
			- dt/(DX*DX)*(Im[ind(i+1,j,k)] - 2*Im[ind(i,j,k)] + Im[ind(i-1,j,k)])
			- dt/(DY*DY)*(Im[ind(i,j+1,k)] - 2*Im[ind(i,j,k)] + Im[ind(i,j-1,k)])
			- dt/(DZ*DZ)*(Im[ind(i,j,k+1)] - 2*Im[ind(i,j,k)] + Im[ind(i,j,k-1)]);
}

__global__ void Im_lin_kernel(double *Re, double *Im, double dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;

   	// We're avoiding Boundary Elements (kept at initial value approx = 0)
   	if(i == 0 || j == 0 || k == 0 || i >= XN-1 || j >= YN-1 || k >= ZN-1) return;
   	
   	Im[ind(i,j,k)] = Im[ind(i,j,k)] 
   			+ dt/(DX*DX)*(Re[ind(i+1,j,k)] - 2*Re[ind(i,j,k)] + Re[ind(i-1,j,k)])
  			+ dt/(DY*DY)*(Re[ind(i,j+1,k)] - 2*Re[ind(i,j,k)] + Re[ind(i,j-1,k)])
   			+ dt/(DZ*DZ)*(Re[ind(i,j,k+1)] - 2*Re[ind(i,j,k)] + Re[ind(i,j,k-1)]);
}

__global__ void nonlin_kernel(double *Re, double *Im, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
   	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    int k = threadIdx.z + blockIdx.z * blockDim.z;

   	// We're avoiding Boundary Elements (kept at initial value approx = 0)
   	if(i == 0 || j == 0 || k == 0 || i >= XN-1 || j >= YN-1 || k >= ZN-1) return;
   	
	double Rp, Ip, A2;
	
	Rp = Re[ind(i,j,k)];  Ip = Im[ind(i,j,k)];
	A2 = Rp*Rp+Ip*Ip; // |psi|^2
	
	Re[ind(i,j,k)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	Im[ind(i,j,k)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}

