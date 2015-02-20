// nlse
#include<stdio.h>
#include<math.h>

// given stuff
#define XNODES	1000
#define TNODES	1000000
#define L		10.0
#define TMAX	1.0

// calculated from given
#define DELTAX	(L / (XNODES - 1.0))
#define DELTAT	(TMAX / (TNODES - 1.0))

// in case of gaussian
#define A 1.0
#define Rad 2.0

__global__ void R_lin_kernel(double *R, double *I, double dt);
__global__ void I_lin_kernel(double *R, double *I, double dt);
__global__ void nonlin_kernel(double *R, double *I, double dt);
void matlab_plot(double *h_R, double *h_I, double *d_R, double *d_I);

int main(void)
{
    printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));

    cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
 
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	
	// create the arrays x and t
    double *h_x = (double*)malloc(sizeof(double) * XNODES);
    // create used arrays
	double *h_R = (double*)malloc(sizeof(double) * XNODES);
    double *h_I	= (double*)malloc(sizeof(double) * XNODES);   

	// initial conditions.
	for(int i = 0; i < XNODES ; i++)
	{
		h_x[i] = (i-XNODES/2)*DELTAX;
//		h_R[i]	= sqrt(2.0)/(cosh(h_x[i]));	// initial
		h_I[i]	= 0;    					// initial
		h_R[i]	= 2*exp(-(h_x[i]*h_x[i])/2.0/2.0);	// initial
	}
    
    // allocate arrays on device and copy them
	double *d_R, *d_I;
	cudaMalloc(&d_R, sizeof(double) * XNODES);
	cudaMalloc(&d_I, sizeof(double) * XNODES);
	cudaMemcpy(d_R, h_R, sizeof(double) * XNODES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, sizeof(double) * XNODES, cudaMemcpyHostToDevice);

	// initialize the grid
	dim3 threadsPerBlock(128,1,1);
	dim3 blocksPerGrid((XNODES + 127)/128,1,1);

	// solve 
	cudaEventRecord(beginEvent, 0);
	for (int i = 1; i < TNODES; i++)
	{
		// linear
		R_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_R, d_I, DELTAT*0.5);
        I_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_R, d_I, DELTAT*0.5);
		// nonlinear
		nonlin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_R, d_I, DELTAT);
		// linear
		R_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_R, d_I, DELTAT*0.5);
        I_lin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_R, d_I, DELTAT*0.5);
	}
	cudaEventRecord(endEvent, 0);

    cudaEventSynchronize(endEvent);
 
	float timeValue;
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
 
	FILE *time_file;
	time_file = fopen("cuda_time.txt", "a"); 
	fprintf(time_file, "%f, ", timeValue/1000.0);
	fclose(time_file);

	matlab_plot(h_R, h_I, d_R, d_I);

	// wrap up
	free(h_R); 
	free(h_I); 
	free(h_x); 
	cudaFree(d_R); 
	cudaFree(d_I); 

	return 0;
}

__global__ void R_lin_kernel(double *R, double *I, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	if (i >= XNODES - 1 || i == 0) return; // avoid first and last elements

	R[i] = R[i] - dt/(DELTAX*DELTAX)*(I[i+1] - 2*I[i] + I[i-1]);
}

__global__ void I_lin_kernel(double *R, double *I, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	if (i >= XNODES - 1 || i == 0) return; // avoid first and last elements

	I[i] = I[i] + dt/(DELTAX*DELTAX)*(R[i+1] - 2*R[i] + R[i-1]);
}

__global__ void nonlin_kernel(double *R, double *I, double dt)
{                  
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double Rp = R[i]; double Ip = I[i];
	double A2 = Rp*Rp+Ip*Ip;
	
	if (i > XNODES - 1) return; 
	
	R[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
	I[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
}

void matlab_plot(double *h_R, double *h_I, double *d_R, double *d_I)
{
	FILE *matlab_file;
	matlab_file = fopen("plot_CUDA.m", "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n", -L, L, XNODES);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(h_R[i] * h_R[i] + h_I[i] * h_I[i]));
	fprintf(matlab_file,"];\n");                                                                 

	cudaMemcpy(h_R, d_R, sizeof(double) * XNODES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, sizeof(double) * XNODES, cudaMemcpyDeviceToHost);

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(h_R[i] * h_R[i] + h_I[i] * h_I[i]));
	fprintf(matlab_file,"];\n");                                                                 
	
	fprintf(matlab_file, "plot(x, psi_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, psi_f, '--b', 'LineWidth', 1);\n"
						 "legend('t = 0', 't = %f', 0);\n"
						 "title('Soliton Solution for GPU');\n"
						 "xlabel('x values'); ylabel('|psi|');", TMAX);
	fclose(matlab_file);
}

