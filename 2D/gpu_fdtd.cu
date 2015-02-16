// Cubic Quintic Nonlinear Schrodinger Equation
#include<stdio.h>
#include<math.h>

// given parameters for FDTD                                              XNODES
#define XNODES	256			// number of X nodes                       _____________
#define YNODES	256			// number of Y nodes                    Y |_|_|_|_|_|_|_|H
#define TNODES	10000		// number of temporal nodes             N |_|_|_|_|_|_|_|E
#define LX		50.0		// maximum X                            O |_|_|_|_|_|_|_|I
#define LY		50.0		// maximum Y                            D |_|_|_|_|_|_|_|G
#define TMAX	100.0  		// maximum t                            E |_|_|_|_|_|_|_|H
//                                                                  S |_|_|_|_|_|_|_|T
// Gaussian Parameters                                                     WIDTH
#define  A_s 	(3.0/sqrt(8.0))
#define  Rad_s 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  Rad 	(1.0/(A*sqrt(1.0-A*A)))

// calculated from given
#define DELTAX	(LX / (XNODES - 1.0))
#define DELTAY	(LY / (YNODES - 1.0))
#define DELTAT	(TMAX / (TNODES - 1.0))


// Index linearization for kernels [x,y] = [x * width + y] 
#define INDEX_C  i    * XNODES +  j		//[i  ,j  ] 
#define INDEX_1 (i+1) * XNODES +  j		//[i+1,j  ]
#define INDEX_2 (i-1) * XNODES +  j		//[i-1,j  ]
#define INDEX_3  i    * XNODES + (j+1)	//[i  ,j+1]
#define INDEX_4  i    * XNODES + (j-1)	//[i  ,j-1]

// Function prototypes 
__global__ void R_lin_kernel(double *R, double *I, double dt);
__global__ void I_lin_kernel(double *R, double *I, double dt);
__global__ void nonlin_kernel(double *R, double *I, double dt);
void checker(double *d_R, double *d_I, double *max, int step);
int max_index(double arr[], int size);
void matlab_gen(double *h_R, double *h_I, double *d_R, double *d_I, double *max);

int main(void)
{
    printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));
    
	// Timing set up
	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;
 
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	
    // allocate x, y, R and I on host. Max will be use to store max |psi|
	double *h_x = (double*)malloc(sizeof(double) * XNODES);
	double *h_y = (double*)malloc(sizeof(double) * YNODES);
	double *h_max = (double*)malloc(sizeof(double) * TNODES);
	double *h_R = (double*)malloc(sizeof(double) * XNODES * YNODES);
    double *h_I	= (double*)malloc(sizeof(double) * XNODES * YNODES);   

	// initialize x and y.
	for(int i = 0; i < XNODES ; i++)
		h_x[i] = (i-XNODES/2)*DELTAX;
		
    for(int i = 0; i < YNODES ; i++)
		h_y[i] = (i-YNODES/2)*DELTAY;

    // Initial Conditions
    for(int j = 0; j < YNODES; j++)
		for(int i = 0; i < XNODES; i++)
			{
				h_R[i*XNODES+j] = A_s*A*exp(-(h_x[i]*h_x[i]+h_y[j]*h_y[j])/(2*Rad*Rad*Rad_s*Rad_s)); 
				h_I[i*XNODES+j] = 0;    				   
			}
	
	// Allocate device arrays and copy from host.
	double *d_R, *d_I;
	cudaMalloc(&d_R, sizeof(double) * XNODES*YNODES);
	cudaMalloc(&d_I, sizeof(double) * XNODES*YNODES);
	cudaMemcpy(d_R, h_R, sizeof(double) * XNODES*YNODES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, sizeof(double) * XNODES*YNODES, cudaMemcpyHostToDevice);

	// Initialize the grid
	dim3 blocksPerGrid((XNODES+15)/16, (YNODES+15)/16, 1);
	dim3 threadsPerBlock(16, 16, 1);

	// print max |psi| for initial conditions
	checker(d_R, d_I, h_max, 0);
	// Begin timing
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
		// find max psi
		checker(d_R, d_I, h_max, i);
	}
	// End timing
	cudaEventRecord(endEvent, 0);

    cudaEventSynchronize(endEvent);
 
	float timeValue;
	cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
 
	printf("Time elapsed: %f.\n", timeValue);

	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	matlab_gen(h_R, h_I, d_R, d_I, h_max);

	// wrap up                                                  
	free(h_R); 
	free(h_I); 
	free(h_x); 
	free(h_y);
	free(h_max);
	cudaFree(d_R); 
	cudaFree(d_I); 
	
	return 0;
}

__global__ void R_lin_kernel(double *R, double *I, double dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
    // We're avoiding Boundary Elements (kept at initial value approx = 0)
    if(i == 0 || j == 0 || i >= XNODES-1 || j >= YNODES-1) return;
	
	R[INDEX_C] = R[INDEX_C] - dt/(DELTAX*DELTAX)*(I[INDEX_1] - 2*I[INDEX_C] + I[INDEX_2])
							- dt/(DELTAY*DELTAY)*(I[INDEX_3] - 2*I[INDEX_C] + I[INDEX_4]);
}

__global__ void I_lin_kernel(double *R, double *I, double dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
    // We're avoiding Boundary Elements (kept at initial value approx = 0);
	if(i == 0 || j == 0 || i >= XNODES-1 || j >= YNODES-1) return;
	
	I[INDEX_C] = I[INDEX_C] + dt/(DELTAX*DELTAX)*(R[INDEX_1] - 2*R[INDEX_C] + R[INDEX_2])
							+ dt/(DELTAY*DELTAY)*(R[INDEX_3] - 2*R[INDEX_C] + R[INDEX_4]);
}

__global__ void nonlin_kernel(double *R, double *I, double dt)
{                  
	// setting up i j
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
    
	// linearizing the index  [x,y] = [x + y * width] 
	int index =  i * XNODES +  j; //[i  ,j] 
    
	// we're avoiding Boundary Elements (kept at initial value approx = 0);
    if(i == 0 || j == 0 || i >= XNODES-1 || j >= YNODES-1) return;

	double Rp = R[index]; double Ip = I[index];
	double A2 = Rp*Rp+Ip*Ip; // |psi|^2
	
	R[index] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
	I[index] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
}

void checker(double *d_R, double *d_I, double *max, int step)
{
	double *h_R	= (double*)malloc(sizeof(double) * XNODES * YNODES);
    double *h_I	= (double*)malloc(sizeof(double) * XNODES * YNODES);   
	double *h_A	= (double*)malloc(sizeof(double) * XNODES * YNODES);
    
	cudaMemcpy(h_R, d_R, sizeof(double) * XNODES * YNODES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, sizeof(double) * XNODES * YNODES, cudaMemcpyDeviceToHost);

	for(int i = 0; i < XNODES * YNODES; i++)
		h_A[i] = sqrt(h_R[i] * h_R[i] + h_I[i] * h_I[i]);

    int index = max_index(h_A, XNODES*YNODES);

	max[step] = h_A[index];

    free(h_R);
	free(h_I);
	free(h_A);
}

int max_index(double arr[], int size)
{
	int largestIndex = 0;

	for (int index = largestIndex; index < size; index++) 
	{
		if (arr[largestIndex] <= arr[index])
            largestIndex = index;
    }

    return largestIndex;
}

void matlab_gen(double *h_R, double *h_I, double *d_R, double *d_I, double *max)
{
    FILE *matlab_file = fopen("GPU_2dpsi.m", "w");

	// Initialize Arrays
	fprintf(matlab_file, "[x,y] = meshgrid(linspace(%f,%f,%d), linspace(%f, %f, %d));\n", -LX, LX, XNODES, -LY, LY, YNODES);
	fprintf(matlab_file, "steps = [0:%d-1];\n\n", TNODES);

	// Generate the array for max |psi|
    fprintf(matlab_file, "max = [");
	for(int i = 0; i < TNODES; i++)
		fprintf(matlab_file, "%0.10f ", max[i]);
	fprintf(matlab_file, "];\n\n");
	
	// generate initial pulse matrix
	fprintf(matlab_file, "psi_0 = [");
	for(int i = 0; i < XNODES*YNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(h_R[i] * h_R[i] + h_I[i] * h_I[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"psi_0 = vec2mat(psi_0,%d);\n\n", XNODES);

	// Copy final pulse to CPU
	cudaMemcpy(h_R, d_R, sizeof(double) * XNODES*YNODES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, sizeof(double) * XNODES*YNODES, cudaMemcpyDeviceToHost);

	// Generate final pulse matrix
	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES*YNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(h_R[i] * h_R[i] + h_I[i] * h_I[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"psi_f = vec2mat(psi_f,%d);\n\n", XNODES);
	
	// plot max |psi| versus time step
	fprintf(matlab_file, "plot(steps, max, '-r', 'LineWidth', 1); grid on;\n"
						 "title('Maximum Value of |psi| per time step, from t = 0 to t = %f');\n"
						 "xlabel('Time Step'); ylabel('max |psi|');\n\n", TMAX);

	// generate initial pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,psi_0);\n"
						 "title('Initial Pulse = 0');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|psi|');\n\n");

	// generate final pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,psi_f);\n"
						 "title('Final Pulse at t = %f');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|psi|');\n\n", TMAX);

	fclose(matlab_file);
	
}
