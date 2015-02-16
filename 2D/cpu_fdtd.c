 // Cubic Quintic Nonlinear Schrodinger Equation
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

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
void R_lin(double *R, double *I, double dt);
void I_lin(double *R, double *I, double dt);
void nonlin(double *R, double *I, double dt);
void checker(double *R, double *I, double *max, int step);
int max_index(double arr[], int size);
void matlab_gen(double *R_0, double *I_0, double *R, double *I, double *max);

int main(void)
{
    printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));
	
    // Allocate x, y, R and I on host. Max will be use to store max |psi|
	// R_0 and I_0 will keep a copy of initial pulse for printing
	double *x = (double*)malloc(sizeof(double) * XNODES);
	double *y = (double*)malloc(sizeof(double) * YNODES);
	double *max = (double*)malloc(sizeof(double) * TNODES);
	double *R = (double*)malloc(sizeof(double) * XNODES * YNODES);
    double *I = (double*)malloc(sizeof(double) * XNODES * YNODES);   
	double *R_0 = (double*)malloc(sizeof(double) * XNODES * YNODES);
    double *I_0 = (double*)malloc(sizeof(double) * XNODES * YNODES);   
	
	// initialize x and y.
	for(int i = 0; i < XNODES ; i++)
		x[i] = (i-XNODES/2)*DELTAX;
		
    for(int i = 0; i < YNODES ; i++)
		y[i] = (i-YNODES/2)*DELTAY;

    // Initial Conditions
    for(int j = 0; j < YNODES; j++)
		for(int i = 0; i < XNODES; i++)
			{
				R[INDEX_C] = A_s*A*exp(-(x[i]*x[i]+y[j]*y[j])/(2*Rad*Rad*Rad_s*Rad_s)); 
				I[INDEX_C] = 0;
				R_0[INDEX_C] = R[INDEX_C];
				I_0[INDEX_C] = I[INDEX_C];
			}
	
	// print max |psi| for initial conditions
	checker(R, I, max, 0);
	// Begin timing
	for (int i = 1; i < TNODES; i++)
	{
		// linear
		R_lin(R, I, DELTAT*0.5);
        I_lin(R, I, DELTAT*0.5);
		// nonlinear
		nonlin(R, I, DELTAT);
		// linear
		R_lin(R, I, DELTAT*0.5);
        I_lin(R, I, DELTAT*0.5);
		// find max psi
		checker(R, I, max, i);
	}

	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	matlab_gen(R_0, I_0, R, I, max);

	// wrap up                                                  
	free(R); 
	free(I); 
	free(R_0); 
	free(I_0); 
	free(x); 
	free(y);
	free(max);
	
	return 0;
}

void R_lin(double *R, double *I, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int j = 1; j < YNODES - 1; j++)
		for(int i = 1; i < XNODES - 1; i++)
			R[INDEX_C] = R[INDEX_C] - dt/(DELTAX*DELTAX)*(I[INDEX_1] - 2*I[INDEX_C] + I[INDEX_2])
									- dt/(DELTAY*DELTAY)*(I[INDEX_3] - 2*I[INDEX_C] + I[INDEX_4]);
}

void I_lin(double *R, double *I, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int j = 1; j < YNODES - 1; j++)
		for(int i = 1; i < XNODES - 1; i++)
			I[INDEX_C] = I[INDEX_C] + dt/(DELTAX*DELTAX)*(R[INDEX_1] - 2*R[INDEX_C] + R[INDEX_2])
									+ dt/(DELTAY*DELTAY)*(R[INDEX_3] - 2*R[INDEX_C] + R[INDEX_4]);
}

void nonlin(double *R, double *I, double dt)
{                  
	double Rp, Ip, A2;
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
	for(int j = 1; j < YNODES-1; j++)
		for(int i = 1; i < XNODES-1; i++)
		{
			Rp = R[INDEX_C];  Ip = I[INDEX_C];
			A2 = Rp*Rp+Ip*Ip; // |psi|^2
			
			R[INDEX_C] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
			I[INDEX_C] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
		}
}

void checker(double *R, double *I, double *max, int step)
{
	double *Arr = (double*)malloc(sizeof(double) * XNODES * YNODES);
    
	for(int i = 0; i < XNODES * YNODES; i++)
		Arr[i] = sqrt(R[i] * R[i] + I[i] * I[i]);

    int index = max_index(Arr, XNODES*YNODES);

	max[step] = Arr[index];

	free(Arr);
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

void matlab_gen(double *R_0, double *I_0, double *R, double *I, double *max)
{
    FILE *matlab_file = fopen("cpu_2dpsi.m", "w");

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
		fprintf(matlab_file, "%0.10f ", sqrt(R_0[i] * R_0[i] + I_0[i] * I_0[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"psi_0 = vec2mat(psi_0,%d);\n\n", XNODES);

	// Generate final pulse matrix
	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES*YNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(R[i] * R[i] + I[i] * I[i]));
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

