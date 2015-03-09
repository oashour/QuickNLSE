/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation in      *
* (3+1)D using explicit FDTD with second order splitting.                         *   
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
**********************************************************************************/
#include "../lib/helpers.h"
#include "../lib/timers.h"

// Grid parameters                                              
#define XN	64			// Number of x-spatial nodes               _____XN _____
#define YN	64			// Number of y-spatial nodes            Y |_|_|_|_|_|_|_|H
#define TN	10000		// number of temporal nodes             N |_|_|_|_|_|_|_|E
#define LX	50.0		// x-spatial domain [-LX,LX)            O |_|_|_|_|_|_|_|I
#define LY	50.0		// y-spatial domain [-LY,LY)            D |_|_|_|_|_|_|_|G
#define TT	10.0  		// Maximum t                            E |_|_|_|_|_|_|_|H
#define DX	(2*LX / XN)	// Spacing between x nodes				S |_|_|_|_|_|_|_|T
#define DY	(2*LY / YN)	// Spacing between y nodes				       WIDTH
#define DT	(TT / TN)   // Spacing between temporal nodes                        

// Gaussian parameters                                                 
#define  A_S 	(3.0/sqrt(8.0))     	// A*
#define  R_S 	(sqrt(32.0/9.0))    	// R*
#define  A 		0.6                 	// A
#define  R 		(1.0/(A*sqrt(1.0-A*A))) // R

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "cpu_fdtd_plot.m"
#define TIME_F "cpu_fdtd_time.m"

// Index flattening macro [x,y] = [x * width + y] 
#define ind(i,j)  ((i)*XN+(j))		//[i  ,j  ] 

// Function prototypes 
void Re_lin(double *Re, double *Im, double dt, int xn, int yn, double dx, double dy);
void Im_lin(double *Re, double *Im, double dt, int xn, int yn, double dx, double dy);
void nonlin(double *Re, double *Im, double dt, int xn, int yn);

int main(void)
{
	// Timing starts here
	double t1 = get_cpu_time();
    
	// Print basic info about simulation
	printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
	
	// Allocate the arrays
	double *x    = (double*)malloc(sizeof(double) * XN);
	double *y    = (double*)malloc(sizeof(double) * YN);
	double *max  = (double*)malloc(sizeof(double) * (TN + 1));
	double *Re   = (double*)malloc(sizeof(double) * XN * YN);
    double *Im   = (double*)malloc(sizeof(double) * XN * YN);   
	double *Re_0 = (double*)malloc(sizeof(double) * XN * YN);
    double *Im_0 = (double*)malloc(sizeof(double) * XN * YN);   
	
	// Initialize x and y
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

    // Initial conditions
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				Re[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])/(2*R*R*R_S*R_S)); 
				Im[ind(i,j)] = 0;
				Re_0[ind(i,j)] = Re[ind(i,j)];
				Im_0[ind(i,j)] = Im[ind(i,j)];
			}
	
	// Print timing info to file
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	max_psi(Re, Im, max, 0, XN*YN);
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, YN, DX, DY);
        Im_lin(Re, Im, DT*0.5, XN, YN, DX, DY);
		// Solve nonlinear part
		nonlin(Re, Im, DT, XN, YN);
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, YN, DX, DY);
        Im_lin(Re, Im, DT*0.5, XN, YN, DX, DY);
		// Save max |psi| for later printing
		max_psi(Re, Im, max, i, XN*YN);
		// Print time at specific intervals
		if(i % IRVL == 0)
			fprintf(fp, "%f, ", get_cpu_time()-t1);
	}
	// Wrap up timing file
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time, '-*r');\n");
	fclose(fp);
	
	// Plot results
	m_plot_2d(Re_0, Im_0, Re, Im, max, LX, LY, XN, YN, TN, PLOT_F);
	
	// Clean up                                                  
	free(Re); 
	free(Im); 
	free(Re_0); 
	free(Im_0); 
	free(x); 
	free(y);
	free(max);
	
	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn, int yn, double dx, double dy)
{                  
	// Avoid first and last point (boundary conditions)
    for(int j = 1; j < yn - 1; j++)
		for(int i = 1; i < xn - 1; i++)
			Re[ind(i,j)] = Re[ind(i,j)] 
						   - dt/(dx*dx)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
						   - dt/(dy*dy)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

void Im_lin(double *Re, double *Im, double dt, int xn, int yn, double dx, double dy)
{                  
	// Avoid first and last point (boundary conditions)
    for(int j = 1; j < yn - 1; j++)
		for(int i = 1; i < xn - 1; i++)
			Im[ind(i,j)] = Im[ind(i,j)] 
						   + dt/(dx*dx)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
						   + dt/(dy*dy)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

void nonlin(double *Re, double *Im, double dt, int xn, int yn)
{                  
	double Rp, Ip, A2;
	// Avoid first and last point (boundary conditions)
	for(int j = 1; j < yn-1; j++)
		for(int i = 1; i < xn-1; i++)
		{
			Rp = Re[ind(i,j)];  Ip = Im[ind(i,j)];
			A2 = Rp*Rp+Ip*Ip; 
			
			Re[ind(i,j)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
			Im[ind(i,j)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
		}
}

