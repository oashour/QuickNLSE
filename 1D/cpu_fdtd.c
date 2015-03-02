/**********************************************************************************
* Numerical Solution for the Cubic Nonlinear Schrodinger Equation in (1+1)D	  	  *
* using explicit FDTD with second order splitting.                                *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/helpers.h"
#include "../lib/timers.h"

// Grid Parameters
#define XN	1024				// number of spatial ndes
#define TN	100000				// number of temporal nodes
#define L	10.0				// Spatial Period
#define TT	10.0                // Max time
#define DX	(2*L / XN)			// spatial step size
#define DT	(TT / TN)			// temporal step size

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Function Prototypes
void Re_lin(double *Re, double *Im, double dt, int xn, double dx);
void Im_lin(double *Re, double *Im, double dt, int xn, double dx);
void nonlin(double *Re, double *Im, double dt, int xn);

int main(int argc, char *argv[])
{
	// Timing starts here
	double t1 = get_cpu_time();
	
    // Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
    double *x = (double*)malloc(sizeof(double) * XN);
	double *Re = (double*)malloc(sizeof(double) * XN);
    double *Im = (double*)malloc(sizeof(double) * XN);
	double *Re_0 = (double*)malloc(sizeof(double) * XN);
    double *Im_0 = (double*)malloc(sizeof(double) * XN);
	
	// Initial conditions
	for (int i = 0; i < XN; i++)
	{
		x[i] = (i-XN/2)*DX;
		Re[i] = sqrt(2.0)/(cosh(x[i]));  
		Im[i] = 0;
		Re_0[i] = Re[i];
		Im_0[i] = Im[i];
	}
	
	// Print timing info to file
	FILE *fp = fopen("test_1.m", "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Start time evolution
	for (int i = 1; i < TN; i++)
	{
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, DX);
        Im_lin(Re, Im, DT*0.5, XN, DX);
		// Solve nonlinear part
		nonlin(Re, Im, DT, XN);
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, DX);
        Im_lin(Re, Im, DT*0.5, XN, DX);
		// Print time at specific intervals
		if(i % IRVL == 0)
			fprintf(fp, "%f, ", get_cpu_time()-t1);
	}
	// Wrap up timing file
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time, '-*r');\n");
	fclose(fp);
	
	// Plot results
	m_plot_1d(Re_0, Im_0, Re, Im, L, XN, "cpu_fdtd.m");

	// Clean up
	free(Re); 
	free(Im); 
	free(Re_0); 
	free(Im_0); 
	free(x); 

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn, double dx)
{                  
	// Avoid first and last point (boundary conditions)
	for(int i = 1; i < xn-1; i++)
		Re[i] = Re[i] - dt/(dx*dx)*(Im[i+1] - 2*Im[i] + Im[i-1]);
}

void Im_lin(double *Re, double *Im, double dt, int xn, double dx)
{                  
	// Avoid first and last point (boundary conditions)
	for(int i = 1; i < xn-1; i++)
		Im[i] = Im[i] + dt/(dx*dx)*(Re[i+1] - 2*Re[i] + Re[i-1]);
}

void nonlin(double *Re, double *Im, double dt, int xn)
{                  
	double Rp, Ip, A2;
	// Avoid first and last point (boundary conditions)
	for(int i = 1; i < xn-1; i++)
	{
		Rp = Re[i]; 
		Ip = Im[i];
		A2 = Rp*Rp+Ip*Ip;
	
		Re[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
   		Im[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
	}
}

