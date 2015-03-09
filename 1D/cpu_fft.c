/**********************************************************************************
* Numerical Solution for the Cubic Nonlinear Schrodinger Equation in (1+1)D	 	  *
* using symmetric split step Fourier method		                              	  *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/timers.h"
#include "../lib/helpers.h"
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// Grid parameters
#define XN	1024				// number of Fourier Modes
#define TN	100000				// number of temporal nodes
#define L	10.0				// Spatial Period
#define TT	10.0                // Max time
#define DX	(2*L / XN)			// spatial step size
#define DT	(TT / TN)			// temporal step size

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "cpu_fft_plot.m"
#define TIME_F "cpu_fft_time.m"

// Function Prototypes
void nonlin(fftw_complex *psi, double dt, int xn);
void lin(fftw_complex *psi, double *k2, double dt, int xn);
void normalize(fftw_complex *psi, int size);

int main(int argc, char *argv[])
{
	// Timing starts here
	double t1 = get_cpu_time();
	
	// Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
    double *x = (double*)malloc(sizeof(double) * XN);
    double *k2 = (double*)malloc(sizeof(double) * XN);
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	
	// Create transform plans
	fftw_plan forward, backward;
	forward = fftw_plan_dft_1d(XN, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_1d(XN, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Create wave number
	double dkx=2*M_PI/XN/DX;
	double *kx = (double*)malloc(XN*sizeof(double));
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2-i]=(XN/2-i)*dkx;
	for(int i = XN/2+1; i < XN; i++)
		kx[i]=(i-XN)*dkx; 

	// Initial conditions
	for (int i = 0; i < XN; i++)
	{
		x[i] = (i-XN/2)*DX;
		k2[i] = kx[i]*kx[i];
		psi[i] = sqrt(2.0)/(cosh(x[i])) + 0*I;  
		//psi[i] = 4.0*exp(-(x[i]*x[i])/4.0/4.0) + 0*I;
		psi_0[i] = psi[i];  
	}
    
	// Forward transform
	fftw_execute(forward);
	
	// Print timing info to file
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin(psi, k2, DT/2, XN);  
		// Backward transform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN);
		// Solve nonlinear part
		nonlin(psi, DT, XN);
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, XN);
		// Print time at specific intervals
		if(i % IRVL == 0)
			fprintf(fp, "%f, ", get_cpu_time()-t1);
	}
	// Wrap up timing file
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time, '-*r');\n");
	fclose(fp);
	
	// Backward transform to retreive data
	fftw_execute(backward);
	// Normalize
	normalize(psi, XN);
	
	// Plot results
	cm_plot_1d(psi_0, psi, L, XN, PLOT_F);

	// Clean up
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	free(x);
	free(k2);
	free(kx);

	return 0;
}

void nonlin(fftw_complex *psi, double dt, int xn)
{                  
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
    	psi[i] = cexp(I * cabs(psi[i]) * cabs(psi[i]) * dt)*psi[i];
}

void lin(fftw_complex *psi, double *k2, double dt, int xn)
{                  
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
    	psi[i] = cexp(-I * k2[i] * dt)*psi[i];
}

void normalize(fftw_complex *psi, int size)
{
	for (int i = 0; i < size; i++)
		psi[i] = psi[i]/size;
}
