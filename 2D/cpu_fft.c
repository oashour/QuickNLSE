/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation in      *
* (2+1)D using symmetric split step Fourier method								  *		                              	  *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/timers.h"
#include "../lib/helpers.h"
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// Grid parameters                                              
#define XN	64			// Number of x-spatial nodes               ______XN_____
#define YN	64			// Number of y-spatial nodes            Y |_|_|_|_|_|_|_|H
#define TN	10000  		// number of temporal nodes             N |_|_|_|_|_|_|_|E
#define LX	50.0		// x-spatial domain [-LX,LX)            O |_|_|_|_|_|_|_|I
#define LY	50.0		// y-spatial domain [-LY,LY)            D |_|_|_|_|_|_|_|G
#define TT	10.0  		// Maximum t                            E |_|_|_|_|_|_|_|H
#define DX	(2*LX / XN)	// x-spatial step size    				S |_|_|_|_|_|_|_|T
#define DY	(2*LY / YN)	// y-spatial step size   				       WIDTH
#define DT	(TT / TN)   // temporal step size                     

// Gaussian Parameters                                                 
#define  A_S 	(3.0/sqrt(8.0))     	// A*
#define  R_S 	(sqrt(32.0/9.0))    	// R*
#define  A 		0.6                 	// A
#define  R 		(1.0/(A*sqrt(1.0-A*A))) // R

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "cpu_fft_plot.m"
#define TIME_F "cpu_fft_time.m"

// Index flattening macro [x,y] = [x * width + y] 
#define ind(i,j)  ((i)*XN+(j))		//[i  ,j  ] 

// Function prototypes
void nonlin(fftw_complex *psi, double dt, int xn, int yn);
void lin(fftw_complex *psi, double *k2, double dt, int xn, int yn);
void normalize(fftw_complex *psi, int size);

int main(int argc, char *argv[])
{                                                                          
	// Timing starts here
	double t1 = get_cpu_time();
	
	// Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
    double *x           = (double*)malloc(sizeof(double) * XN);
	double *y           = (double*)malloc(sizeof(double) * YN);
	double *k2          = (double*)malloc(sizeof(double) * XN * YN);
	double *max         = (double*)malloc(sizeof(double) * TN);
	fftw_complex *psi   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN);
	fftw_complex *psi_0 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN);
	
	// Create transform plans
	fftw_plan forward, backward;
	forward = fftw_plan_dft_2d(XN, YN, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_2d(XN, YN, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Create wave numbers
	double dkx = 2*M_PI/XN/DX;
	double *kx = (double*)malloc(XN * sizeof(double));
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		kx[i]=(i - XN) * dkx; 

	double dky = 2*M_PI/YN/DY;
	double *ky = (double*)malloc(YN * sizeof(double));
	for(int i = YN/2; i >= 0; i--) 
		ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		ky[i]=(i - YN) * dky; 

	// Initialize x and y
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

    // Initial conditions 
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				psi[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])
													/(2*R*R*R_S*R_S)) + 0*I; 
				psi_0[ind(i,j)] = psi[ind(i,j)];
				k2[ind(i,j)] = kx[i]*kx[i] + ky[j]*ky[j];
			}   
	
	// Print timing info to file
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	cmax_psi(psi, max, 0, XN*YN);
	
	// Forward transform
	fftw_execute(forward);
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin(psi, k2, DT*0.5, XN, YN);  
		// Backward transform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN*YN);
		// Solve nonlinear 
		nonlin(psi, DT, XN, YN);
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, XN, YN);
		// Save max |psi| for printing
		cmax_psi(psi, max, i, XN*YN);
		// Print time at specific intervals
		if(i % IRVL == 0)
			fprintf(fp, "%f, ", get_cpu_time()-t1);
	}
	// Wrap up timing file
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time, '-*r');\n");
	fclose(fp);
	
	// Backward tranform to retreive data
	fftw_execute(backward);
	// Normalize
	normalize(psi, XN*YN);
	
	// Plot results
	cm_plot_2d(psi_0, psi, max, LX, LY, XN, YN, TN, PLOT_F);

	// Clean up 
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	free(x);
	free(y);
	free(k2);
	free(kx);
	free(ky);
	free(max);

	return 0;
}

void lin(fftw_complex *psi, double *k2, double dt, int xn, int yn)
{                  
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
		for(int j = 0; j < yn; j++)
    		psi[ind(i,j)] = cexp(-I * k2[ind(i,j)] * dt)*psi[ind(i,j)];
}

void nonlin(fftw_complex *psi, double dt, int xn, int yn)
{                  
	double psi2;
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
		for(int j = 0; j < yn; j++)
		{
    		psi2 = cabs(psi[ind(i,j)])*cabs(psi[ind(i,j)]);
			psi[ind(i,j)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j)];
		}
}

void normalize(fftw_complex *psi, int size)
{
		for(int i = 0; i < size; i++)
			psi[i] = psi[i]/size;
}
