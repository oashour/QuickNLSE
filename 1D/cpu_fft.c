// nlse (1+1)D
#include <sys/time.h>
#include "../lib/helpers.h"
#include <fftw3.h>
#define M_PI 3.14159265358979323846264338327

// given stuff
#define XN	1024				// number of Fourier Modes
#define TN	10000				// number of temporal nodes
#define L	10.0				// Spatial Period
#define TT	10.0                // Max time
#define DX	(2*L / XN) 		// spatial step size
#define DT	(TT / TN)     // temporal step size

void nonlin(fftw_complex *psi, double dt);
void lin(fftw_complex *psi, double *k2, double dt);
void normalize(fftw_complex *psi, int size);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));

    double t1,t2,elapsed;
	struct timeval tp;
	int rtn;

	// allocate and initialize the arrays
    double *x = (double*)malloc(sizeof(double) * XN);
    double *k2 = (double*)malloc(sizeof(double) * XN);
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XN);
	
	fftw_plan forward, backward;
	forward = fftw_plan_dft_1d(XN, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_1d(XN, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // generate wave number (move into function)
	double dkx=2*M_PI/XN/DX;
	double *kx;
	kx=(double*)malloc(XN*sizeof(double));
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2-i]=(XN/2-i)*dkx;
	for(int i = XN/2+1; i < XN; i++)
		kx[i]=(i-XN)*dkx; 

	for (int i = 0; i < XN; i++)
	{
		x[i] = (i-XN/2)*DX;
		k2[i] = kx[i]*kx[i];
		//psi[i] = sqrt(2.0)/(cosh(x[i])) + 0*I;  
		psi[i] = 4.0*exp(-(x[i]*x[i])/4.0/4.0) + 0*I;
		psi_0[i] = psi[i];  
	}
    
	rtn=gettimeofday(&tp, NULL);
	t1=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	for (int i = 1; i < TN; i++)
	{
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2);  
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN);
		// nonlinear
		nonlin(psi, DT);
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DT/2);
		// backward tranform
		fftw_execute(backward);
		// scale down
		normalize(psi, XN);
	}
	rtn=gettimeofday(&tp, NULL);
	t2=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	elapsed=t2-t1;
	
	printf("time elapsed: %f s.\n", elapsed);

	cm_plot_1d(psi_0, psi, L, XN, "cpufft.m");

	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	free(x);
	free(k2);

	return 0;
}

void nonlin(fftw_complex *psi, double dt)
{                  
	for(int i = 0; i < XN; i++)
    	psi[i] = cexp(I * cabs(psi[i]) * cabs(psi[i]) * dt)*psi[i];
}

void lin(fftw_complex *psi, double *k2, double dt)
{                  
	for(int i = 0; i < XN; i++)
    	psi[i] = cexp(-I * k2[i] * dt)*psi[i];
}

void normalize(fftw_complex *psi, int size)
{
	for (int i = 0; i < XN; i++)
		psi[i] = psi[i]/size;
}
