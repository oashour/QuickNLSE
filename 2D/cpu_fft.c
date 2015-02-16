// nlse (1+1)D
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stddef.h>
#include <complex.h>
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// given stuff
#define XNODES	1000				// number of Fourier Modes
#define YNODES	1000				// number of y- fourier modes
#define TNODES	100000				// number of temporal nodes
#define L		50.0				// Spatial Period
#define Tmax	10.0                // Max time
#define DELTAX	(2*L / XNODES) 		// spatial step size
#define DELTAT	(Tmax / TNODES)     // temporal step size

// Gaussian Parameters                                     
#define  A_s 	(3.0/sqrt(8.0))
#define  Rad_s 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  Rad 	(1.0/(A*sqrt(1.0-A*A)))   

// Index linearization for kernels [x,y] = [x * width + y] 
#define INDEX_C  i    * XNODES +  j		//[i  ,j  ] 
#define INDEX_1 (i+1) * XNODES +  j		//[i+1,j  ]
#define INDEX_2 (i-1) * XNODES +  j		//[i-1,j  ]
#define INDEX_3  i    * XNODES + (j+1)	//[i  ,j+1]
#define INDEX_4  i    * XNODES + (j-1)	//[i  ,j-1]   

// Function prototypes
void nonlin(fftw_complex *psi, double dt);
void lin(fftw_complex *psi, double *k2, double dt);
void checker(double *R, double *I, double *max, int step);
int max_index(double arr[], int size);
void matlab_gen(double *R_0, double *I_0, double *R, double *I, double *max); 

int main(void)
{
    printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));

    double t1,t2,elapsed;
	struct timeval tp;
	int rtn;

	// allocate and initialize the arrays
    double *x = (double*)malloc(sizeof(double) * XNODES);
	double *y = (double*)malloc(sizeof(double) * YNODES);
	double *k2x = (double*)malloc(sizeof(double) * XNODES);
	double *k2y = (double*)malloc(sizeof(double) * YNODES);
	double *max = (double*)malloc(sizeof(double) * TNODES);
	fftw_complex *psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XNODES * YNODES);
	fftw_complex *psi_0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * XNODES * YNODES);
	
	fftw_plan forward, backward;
	forward = fftw_plan_dft_2d(XNODES, YNODES, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_2d(XNODES, YNODES, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    double dkx=2*M_PI/XNODES/DELTAX;
	double *KX;
	KX=(double*)malloc(XNODES*sizeof(double));
	for(int i=XNODES/2;i>=0;i--) KX[XNODES/2-i]=(XNODES/2-i)*dkx;
	for(int i=XNODES/2+1;i<XNODES;i++) KX[i]=(i-XNODES)*dkx; 

    double dky=2*M_PI/YNODES/DELTAX;
	double *KY;
	KX=(double*)malloc(XNODES*sizeof(double));
	for(int i=YNODES/2;i>=0;i--) KX[YNODES/2-i]=(YNODES/2-i)*dky;
	for(int i=YNODES/2+1;i<YNODES;i++) KX[i]=(i-YNODES)*dky; 

	for (int i = 0; i < XNODES; i++)
	{
		x[i] = (i-XNODES/2)*DELTAX;
		k2[i] = KX[i]*KX[i];
		//psi[i] = sqrt(2.0)/(cosh(x[i])) + 0*I;  
		psi[i] = 4.0*exp(-(x[i]*x[i])/4.0/4.0) + 0*I;
		psi_0[i] = psi[i];  
	}
    
	rtn=gettimeofday(&tp, NULL);
	t1=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	for (int i = 1; i < TNODES; i++)
	{
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DELTAT/2);  
		// backward tranform
		fftw_execute(backward);
		// scale down
		for(int i = 0; i < XNODES; i++)
			psi[i] = psi[i]/XNODES;
		// nonlinear
		nonlin(psi, DELTAT);
		// forward transform
		fftw_execute(forward);
		// linear
		lin(psi, k2, DELTAT/2);
		// backward tranform
		fftw_execute(backward);
		// scale down
		for(int i = 0; i < XNODES; i++)
			psi[i] = psi[i]/XNODES;
	}
	rtn=gettimeofday(&tp, NULL);
	t2=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	elapsed=t2-t1;
	
	FILE *time_file;
	time_file = fopen("cpu_time.txt", "a"); 
	fprintf(time_file, "%f, ", elapsed);
	fclose(time_file);

	matlab_plot(psi_0, psi);

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
	for(int i = 0; i < XNODES; i++)
	{
    	psi[i] = cexp(I * cabs(psi[i]) * cabs(psi[i]) * dt)*psi[i];
	}
}

void lin(fftw_complex *psi, double *k2, double dt)
{                  
	for(int i = 0; i < XNODES; i++)
	{
    	psi[i] = cexp(-I * k2[i] * dt)*psi[i];
	}
}

void matlab_plot(fftw_complex *psi_0, fftw_complex *psi)
{
	FILE *matlab_file;
	matlab_file = fopen("plot_CPU_f.m", "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n", -L, L, XNODES);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(psi_0[i]));
	fprintf(matlab_file,"];\n");                                                                 

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(psi[i]));
	fprintf(matlab_file,"];\n");                                                                 
	
	fprintf(matlab_file, "plot(x, psi_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, psi_f, '--b', 'LineWidth', 1);\n"
						 "legend('t = 0', 't = %f', 0);\n"
						 "title('Soliton Solution for CPU');\n"
						 "xlabel('x values'); ylabel('|psi|');", DELTAT*TNODES);
	fclose(matlab_file);
}

