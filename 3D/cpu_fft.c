/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/timers.h"
#include "../lib/helpers.h"
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// Grid parameters
#define XN	32						// Number of x-spatial nodes        
#define YN	32						// Number of y-spatial nodes          
#define ZN	32						// Number of z-spatial nodes         
#define TN	1000					// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	10.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Timing parameters
#define IRVL	10				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0  "cpu_fft_0.vtk"
#define VTK_1  "cpu_fft_1.vtk"
#define TIME_F "cpu_fft_time.m"

// Index flattening macro                                                    
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
void nonlin(fftw_complex *psi, double dt, int xn, int yn, int zn);                                
void lin(fftw_complex *psi, double *k2, double dt, int xn, int yn, int zn);                       
void normalize(fftw_complex *psi, int size);                              

int main(void)
{                                                                          
	// Timing starts here
	double t1 = get_cpu_time();
	
	// Print basic info about simulation
	printf("XN: %d. DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
	fftw_complex *psi = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN * ZN);
	fftw_complex *psi_0 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN * ZN);
	double *kx = (double*)malloc(XN * sizeof(double));
	double *ky = (double*)malloc(YN * sizeof(double));
	double *kz = (double*)malloc(ZN * sizeof(double));
    double *x = (double*)malloc(sizeof(double) * XN);
	double *y = (double*)malloc(sizeof(double) * YN);
	double *z = (double*)malloc(sizeof(double) * ZN);
	double *k2 = (double*)malloc(sizeof(double) * XN*YN*ZN);
	double *max = (double*)calloc(TN+1, sizeof(double));
	
	// Create transform plans
	fftw_plan forward, backward;
	forward = fftw_plan_dft_3d(XN, YN, ZN, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_3d(XN, YN, ZN, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Create wave numbers
	double dkx = 2*M_PI/XN/DX;
	for(int i = XN/2; i >= 0; i--) 
		kx[XN/2 - i]=(XN/2 - i) * dkx;
	for(int i = XN/2+1; i < XN; i++) 
		kx[i]=(i - XN) * dkx; 

	double dky = 2*M_PI/YN/DY;
	for(int i = YN/2; i >= 0; i--) 
		ky[YN/2 - i]=(YN/2 - i) * dky;
	for(int i = YN/2+1; i < YN; i++) 
		ky[i]=(i - YN) * dky; 

	double dkz = 2*M_PI/ZN/DZ;
	for(int i = ZN/2; i >= 0; i--) 
		kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		kz[i]=(i - ZN) * dkz; 
	
	// Initialize x, y and z
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

	for(int i = 0; i < ZN ; i++)
		z[i] = (i-ZN/2)*DZ;
     
	// Initial conditions
	for(int i = 0; i < XN; i++)
		for(int j = 0; j < YN; j++)
			for(int k = 0; k < ZN; k++)
			{
				psi[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
													/(2*R*R*R_S*R_S)) + 0*I; 
				psi_0[ind(i,j,k)] = psi[ind(i,j,k)];
				k2[ind(i,j,k)] = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
			}   
	
	// Print timing info to file
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	#if MAX_PSI_CHECKING
	cmax_psi(psi, max, 0, XN*YN*ZN);
	#endif // MAX_PSI_CHECKING
	
	// Forward transform
	fftw_execute(forward);
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		lin(psi, k2, DT/2, XN, YN, ZN);  
		// Backward transform
		fftw_execute(backward);
		// Normalize the transform
		normalize(psi, XN*YN*ZN);
		// Solve nonlinear part
		nonlin(psi, DT, XN, YN, ZN);  
		// Forward transform
		fftw_execute(forward);
		// Solve linear part
		lin(psi, k2, DT/2, XN, YN, ZN);  
		// Save max |psi| for printing
		#if MAX_PSI_CHECKING
		cmax_psi(psi, max, 0, XN*YN*ZN);
		#endif // MAX_PSI_CHECKING
		// Print tiem at specific intervals
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
	normalize(psi, XN*YN*ZN);

	// Plot results
 	vtk_3dc(x, y, z, psi, XN, YN, ZN, VTK_1);
	vtk_3dc(x, y, z, psi_0, XN, YN, ZN, VTK_0);

	// Clean up
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_free(psi_0); 
	fftw_free(psi);
	free(x);
	free(y);
	free(z);
	free(k2);
	free(kx);
	free(ky);
	free(kz);
	free(max);

	return 0;
}

void lin(fftw_complex *psi, double *k2, double dt, int xn, int yn, int zn)
{                  
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
		for(int j = 0; j < yn; j++)
			for(int k = 0; k < zn; k++)
				psi[ind(i,j,k)] = cexp(-I * k2[ind(i,j,k)] * dt)*psi[ind(i,j,k)];
}

void nonlin(fftw_complex *psi, double dt, int xn, int yn, int zn)
{                  
	double psi2;
	// Avoid first and last point (boundary conditions) (needs fixing)
	for(int i = 0; i < xn; i++)
		for(int j = 0; j < yn; j++)
			for(int k = 0; k < zn; k++)
			{
				psi2 = cabs(psi[ind(i,j,k)])*cabs(psi[ind(i,j,k)]);
				psi[ind(i,j,k)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j,k)];
			}
}

void normalize(fftw_complex *psi, int size)
{
		for(int i = 0; i < size; i++)
			psi[i] = psi[i]/size;
}
