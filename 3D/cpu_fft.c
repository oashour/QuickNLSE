/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* using second order split step Fourier method.                                   *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include <sys/time.h>
#include <stddef.h>
#include "../lib/helpers.h"
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// Grid Parameters
#define XN	256						// Number of x-spatial nodes        
#define YN	256						// Number of y-spatial nodes          
#define ZN	256						// Number of z-spatial nodes         
#define TN	1000					// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	100.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Index linearization                                                    
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
void nonlin(fftw_complex *psi, double dt);                                
void lin(fftw_complex *psi, double *k2, double dt);                       
void normalize(fftw_complex *psi, int size);                              

int main(void)
{                                                                          
    // Timer initialization variables
	double t1,t2,elapsed;
	struct timeval tp;

	// Allocate the arrays
	fftw_complex *psi = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN * ZN);
	fftw_complex *psi_0 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * XN * YN * ZN);
	
	// Create transform plans
	fftw_plan forward, backward;
	forward = fftw_plan_dft_3d(XN, YN, ZN, psi, psi, FFTW_FORWARD, FFTW_ESTIMATE);
	backward = fftw_plan_dft_3d(XN, YN, ZN, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    // X and Y wave numbers
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

	double dkz = 2*M_PI/ZN/DZ;
	double *kz = (double*)malloc(ZN * sizeof(double));
	for(int i = ZN/2; i >= 0; i--) 
		kz[ZN/2 - i]=(ZN/2 - i) * dkz;
	for(int i = ZN/2+1; i < ZN; i++) 
		kz[i]=(i - ZN) * dkz; 
	
	// initialize x and y and z
    double *x = (double*)malloc(sizeof(double) * XN);
	double *y = (double*)malloc(sizeof(double) * YN);
	double *z = (double*)malloc(sizeof(double) * ZN);
	
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
    
	for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

	for(int i = 0; i < ZN ; i++)
		z[i] = (i-ZN/2)*DZ;
     
	// Initial Conditions and square of wave number
	double *k2 = (double*)malloc(sizeof(double) * XN * YN * ZN);
	
	for (int k = 0; k < ZN; k++)
    	for(int j = 0; j < YN; j++)
			for(int i = 0; i < XN; i++)
				{
					psi[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
														/(2*R*R*R_S*R_S)) + 0*I; 
					psi_0[ind(i,j,k)] = psi[ind(i,j,k)];
					k2[ind(i,j,k)] = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
				}   
	
	// Find max(|psi|) for initial pulse.
	double *max = (double*)malloc(sizeof(double) * TN);
	cmax_psi(psi, max, 0, XN*YN*ZN);
	// forward transform
	fftw_execute(forward);
    // Start time evolution and start performance timing
	gettimeofday(&tp, NULL);
	t1=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	for (int i = 1; i < TN; i++)
	{
		// linear calculation
		lin(psi, k2, DT/2);  
		// backward transform
		fftw_execute(backward);
		// normalize the transform
		normalize(psi, XN*YN*ZN);
		// nonlinear calculation
		nonlin(psi, DT);
		// forward transform
		fftw_execute(forward);
		// linear calculation
		lin(psi, k2, DT/2);
		// find maximum |psi|
		cmax_psi(psi, max, i, XN*YN*ZN);
	}
	
	// End of time evolution and end and print performance timing
	gettimeofday(&tp, NULL);
	t2=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	elapsed=t2-t1;
	printf("Tine elsapsed: %f\n", elapsed);
	// backward tranform
	fftw_execute(backward);
	// normalize the transform
	normalize(psi, XN*YN*ZN);

	double *psi2 = (double*)malloc(sizeof(double)*XN*YN*ZN);
    double *psi2_0 = (double*)malloc(sizeof(double)*XN*YN*ZN);
	
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				psi2[ind(i,j,k)] = cabs(psi[ind(i,j,k)]);
				psi2_0[ind(i,j,k)] = cabs(psi_0[ind(i,j,k)]);
            }
	// plot results
 	vtk_3d(x, y, z, psi2, XN, YN, ZN, "test_fft1.vtk");
	vtk_3d(x, y, z, psi2_0, XN, YN, ZN, "test_fft0.vtk");

	// garbage collection
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

// linear function
void lin(fftw_complex *psi, double *k2, double dt)
{                  
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
			for(int i = 0; i < XN; i++)
				psi[ind(i,j,k)] = cexp(-I * k2[ind(i,j,k)] * dt)*psi[ind(i,j,k)];
}

// nonlinear function
void nonlin(fftw_complex *psi, double dt)
{                  
	double psi2;
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
			for(int i = 0; i < XN; i++)
			{
				psi2 = cabs(psi[ind(i,j,k)])*cabs(psi[ind(i,j,k)]);
				psi[ind(i,j,k)] = cexp(I * (psi2-psi2*psi2) * dt) * psi[ind(i,j,k)];
			}
}

// normalization
void normalize(fftw_complex *psi, int size)
{
		for(int i = 0; i < size; i++)
			psi[i] = psi[i]/size;
}
