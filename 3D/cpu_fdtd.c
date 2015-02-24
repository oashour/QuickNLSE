 // Cubic Quintic Nonlinear Schrodinger Equation
#include <sys/time.h>
#include <stddef.h>
#include "../lib/helpers.h"
#include <fftw3.h>

#define M_PI 3.14159265358979323846264338327

// Grid Parameters
#define XN	32						// Number of x-spatial nodes        
#define YN	32						// Number of y-spatial nodes          
#define ZN	32						// Number of z-spatial nodes         
#define TN	100						// Number of temporal nodes          
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
void Re_lin(double *Re, double *Im, double dt);
void Im_lin(double *Re, double *Im, double dt);
void nonlin(double *Re, double *Im, double dt);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));
	
    // Allocate x, y, Re and Im on host. Max will be use to store max |psi|
	// Re_0 and Im_0 will keep a copy of initial pulse for printing
	double *x = (double*)malloc(sizeof(double) * XN);
	double *y = (double*)malloc(sizeof(double) * YN);
	double *z = (double*)malloc(sizeof(double) * YN);
	double *max = (double*)malloc(sizeof(double) * TN);
	double *Re = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *Im = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	double *Re_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *Im_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	
	// initialize x and y.
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

    for(int i = 0; i < YN ; i++)
		z[i] = (i-ZN/2)*DZ; 
    
	// Initial Conditions
    for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				Re[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
																/(2*R*R*R_S*R_S)); 
				Im[ind(i,j,k)] = 0;
				Re_0[ind(i,j,k)] = Re[ind(i,j,k)];
				Im_0[ind(i,j,k)] = Im[ind(i,j,k)];
			}
	
	// print max |psi| for initial conditions
	max_psi(Re, Im, max, 0, XN*YN*ZN);
	// Begin timing
	for (int i = 1; i < TN; i++)
	{
		// linear
		Re_lin(Re, Im, DT*0.5);
        Im_lin(Re, Im, DT*0.5);
		// nonlinear
		nonlin(Re, Im, DT);
		// linear
		Re_lin(Re, Im, DT*0.5);
        Im_lin(Re, Im, DT*0.5);
		// find max psi
		max_psi(Re, Im, max, i, XN*YN*ZN);
	}

    double *psi2 = (double*)malloc(sizeof(double)*XN*YN*ZN);
    double *psi2_0 = (double*)malloc(sizeof(double)*XN*YN*ZN);
	
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
	   		for(int i = 0; i < XN; i++)
			{
				psi2[ind(i,j,k)] = sqrt(Re[ind(i,j,k)]*Re[ind(i,j,k)] +
									   Im[ind(i,j,k)]*Im[ind(i,j,k)]);
				psi2_0[ind(i,j,k)] = sqrt(Re_0[ind(i,j,k)]*Re_0[ind(i,j,k)] +
									   Im_0[ind(i,j,k)]*Im_0[ind(i,j,k)]);
            }
	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	vtk_3d(x, y, z, psi2, XN, YN, ZN, "test1.vtk");
	vtk_3d(x, y, z, psi2_0, XN, YN, ZN, "test0.vtk");
	
	// wrap up                                                  
	free(Re); 
	free(Im); 
	free(Re_0); 
	free(Im_0); 
	free(x); 
	free(y);
	free(max);
	free(psi2);

	return 0;
}

void Re_lin(double *Re, double *Im, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int k = 1; k < ZN - 1; k++)
    	for(int j = 1; j < YN - 1; j++)
			for(int i = 1; i < XN - 1; i++)
				Re[ind(i,j,k)] = Re[ind(i,j,k)] 
				- dt/(DX*DX)*(Im[ind(i+1,j,k)] - 2*Im[ind(i,j,k)] + Im[ind(i-1,j,k)])
				- dt/(DY*DY)*(Im[ind(i,j+1,k)] - 2*Im[ind(i,j,k)] + Im[ind(i,j-1,k)])
				- dt/(DZ*DZ)*(Im[ind(i,j,k+1)] - 2*Im[ind(i,j,k)] + Im[ind(i,j,k-1)]);
}

void Im_lin(double *Re, double *Im, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int k = 1; k < ZN - 1; k++)
    	for(int j = 1; j < YN - 1; j++)
			for(int i = 1; i < XN - 1; i++)
				Im[ind(i,j,k)] = Im[ind(i,j,k)] 
				+ dt/(DX*DX)*(Re[ind(i+1,j,k)] - 2*Re[ind(i,j,k)] + Re[ind(i-1,j,k)])
				+ dt/(DY*DY)*(Re[ind(i,j+1,k)] - 2*Re[ind(i,j,k)] + Re[ind(i,j-1,k)])
				+ dt/(DZ*DZ)*(Re[ind(i,j,k+1)] - 2*Re[ind(i,j,k)] + Re[ind(i,j,k-1)]);
}

void nonlin(double *Re, double *Im, double dt)
{                  
	double Rp, Ip, A2;
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int k = 1; k < ZN - 1; k++)
    	for(int j = 1; j < YN - 1; j++)
			for(int i = 1; i < XN - 1; i++)
			{
				Rp = Re[ind(i,j,k)];  Ip = Im[ind(i,j,k)];
				A2 = Rp*Rp+Ip*Ip; // |psi|^2
				
				Re[ind(i,j,k)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
				Im[ind(i,j,k)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
			}
}

