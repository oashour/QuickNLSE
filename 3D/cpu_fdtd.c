/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation in      *
* (3+1)D using explicit FDTD with second order splitting.                         *   
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
**********************************************************************************/
#include "../lib/helpers.h"
#include "../lib/timers.h"

#define M_PI 3.14159265358979323846264338327

// Grid Parameters
#define XN	32						// Number of x-spatial nodes        
#define YN	32						// Number of y-spatial nodes          
#define ZN	32						// Number of z-spatial nodes         
#define TN	1000  					// Number of temporal nodes          
#define LX	50.0					// x-spatial domain [-LX,LX)         
#define LY	50.0					// y-spatial domain [-LY,LY)         
#define LZ	50.0					// z-spatial domain [-LZ,LZ)         
#define TT	10.0            		// Max time                          
#define DX	(2*LX / XN)				// x-spatial step size               
#define DY	(2*LY / YN)				// y-spatial step size
#define DZ	(2*LZ / ZN)				// z-spatial step size
#define DT	(TT / TN)    			// temporal step size

// Gaussian parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Timing parameters
#define IRVL	10				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0  "cpu_fdtd_0.vtk"
#define VTK_1  "cpu_fdtd_1.vtk"
#define TIME_F "cpu_fdtd_time.m"

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
void Re_lin(double *Re, double *Im, double dt, int xn, int yn, int zn,
													double dx, double dy, double dz);
void Im_lin(double *Re, double *Im, double dt, int xn, int yn, int zn,             
												 	double dx, double dy, double dz);
void nonlin(double *Re, double *Im, double dt, int xn, int yn, int zn);  

int main(void)
{
	// Timing starts here
	double t1 = get_cpu_time();
    
	// Print basic info about simulation
	printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));
	
	// Allocate the arrays
	double *x = (double*)malloc(sizeof(double) * XN);
	double *y = (double*)malloc(sizeof(double) * YN);
	double *z = (double*)malloc(sizeof(double) * YN);
	double *max = (double*)calloc(TN+1, sizeof(double));
	double *Re = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *Im = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	double *Re_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);
    double *Im_0 = (double*)malloc(sizeof(double) * XN * YN * ZN);   
	
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
				Re[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
																/(2*R*R*R_S*R_S)); 
				Im[ind(i,j,k)] = 0;
				Re_0[ind(i,j,k)] = Re[ind(i,j,k)];
				Im_0[ind(i,j,k)] = Im[ind(i,j,k)];
			}
	
	// Print timing info to file
	FILE *fp = fopen(TIME_F, "w");
	fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
	fprintf(fp, "time = [0, ");
	
	// Save max |psi| for printing
	#if MAX_PSI_CHECKING
	max_psi(Re, Im, max, 0, XN*YN*ZN);
	#endif // MAX_PSI_CHECKING

	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, YN, ZN, DX, DY, DZ);
		Im_lin(Re, Im, DT*0.5, XN, YN, ZN, DX, DY, DZ);
		// Solve nonlinear part
		nonlin(Re, Im, DT*0.5, XN, YN, ZN);
		// Solve linear part
		Re_lin(Re, Im, DT*0.5, XN, YN, ZN, DX, DY, DZ);
		Im_lin(Re, Im, DT*0.5, XN, YN, ZN, DX, DY, DZ);
		// Save max |psi| for printing
		#if MAX_PSI_CHECKING
		max_psi(Re, Im, max, 0, XN*YN*ZN);
		#endif // MAX_PSI_CHECKING
		// Print time at specific intervals
		if(i % IRVL == 0)
			fprintf(fp, "%f, ", get_cpu_time()-t1);
	}
	// Wrap up timing file
	fprintf(fp, "];\n");
	fprintf(fp, "plot(steps, time, '-*r');\n");
	fclose(fp);
    
	// Plot results
	vtk_3d(x, y, z, Re_0, Im_0, XN, YN, ZN, VTK_0);
	vtk_3d(x, y, z, Re, Im, XN, YN, ZN, VTK_1);
	
	// Clean up
	free(Re);
	free(Im); 
	free(Re_0); 
	free(Im_0); 
	free(x); 
	free(y);
	free(z);
	free(max);

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn, int yn, int zn,
														double dx, double dy, double dz)
{                  
	// Avoid first and last point (boundary conditions)
    for(int k = 1; k < zn - 1; k++)
    	for(int j = 1; j < yn - 1; j++)
			for(int i = 1; i < xn - 1; i++)
				Re[ind(i,j,k)] = Re[ind(i,j,k)] 
				- dt/(dx*dx)*(Im[ind(i+1,j,k)] - 2*Im[ind(i,j,k)] + Im[ind(i-1,j,k)])
				- dt/(dy*dy)*(Im[ind(i,j+1,k)] - 2*Im[ind(i,j,k)] + Im[ind(i,j-1,k)])
				- dt/(dz*dz)*(Im[ind(i,j,k+1)] - 2*Im[ind(i,j,k)] + Im[ind(i,j,k-1)]);
}

void Im_lin(double *Re, double *Im, double dt, int xn, int yn, int zn,
														double dx, double dy, double dz)
{                  
	// Avoid first and last point (boundary conditions)
    for(int k = 1; k < zn - 1; k++)
    	for(int j = 1; j < yn - 1; j++)
			for(int i = 1; i < xn - 1; i++)
				Im[ind(i,j,k)] = Im[ind(i,j,k)] 
				+ dt/(dx*dx)*(Re[ind(i+1,j,k)] - 2*Re[ind(i,j,k)] + Re[ind(i-1,j,k)])
				+ dt/(dy*dy)*(Re[ind(i,j+1,k)] - 2*Re[ind(i,j,k)] + Re[ind(i,j-1,k)])
				+ dt/(dz*dz)*(Re[ind(i,j,k+1)] - 2*Re[ind(i,j,k)] + Re[ind(i,j,k-1)]);
}

void nonlin(double *Re, double *Im, double dt, int xn, int yn, int zn)
{                  
	double Rp, Ip, A2;
	// Avoid first and last point (boundary conditions)
    for(int k = 1; k < zn - 1; k++)
    	for(int j = 1; j < yn - 1; j++)
			for(int i = 1; i < xn - 1; i++)
			{
				Rp = Re[ind(i,j,k)];  Ip = Im[ind(i,j,k)];
				A2 = Rp*Rp+Ip*Ip; 
				
				Re[ind(i,j,k)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
				Im[ind(i,j,k)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
			}
}

