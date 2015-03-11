/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* in (3+1)D	 using explicit FDTD with second order splitting.                     *           *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/helpers.h"
#include <mpi.h>

// Define message sending tags and MPI root
#define ROOT 0					// Root process rank
#define TAG1 1                  // Send message right
#define TAG2 2                  // Send message left

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

// Gaussian Parameters                                     
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 		(1.0/(A*sqrt(1.0-A*A)))   
                                                                          
// Timing parameters
#define IRVL	10				// Timing interval. Take a reading every N iterations.

// Output files
#define VTK_0 "mpi_fdtd_0.vtk"
#define VTK_1 "mpi_fdtd_1.vtk"
#define TIME_F "mpi_fdtd_time.m"

// Index flattening                                                   
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]                  
#define ind(i,j,k) (((((i) * ZN) * YN) + ((j) * YN)) + (k))
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
void Re_lin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, double dx,
												double dy, double dz, int rank, int p);
void Im_lin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, double dx,
												double dy, double dz, int rank, int p);
void nonlin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, 
																	  int rank, int p);
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, 
								   				   int yn, int zn, MPI_Status *status);
void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, 
								   				   int yn, int zn, MPI_Status *status);
void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, 
								   				   int yn, int zn, MPI_Status *status);

int main(int argc, char *argv[])
{
	// MPI set up
	int rank, p;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
    // Print basic info about simulation
	if(rank == ROOT)
		printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
	int xn_loc = XN/p;
	double *x, *y, *z, *Re_0, *Im_0, *Re_new, *Im_new;
   	double *Re = (double*)malloc(sizeof(double) * (xn_loc+2)*YN*ZN);
   	double *Im = (double*)malloc(sizeof(double) * (xn_loc+2)*YN*ZN);

	// Add ghost values to the arrays
	for(int k = 0; k < ZN; k++)
		for(int j = 0; j < YN; j++)
		{
			Re[ind(0,j,k)] = 0; Im[ind(0,j,k)] = 0;
			Re[ind(xn_loc+1,j,k)] = 0; Im[ind(xn_loc+1,j,k)] = 0;
		}

	// Initial conditions on root
	if (rank == ROOT)
	{
		x = (double*)malloc(sizeof(double) * XN);
    	y = (double*)malloc(sizeof(double) * YN);
    	z = (double*)malloc(sizeof(double) * ZN);
		Re_0 = (double*)malloc(sizeof(double) * XN*YN*ZN);
    	Im_0 = (double*)malloc(sizeof(double) * XN*YN*ZN);

		// Initialize x, y and z
		for(int i = 0; i < XN; i++)
			x[i] = (i-XN/2)*DX;
			
		for(int i = 0; i < YN ; i++)
			y[i] = (i-YN/2)*DY;

		for(int i = 0; i < ZN ; i++)
			z[i] = (i-ZN/2)*DZ;
		
		// Initial Conditions
		for(int k = 0; k < ZN; k++)
			for(int j = 0; j < YN; j++)
				for(int i = 0; i < XN; i++)
				{
					Re_0[ind(i,j,k)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
											/(2*R*R*R_S*R_S)); 
					Im_0[ind(i,j,k)] = 0;
				}
	}
	// Scatter the initial array to divide among processes
	// Starts from second row in local arrays, 1st is reserved for syncing
	MPI_Scatter(&Re_0[ind(0,0,0)], xn_loc*YN*ZN, MPI_DOUBLE, &Re[ind(1,0,0)], xn_loc*YN*ZN,
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Im_0[ind(0,0,0)], xn_loc*YN*ZN, MPI_DOUBLE, &Im[ind(1,0,0)], xn_loc*YN*ZN, 
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	// Sync between nodes in preparation for time evolution
	syncImRe(rank, p, xn_loc, Re, Im, YN, ZN, &status);
	
	// Print timing info to file
	FILE *fp;
	if (rank == ROOT)
	{
		fp = fopen(TIME_F, "w");
		fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
		fprintf(fp, "time = [0, ");
	}

	// Timing starts here
	double t1 = MPI_Wtime();
	
	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, xn_loc, YN, ZN, DX, DY, DZ, rank, p);
		syncRe(rank, p, xn_loc, Re, Im, YN, ZN, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc, YN, ZN, DX, DY, DZ, rank, p);
		syncIm(rank, p, xn_loc, Re, Im, YN, ZN, &status);
		// Solve nonlinear partand sync
		nonlin(Re, Im, DT*0.5, xn_loc, YN, ZN, rank, p);
		syncImRe(rank, p, xn_loc, Re, Im, YN, ZN, &status);
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, xn_loc, YN, ZN, DX, DY, DZ, rank, p);
		syncRe(rank, p, xn_loc, Re, Im, YN, ZN, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc, YN, ZN, DX, DY, DZ, rank, p);
		syncIm(rank, p, xn_loc, Re, Im, YN, ZN, &status);
		if (rank == ROOT)
			if (i % IRVL == 0)
				fprintf(fp, "%f, ", MPI_Wtime() -t1);
	}
	// Wrap up timing file
	if (rank == ROOT)
	{
		fprintf(fp, "];\n");
		fprintf(fp, "plot(steps, time, '-*r');\n");
		fclose(fp);

		// Prepare receiving arrays for final results
		Re_new = (double*)malloc(sizeof(double) * XN*YN*ZN);
		Im_new = (double*)malloc(sizeof(double) * XN*YN*ZN);
	}

	// Gather results into new array
    MPI_Gather(&Re[ind(1,0,0)], xn_loc*YN*ZN, MPI_DOUBLE, &Re_new[ind(0,0,0)], 
										xn_loc*YN*ZN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&Im[ind(1,0,0)], xn_loc*YN*ZN, MPI_DOUBLE, &Im_new[ind(0,0,0)], 
										xn_loc*YN*ZN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Plot results
	if(rank == ROOT)
	{
		vtk_3d(x, y, z, Re_new, Im_new, XN, YN, ZN, VTK_1);
		vtk_3d(x, y, z, Re_0, Im_0, XN, YN, ZN, VTK_0);
		
		free(x); free(y); free(z);	
		free(Re_0); free(Im_0); free(Re_new); free(Im_new);
	}
	
	// Clean up
	free(Re); 
	free(Im); 
	
	MPI_Finalize();

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, double dx,
												double dy, double dz, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < xn_loc+1; i++)  
		for(int j = 1; j < yn-1; j++)
			for(int k = 1; k < zn-1; k++)
			{
				// Avoid boundary conditions
				if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) 
					continue;	
				
				Re[ind(i,j,k)] = Re[ind(i,j,k)] 
				- dt/(dx*dx)*(Im[ind(i+1,j,k)] - 2*Im[ind(i,j,k)] + Im[ind(i-1,j,k)])
				- dt/(dy*dy)*(Im[ind(i,j+1,k)] - 2*Im[ind(i,j,k)] + Im[ind(i,j-1,k)])
				- dt/(dz*dz)*(Im[ind(i,j,k+1)] - 2*Im[ind(i,j,k)] + Im[ind(i,j,k-1)]);
			}
}

void Im_lin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, double dx,
												double dy, double dz, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < xn_loc+1; i++) 
		for(int j = 1; j < yn-1; j++) 
			for(int k = 1; k < zn-1; k++) 
			{
				// Avoid boundary conditions
				if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) 
					continue;	
				
				Im[ind(i,j,k)] = Im[ind(i,j,k)] 
				+ dt/(dx*dx)*(Re[ind(i+1,j,k)] - 2*Re[ind(i,j,k)] + Re[ind(i-1,j,k)])
				+ dt/(dy*dy)*(Re[ind(i,j+1,k)] - 2*Re[ind(i,j,k)] + Re[ind(i,j-1,k)])
				+ dt/(dz*dz)*(Re[ind(i,j,k+1)] - 2*Re[ind(i,j,k)] + Re[ind(i,j,k-1)]);
			}
}

void nonlin(double *Re, double *Im, double dt, int xn_loc, int yn, int zn, 
																int rank, int p)
{                  
	double Rp, Ip, A2;
	for(int i = 1; i < xn_loc+1; i++) 
		for(int j = 1; j < yn-1; j++) 
			for(int k = 1; k < zn-1; k++)
			{
				// Avoid boundary conditions
				if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) 
					continue;	
				
				Rp = Re[ind(i,j,k)]; Ip = Im[ind(i,j,k)];
				A2 = Rp*Rp+Ip*Ip;
				
				Re[ind(i,j,k)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
				Im[ind(i,j,k)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
			}
}

void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, 
									int yn, int zn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, 
													int yn, int zn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, 
								   					 int yn, int zn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0,0)], yn*zn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0,0)], yn*zn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0,0)], yn*zn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

