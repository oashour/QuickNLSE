/**********************************************************************************
* Numerical Solution for the Cubic-Quintic Nonlinear Schrodinger Equation         *
* in (2+1)D	 using explicit FDTD with second order splitting.                     *           *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/helpers.h"
#include <mpi.h>

#define ROOT 0
#define TAG1 1
#define TAG2 2

// Define message sending tags and MPI root
#define ROOT 0					// Root process rank
#define TAG1 1                  // Send message right
#define TAG2 2                  // Send message left

// Grid Parameters
#define XN	 64					// Number of spatial nodes
#define YN	 64					// Number of spatial nodes
#define TN	 100000  			// Number of temporal nodes
#define LX	 50.0				// Spatial Period [-LX,LX)
#define LY	 50.0				// Spatial Period [-LY,LY)
#define TT	 10.0               // Max time
#define DX	 (2*LX / XN)   		// x-spatial step size
#define DY	 (2*LY / YN)   		// y-spatial step size
#define DT	 (TT / TN)			// Temporal step size

// Gaussian Parameters                      
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R	 	(1.0/(A*sqrt(1.0-A*A)))

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Output files
#define PLOT_F "mpi_fdtd_plot.m"
#define TIME_F "mpi_fdtd_time.m"

// Index linearization for kernels [x,y] = [x * width + y] 
#define ind(i,j)  ((i)*XN+(j))		//[i  ,j  ] 

// Function prototypes
void Re_lin(double *Re, double *Im, double dt, int xn_loc, int yn, 
										double dx, double dy, int rank, int p);
void Im_lin(double *Re, double *Im, double dt, int xn_loc, int yn, 
										double dx, double dy, int rank, int p);
void nonlin(double *Re, double *Im, double dt, int xn_loc, int yn, int rank, int p);
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, int yn, MPI_Status *status);
void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, int yn, MPI_Status *status);
void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, int yn, MPI_Status *status);

int main(int argc, char** argv)
{

	// MPI set up
	int rank, p;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	// Timing starts here
	double t1 = MPI_Wtime();

    // Print basic info about simulation
	if(rank == ROOT)
		printf("XN: %d, DX: %f, DT: %f, dt/dx^2: %f\n", XN, DX, DT, DT/(DX*DX));

	// Allocate the arrays
	int xn_loc = XN/p;
	double *Re_0, *Im_0, *Re_new, *Im_new;
   	double *Re = (double*)malloc(sizeof(double) * (xn_loc+2)*YN);
   	double *Im = (double*)malloc(sizeof(double) * (xn_loc+2)*YN);
    
	// Add ghost values to the arrays
	for(int j = 0; j < YN; j++)
	{
		Re[ind(0,j)] = 0; Im[ind(0,j)] = 0;
		Re[ind(xn_loc+1,j)] = 0; Im[ind(xn_loc+1,j)] = 0;
	}
	
	// Initial conditions on root
	if (rank == ROOT)
	{
		double *x = (double*)malloc(sizeof(double) * XN);
    	double *y = (double*)malloc(sizeof(double) * YN);
		Re_0 = (double*)malloc(sizeof(double) * XN*YN);
    	Im_0 = (double*)malloc(sizeof(double) * XN*YN);

		// Initialize x and y.
		for(int i = 0; i < XN; i++)
			x[i] = (i-XN/2)*DX;
			
		for(int i = 0; i < YN ; i++)
			y[i] = (i-YN/2)*DY;

		// Initial Conditions
		for(int i = 0; i < XN; i++)
			for(int j = 0; j < YN; j++)
				{
					Re_0[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])
											/(2*R*R*R_S*R_S)); 
					Im_0[ind(i,j)] = 0;
				}
		
		free(x); free(y);
	}
	// Scatter the initial array to divide among processes
	// Starts from second row in local arrays, 1st is reserved for syncing
	MPI_Scatter(&Re_0[ind(0,0)], xn_loc*YN, MPI_DOUBLE, &Re[ind(1,0)], xn_loc*YN,
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Im_0[ind(0,0)], xn_loc*YN, MPI_DOUBLE, &Im[ind(1,0)], xn_loc*YN, 
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	// Sync between nodes in preparation for time evolution
	syncImRe(rank, p, xn_loc, Re, Im, YN, &status);
	
	// Print timing info to file
	FILE *fp;
	if (rank == ROOT)
	{
		fp = fopen(TIME_F, "w");
		fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
		fprintf(fp, "time = [0, ");
	}

	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, xn_loc, YN, DX, DY, rank, p);
		syncRe(rank, p, xn_loc, Re, Im, YN, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc, YN, DX, DY, rank, p);
		syncIm(rank, p, xn_loc, Re, Im, YN, &status);
		// Solve nonlinear part and sync
		nonlin(Re, Im, DT, xn_loc, YN, rank, p);
		syncImRe(rank, p, xn_loc, Re, Im, YN, &status);
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, xn_loc, YN, DX, DY, rank, p);
		syncRe(rank, p, xn_loc, Re, Im, YN, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc, YN, DX, DY, rank, p);
		syncIm(rank, p, xn_loc, Re, Im, YN, &status);
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
		Re_new = (double*)malloc(sizeof(double) * XN*YN);
		Im_new = (double*)malloc(sizeof(double) * XN*YN);
	}

	// Gather results into new array
    MPI_Gather(&Re[ind(1,0)], xn_loc*YN, MPI_DOUBLE, &Re_new[ind(0,0)], 
										xn_loc*YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&Im[ind(1,0)], xn_loc*YN, MPI_DOUBLE, &Im_new[ind(0,0)], 
										xn_loc*YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Plot results
	if(rank == ROOT)
	{
		double *max = calloc(TN, sizeof(double));
		m_plot_2d(Re_0, Im_0, Re_new, Im_new, max, LX, LY, XN, YN, TN, PLOT_F);

		free(Re_0); free(Im_0); free(Re_new); free(Im_new); free(max);
	}

	// Clean up
	free(Re); 
	free(Im); 
	
    MPI_Finalize();

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn_loc, int yn, 
												double dx, double dy, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < xn_loc+1; i++) 
		for(int j = 1; j < yn-1; j++)
		{
			// Avoid boundary conditions
			if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) continue;	

		    Re[ind(i,j)] = Re[ind(i,j)] 
						   - dt/(dx*dx)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
						   - dt/(dy*dy)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
		}
}

void Im_lin(double *Re, double *Im, double dt, int xn_loc, int yn, 
												double dx, double dy, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < xn_loc+1; i++) 
		for(int j = 1; j < yn-1; j++) 
		{
			// Avoid boundary conditions
			if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) continue;	
			
			Im[ind(i,j)] = Im[ind(i,j)] 
							+ dt/(dx*dx)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
							+ dt/(dy*dy)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
		}
}

void nonlin(double *Re, double *Im, double dt, int xn_loc, int yn, int rank, int p)
{                  
	double Rp, Ip, A2;
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < xn_loc+1; i++) 
		for(int j = 1; j < yn-1; j++) 
		{
			// Avoid boundary conditions
			if(((i == 1) && (rank == ROOT)) || ((i == xn_loc) && (rank == p-1))) continue;	
		
			Rp = Re[ind(i,j)]; 
			Ip = Im[ind(i,j)];
			A2 = Rp*Rp+Ip*Ip; 
			
			Re[ind(i,j)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
			Im[ind(i,j)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
		}
}

void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, 
													int yn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0)], yn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0)], yn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0)], yn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0)], yn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0)], yn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0)], yn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0)], yn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0)], yn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, 
													int yn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], yn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], yn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], yn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], yn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Receive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], yn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], yn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], yn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], yn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, 
														int yn, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], yn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0)], yn, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], yn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0)], yn, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], yn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0)], yn, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], yn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0)], yn, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
	}
	else
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], yn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0)], yn, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], yn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0)], yn, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], yn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0)], yn, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], yn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0)], yn, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

