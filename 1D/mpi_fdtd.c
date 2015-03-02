/**********************************************************************************
* Numerical Solution for the Cubic Nonlinear Schrodinger Equation in (1+1)D	  	  *
* using explicit FDTD with second order splitting.                                *
* Coded by: Omar Ashour, Texas A&M University at Qatar, February 2015.    	      *
* ********************************************************************************/
#include "../lib/helpers.h"
#include <mpi.h>

// Define message sending tags and MPI root
#define ROOT 0					// Root process rank
#define TAG1 1                  // Send message right
#define TAG2 2                  // Send message left

// Grid Parameters
#define XN	 1024				// number of spatial ndes
#define TN	 100000  			// number of temporal nodes
#define L	 10.0				// Spatial Period
#define TT	 10.0               // Max time
#define DX	 (2*L / XN)			// spatial step size
#define DT	 (TT / TN)			// temporal step size

// Timing parameters
#define IRVL  100				// Timing interval. Take a reading every N iterations.

// Function Prototypes
void Re_lin(double *Re, double *Im, double dt, double dx, int p_nodes, int rank, int p);
void Im_lin(double *Re, double *Im, double dt, double dx, int p_nodes, int rank, int p);
void nonlin(double *Re, double *Im, double dt, int p_nodes, int rank, int p);
void syncImRe(int rank, int p, int p_nodes, double *Re, double *Im, MPI_Status *status);
void syncIm(int rank, int p, int p_nodes, double *Re, double *Im, MPI_Status *status);
void syncRe(int rank, int p, int p_nodes, double *Re, double *Im,  MPI_Status *status);

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
	int p_nodes = XN/p;
	double *Re_0, *Im_0, *Re_new, *Im_new;
   	double *Re = (double*)malloc(sizeof(double) * (p_nodes+2));
   	double *Im = (double*)malloc(sizeof(double) * (p_nodes+2));
    
	// Add ghost values to the arrays
	Re[0] = 0; Im[0] = 0;
	Re[p_nodes+1] = 0; Im[p_nodes+1] = 0;

	// Initial conditions on root
	if (rank == ROOT)
	{
    	double *x = (double*)malloc(sizeof(double) * XN);
		Re_0 = (double*)malloc(sizeof(double) * XN);
    	Im_0 = (double*)malloc(sizeof(double) * XN);

		for (int i = 0; i < XN; i++)
		{
			x[i] = (i-XN/2)*DX; 
			Re_0[i] = sqrt(2.0)/(cosh(x[i]));  
			Im_0[i] = 0;    				 
		}
		
		free(x);
	}
	// Scatter the initial array to divide among processes
	MPI_Scatter(&Re_0[0], p_nodes, MPI_DOUBLE, &Re[1], p_nodes, 
												MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Im_0[0], p_nodes, MPI_DOUBLE, &Im[1], p_nodes, 
												MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	// Sync between nodes in preparation for time evolution
    syncImRe(rank, p, p_nodes, Re, Im, &status);
	
	// Print timing info to file
	FILE *fp;
	if (rank == ROOT)
	{
		fp = fopen("test_1.m", "w");
		fprintf(fp, "steps = [0:%d:%d];\n", IRVL, TN);
		fprintf(fp, "time = [0, ");
	}

	// Start time evolution
	for (int i = 1; i <= TN; i++)
	{
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, DX, p_nodes, rank, p);
		syncRe(rank, p, p_nodes, Re, Im, &status);
		Im_lin(Re, Im, DT*0.5, DX, p_nodes, rank, p);
		syncIm(rank, p, p_nodes, Re, Im, &status);
		// Solve nonlinear part and sync
		nonlin(Re, Im, DT, p_nodes, rank, p);
		syncImRe(rank, p, p_nodes, Re, Im, &status);
		// Solve linear part and sync
		Re_lin(Re, Im, DT*0.5, DX, p_nodes, rank, p);
		syncRe(rank, p, p_nodes, Re, Im, &status);
		Im_lin(Re, Im, DT*0.5, DX, p_nodes, rank, p);
		syncIm(rank, p, p_nodes, Re, Im, &status);
 		// Print time at specific intervals
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
		Re_new = (double*)malloc(sizeof(double) * XN);
		Im_new = (double*)malloc(sizeof(double) * XN);
	}

	// Gather results into new array
    MPI_Gather(&Re[1], p_nodes, MPI_DOUBLE, &Re_new[0], p_nodes, 
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&Im[1], p_nodes, MPI_DOUBLE, &Im_new[0], p_nodes,
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	// Plot results
	if(rank == ROOT)
	{
		m_plot_1d(Re_0, Im_0, Re_new, Im_new, L, XN, "mpi_new.m");
		free(Re_0); free(Im_0); free(Re_new); free(Im_new); 
	}

	// Clean up 
	free(Re); 
	free(Im); 
	
    MPI_Finalize();

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, double dx, int p_nodes, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < p_nodes+1; i++) 
	{
		// Avoid boundary conditions
		if(((i == 1) && (rank == ROOT)) || ((i == p_nodes) && (rank == p-1))) continue;	
		
		Re[i] = Re[i] - dt/(dx*dx)*(Im[i+1] - 2*Im[i] + Im[i-1]);
	}
}

void Im_lin(double *Re, double *Im, double dt, double dx, int p_nodes, int rank, int p)
{                  
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < p_nodes+1; i++) 
	{
		// Avoid boundary conditions
		if(((i == 1) && (rank == ROOT)) || ((i == p_nodes) && (rank == p-1))) continue;	
		
		Im[i] = Im[i] + dt/(dx*dx)*(Re[i+1] - 2*Re[i] + Re[i-1]);
	}
}

void nonlin(double *Re, double *Im, double dt, int p_nodes, int rank, int p)
{                  
	double Rp, Ip, A2;
	// Avoid first and last point (copied values or ghost zeros for root and last)
	for(int i = 1; i < p_nodes+1; i++) 
	{
		// Avoid boundary conditions
		if(((i == 1) && (rank == ROOT)) || ((i == p_nodes) && (rank == p-1))) continue;	
		
		Rp = Re[i]; 
		Ip = Im[i];
		A2 = Rp*Rp+Ip*Ip;
	
		Re[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
   		Im[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
	}
}

void syncRe(int rank, int p, int p_nodes, double *Re, double *Im, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Send to the left (TAG2)
		MPI_Send(&Re[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Re[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&Re[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Re[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncIm(int rank, int p, int p_nodes, double *Re, double *Im, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send to the right (TAG1)
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Send to the left (TAG2)
		MPI_Send(&Im[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Im[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&Im[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Im[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncImRe(int rank, int p, int p_nodes, double *Re, double *Im, MPI_Status *status)
{
	if(rank == 0)
	{
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Send to the left (TAG2)
		MPI_Send(&Re[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Re[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&Re[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the left (TAG1)
		MPI_Recv(&Re[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

