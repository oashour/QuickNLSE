// nlse (1+1)D     
#include "../lib/helpers.h"
#include<mpi.h>

#define ROOT 0
#define TAG1 1
#define TAG2 2

// given stuff
#define XN	1000
#define TN	10000
#define L  	10.0
#define TT	1.0

// calculated from given
#define DX	(2*L / XN)
#define DT	(TT / TN)

void Re_lin(double *Re, double *Im, double dt, int p_nodes);
void Im_lin(double *Re, double *Im, double dt, int p_nodes);
void nonlin(double *Re, double *Im, double dt, int p_nodes);
void syncImRe(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status);
void syncIm(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status);
void syncRe(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status);
//void matlab_plot(double *Re_0, double *Im_0, double *Re_new, double *Im_new);

int main(int argc, char** argv)
{

	int rank, p;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	// allocate and initialize the arrays
    if(rank == ROOT)
		printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));

	int p_nodes = XN/p;
	double Re_1, Re_n;
	double *Re_0, *Im_0, *Re_new, *Im_new;
   	double *Re = (double*)malloc(sizeof(double) * (p_nodes+2));
   	double *Im = (double*)malloc(sizeof(double) * (p_nodes+2));
    Re[0] = 0; Im[0] = 0;
	Re[p_nodes+1] = 0; Im[p_nodes+1] = 0;

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
        Re_1 = Re_0[0];
		Re_n = Re_0[XN-1];
	}
	MPI_Bcast(&Re_1, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&Re_n, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Re_0[0], p_nodes, MPI_DOUBLE, &Re[1], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Im_0[0], p_nodes, MPI_DOUBLE, &Im[1], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    syncImRe(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
	
	double timei = MPI_Wtime();
	for (int i = 1; i < TN; i++)
	{
		// linear
		Re_lin(Re, Im, DT*0.5, p_nodes);
		syncRe(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
		Im_lin(Re, Im, DT*0.5, p_nodes);
		syncIm(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
		// nonlinear
		nonlin(Re, Im, DT, p_nodes);
		syncImRe(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
		// linear
		Re_lin(Re, Im, DT*0.5, p_nodes);
		syncRe(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
		Im_lin(Re, Im, DT*0.5, p_nodes);
		syncIm(rank, p, p_nodes, Re, Im, Re_1, Re_n, &status);
	}
	double timef = MPI_Wtime();
	double wtime = timef-timei;
	
    if (rank == ROOT)
	{
	   printf("%f, ", wtime);
	   Re_new = (double*)malloc(sizeof(double) * XN);
	   Im_new = (double*)malloc(sizeof(double) * XN);
	}

    MPI_Gather(&Re[1], p_nodes, MPI_DOUBLE, &Re_new[0], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&Im[1], p_nodes, MPI_DOUBLE, &Im_new[0], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	if(rank == ROOT)
	{
		m_plot_1d(Re_0, Im_0, Re_new, Im_new, L, XN, "mpi_new.m");
		free(Re_0); free(Im_0); free(Re_new); free(Im_new);
	}

	free(Re); 
	free(Im); 
	
    MPI_Finalize();

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // modify bounds
		Re[i] = Re[i] - dt/(DX*DX)*(Im[i+1] - 2*Im[i] + Im[i-1]);
}

void Im_lin(double *Re, double *Im, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // modify bounds
		Im[i] = Im[i] + dt/(DX*DX)*(Re[i+1] - 2*Re[i] + Re[i-1]);
}

void nonlin(double *Re, double *Im, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // last and first elements are just copied so no need
	{
		double Rep = Re[i]; 
		double Imp = Im[i];
		double A2 = Rep*Rep+Imp*Imp;
	
		Re[i] =	Rep*cos(A2*dt) - Imp*sin(A2*dt);
   		Im[i] =	Rep*sin(A2*dt) + Imp*cos(A2*dt);
	}
}

void syncRe(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status)
{
	if(rank == 0)
	{
		Re[1] = Re_1; 
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		Re[p_nodes+1] = 0; Im[p_nodes+1] = 0;
		Re[p_nodes] = Re_n; Im[p_nodes] = 0;
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
void syncIm(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status)
{
	if(rank == 0)
	{
	    Im[1] = 0;
		// Send to the right (TAG1)
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		Im[p_nodes] = 0;
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
void syncImRe(int rank, int p, int p_nodes, double *Re, double *Im, double Re_1, double Re_n, MPI_Status *status)
{
	if(rank == 0)
	{
		Re[1] = Re_1; Im[1] = 0;
		// Send to the right (TAG1)
		MPI_Send(&Re[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from the right (TAG2)
		MPI_Recv(&Re[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		Re[p_nodes] = Re_n; Im[p_nodes] = 0;
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

