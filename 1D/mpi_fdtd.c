// nlse (1+1)D     
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<mpi.h>
#define ROOT 0
#define TAG1 1
#define TAG2 2

// given stuff
#define XNODES	1000
#define TNODES	1000000
#define L		10.0
#define TMAX	1.0

// calculated from given
#define DELTAX	(L / (XNODES - 1.0))
#define DELTAT	(TMAX / (TNODES - 1.0))

void R_lin(double *R, double *I, double dt, int p_nodes);
void I_lin(double *R, double *I, double dt, int p_nodes);
void nonlin(double *R, double *I, double dt, int p_nodes);
void syncIR(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status);
void syncI(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status);
void syncR(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status);
void matlab_plot(double *R_0, double *I_0, double *R_new, double *I_new);

int main(int argc, char** argv)
{

	int rank, p;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	// allocate and initialize the arrays
    if(rank == ROOT)
		printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));

	int p_nodes = XNODES/p;
	double R_1, R_n;
	double *R_0, *I_0, *R_new, *I_new;
   	double *R = (double*)malloc(sizeof(double) * (p_nodes+2));
   	double *I = (double*)malloc(sizeof(double) * (p_nodes+2));
    R[0] = 0; I[0] = 0;
	R[p_nodes+1] = 0; I[p_nodes+1] = 0;

	if (rank == ROOT)
	{
    	double *x = (double*)malloc(sizeof(double) * XNODES);
		R_0 = (double*)malloc(sizeof(double) * XNODES);
    	I_0 = (double*)malloc(sizeof(double) * XNODES);

		for (int i = 0; i < XNODES; i++)
		{
			x[i] = (i-XNODES/2)*DELTAX; 
			R_0[i] = sqrt(2.0)/(cosh(x[i]));  
			I_0[i] = 0;    				 
		}
		
		free(x);
        R_1 = R_0[0];
		R_n = R_0[XNODES-1];
	}
	MPI_Bcast(&R_1, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&R_n, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(R_0, p_nodes, MPI_DOUBLE, R+1, p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(I_0, p_nodes, MPI_DOUBLE, I+1, p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    syncIR(rank, p, p_nodes, R, I, R_1, R_n, &status);
	
	double timei = MPI_Wtime();
	for (int i = 1; i < TNODES; i++)
	{
		// linear
		R_lin(R, I, DELTAT*0.5, p_nodes);
		syncR(rank, p, p_nodes, R, I, R_1, R_n, &status);
		I_lin(R, I, DELTAT*0.5, p_nodes);
		syncI(rank, p, p_nodes, R, I, R_1, R_n, &status);
		// nonlinear
		nonlin(R, I, DELTAT, p_nodes);
		syncIR(rank, p, p_nodes, R, I, R_1, R_n, &status);
		// linear
		R_lin(R, I, DELTAT*0.5, p_nodes);
		syncR(rank, p, p_nodes, R, I, R_1, R_n, &status);
		I_lin(R, I, DELTAT*0.5, p_nodes);
		syncI(rank, p, p_nodes, R, I, R_1, R_n, &status);
	}
	double timef = MPI_Wtime();
	double wtime = timef-timei;
	
    if (rank == ROOT)
	{
		FILE *time_file;
		time_file = fopen("mpi_time.txt", "a"); 
		fprintf(time_file, "%f, ", wtime);
		fclose(time_file);
	}

    if(rank == ROOT)
	{
	   R_new = (double*)malloc(sizeof(double) * XNODES);
	   I_new = (double*)malloc(sizeof(double) * XNODES);
	}

    MPI_Gather(&R[1], p_nodes, MPI_DOUBLE, &R_new[0], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&I[1], p_nodes, MPI_DOUBLE, &I_new[0], p_nodes, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	if(rank == ROOT)
	{
		matlab_plot(R_0, I_0, R_new, I_new);
						
		free(R_0); free(I_0); free(R_new); free(I_new);
	}

	free(R); 
	free(I); 
	
    MPI_Finalize();

	return 0;
}

void R_lin(double *R, double *I, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // modify bounds
		R[i] = R[i] - dt/(DELTAX*DELTAX)*(I[i+1] - 2*I[i] + I[i-1]);
}

void I_lin(double *R, double *I, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // modify bounds
		I[i] = I[i] + dt/(DELTAX*DELTAX)*(R[i+1] - 2*R[i] + R[i-1]);
}

void nonlin(double *R, double *I, double dt, int p_nodes)
{                  
	for(int i = 1; i < p_nodes+1; i++) // last and first elements are just copied so no need
	{
		double Rp = R[i]; 
		double Ip = I[i];
		double A2 = Rp*Rp+Ip*Ip;
	
		R[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
   		I[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
	}
}

void syncR(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status)
{
	if(rank == 0)
	{
		R[1] = R_1; 
		// Send to the right (TAG1)
		MPI_Send(&R[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from the right (TAG2)
		MPI_Recv(&R[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		R[p_nodes+1] = 0; I[p_nodes+1] = 0;
		R[p_nodes] = R_n; I[p_nodes] = 0;
		// Send to the left (TAG2)
		MPI_Send(&R[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&R[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&R[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&R[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&R[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Receive from the right (TAG2)
		MPI_Recv(&R[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncI(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status)
{
	if(rank == 0)
	{
	    I[1] = 0;
		// Send to the right (TAG1)
		MPI_Send(&I[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from the right (TAG2)
		MPI_Recv(&I[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		I[p_nodes] = 0;
		// Send to the left (TAG2)
		MPI_Send(&I[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&I[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&I[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&I[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&I[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Receive from the right (TAG2)
		MPI_Recv(&I[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncIR(int rank, int p, int p_nodes, double *R, double *I, double R_1, double R_n, MPI_Status *status)
{
	if(rank == 0)
	{
		R[1] = R_1; I[1] = 0;
		// Send to the right (TAG1)
		MPI_Send(&R[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&I[p_nodes], 1, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Receive from the right (TAG2)
		MPI_Recv(&R[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&I[p_nodes+1], 1, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		R[p_nodes] = R_n; I[p_nodes] = 0;
		// Send to the left (TAG2)
		MPI_Send(&R[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		MPI_Send(&I[1], 1, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&R[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&I[0], 1, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
	}
	else
	{
		// Send to the left (TAG2)
		MPI_Send(&R[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		MPI_Send(&I[1], 1, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send to the right (TAG1)
		MPI_Send(&R[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&I[p_nodes], 1, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Receive from the left (TAG1)
		MPI_Recv(&R[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&I[0], 1, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Receive from the right (TAG2)
		MPI_Recv(&R[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&I[p_nodes+1], 1, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

void matlab_plot(double *R_0, double *I_0, double *R_new, double *I_new)
{
	FILE *matlab_file;
	matlab_file = fopen("plot_MPI.m", "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n", -L, L, XNODES);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(R_0[i] * R_0[i] + I_0[i] * I_0[i]));
	fprintf(matlab_file,"];\n");                                                                 

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(R_new[i] * R_new[i] + I_new[i] * I_new[i]));
	fprintf(matlab_file,"];\n");                                                                 
	
	fprintf(matlab_file, "plot(x, psi_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, psi_f, '--b', 'LineWidth', 1);\n"
						 "legend('t = 0', 't = %f', 0);\n"
						 "title('Soliton Solution for MPI');\n"
						 "xlabel('x values'); ylabel('|psi|');", TMAX);
	fclose(matlab_file);
}
